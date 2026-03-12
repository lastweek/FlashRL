"""Actor model (policy model) wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.device import get_device


@dataclass(frozen=True)
class GeneratedSample:
    """One sampled candidate returned directly by the serving model."""

    text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_token_logprobs: list[float]
    log_prob: float


class ActorModel:
    """Wrapper for the actor (policy) model."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the actor model.

        Args:
            config: Model configuration.
        """
        self.config = config
        self.device = get_device(config.device)
        dtype = getattr(torch, config.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
            device_map=None,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        self.generation_defaults: dict[str, Any] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[GeneratedSample]:
        """Generate one structured sample per prompt."""
        grouped_outputs = self.generate_grouped(prompts, group_size=1, **kwargs)
        return [candidates[0] for candidates in grouped_outputs]

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[GeneratedSample]]:
        """Generate grouped candidates with token ids and per-token logprobs."""
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if not prompts:
            return []

        generation_kwargs = {
            **self.generation_defaults,
            **kwargs,
            "num_return_sequences": group_size,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            prompt_token_ids_by_prompt: list[list[int]] = []
            attention_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            for sample_input_ids, sample_mask in zip(input_ids, attention_mask, strict=True):
                prompt_token_ids_by_prompt.append(
                    sample_input_ids[sample_mask.to(dtype=torch.bool)].tolist()
                )

            transition_scores = self._transition_scores(outputs)
            prompt_width = input_ids.shape[1]
            response_tokens = outputs.sequences[:, prompt_width:]

            grouped_outputs: list[list[GeneratedSample]] = []
            for prompt_index in range(len(prompts)):
                candidates: list[GeneratedSample] = []
                prompt_token_ids = prompt_token_ids_by_prompt[prompt_index]
                for candidate_index in range(group_size):
                    row = prompt_index * group_size + candidate_index
                    response_token_ids, response_token_logprobs = self._trim_generated_response(
                        response_tokens[row].tolist(),
                        transition_scores[row].tolist(),
                    )
                    text = self.tokenizer.batch_decode(
                        [response_token_ids],
                        skip_special_tokens=True,
                    )[0]
                    candidates.append(
                        GeneratedSample(
                            text=text,
                            prompt_token_ids=list(prompt_token_ids),
                            response_token_ids=response_token_ids,
                            response_token_logprobs=response_token_logprobs,
                            log_prob=float(sum(response_token_logprobs)),
                        )
                    )
                grouped_outputs.append(candidates)
            return grouped_outputs
        finally:
            self.tokenizer.padding_side = original_padding_side

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text from prompts."""
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def set_generation_defaults(self, **kwargs: Any) -> None:
        """Set default generation kwargs used when rollout code does not pass them explicitly."""
        self.generation_defaults = dict(kwargs)

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for given inputs and labels."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.logits

    def _transition_scores(self, outputs: Any) -> torch.Tensor:
        """Compute log-prob scores for each generated token."""
        if hasattr(self.model, "compute_transition_scores"):
            scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
            return scores.detach().cpu()

        if not outputs.scores:
            return torch.zeros(
                (outputs.sequences.shape[0], 0),
                dtype=torch.float32,
            )

        transition_scores = []
        generated_tokens = outputs.sequences[:, -len(outputs.scores):]
        for step_index, step_scores in enumerate(outputs.scores):
            token_logprobs = F.log_softmax(step_scores, dim=-1)
            step_ids = generated_tokens[:, step_index].unsqueeze(-1)
            transition_scores.append(
                torch.gather(token_logprobs, dim=-1, index=step_ids).squeeze(-1)
            )
        return torch.stack(transition_scores, dim=1).detach().cpu()

    def _trim_generated_response(
        self,
        response_token_ids: list[int],
        response_token_logprobs: list[float],
    ) -> tuple[list[int], list[float]]:
        """Trim generated response tokens to the true sampled response length."""
        trimmed_token_ids: list[int] = []
        trimmed_logprobs: list[float] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

        for token_id, logprob in zip(response_token_ids, response_token_logprobs, strict=True):
            trimmed_token_ids.append(int(token_id))
            trimmed_logprobs.append(float(logprob))
            if eos_token_id is not None and int(token_id) == int(eos_token_id):
                break

        if eos_token_id is None and pad_token_id is not None:
            while trimmed_token_ids and trimmed_token_ids[-1] == int(pad_token_id):
                trimmed_token_ids.pop()
                trimmed_logprobs.pop()

        return trimmed_token_ids, trimmed_logprobs

    def to(self, device: torch.device) -> None:
        """Move model to device."""
        self.device = device
        self.model.to(device)

    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()
