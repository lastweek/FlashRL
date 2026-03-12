"""Shared fakes and helpers for offline FlashRL tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput


class TinyTokenizer:
    """Deterministic char-level tokenizer for fast unit tests."""

    def __init__(self, vocab_size: int = 32, *, pad_token: str | None = "<pad>") -> None:
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = vocab_size - 1
        self.padding_side = "right"
        self.calls: list[dict[str, Any]] = []
        self.batch_decode_calls: list[dict[str, Any]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.calls.append(
            {
                "texts": list(texts),
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
                "padding_side": self.padding_side,
            }
        )
        encoded = [self._encode(text, max_length=max_length) for text in texts]
        width = max(len(tokens) for tokens in encoded)

        input_ids = []
        attention_mask = []
        for tokens in encoded:
            padding_tokens = [0] * (width - len(tokens))
            if self.padding_side == "left":
                input_ids.append(padding_tokens + tokens)
                attention_mask.append([0] * len(padding_tokens) + [1] * len(tokens))
            else:
                input_ids.append(tokens + padding_tokens)
                attention_mask.append([1] * len(tokens) + [0] * len(padding_tokens))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(self, token_batches: torch.Tensor | list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        if not isinstance(token_batches, torch.Tensor):
            token_batches = torch.tensor(token_batches, dtype=torch.long)
        self.batch_decode_calls.append(
            {
                "skip_special_tokens": skip_special_tokens,
                "shape": tuple(token_batches.shape),
            }
        )
        outputs = []
        for batch in token_batches.tolist():
            values = [str(value) for value in batch if value != 0]
            outputs.append("decoded::" + ",".join(values))
        return outputs

    def _encode(self, text: str, *, max_length: int) -> list[int]:
        tokens = [((ord(char) % (self.vocab_size - 2)) + 1) for char in text[:max_length]]
        return tokens or [1]


class TinyCausalLM(torch.nn.Module):
    """Tiny causal LM with deterministic logits and generation."""

    def __init__(self, vocab_size: int = 32, bias_shift: float = 0.0) -> None:
        super().__init__()
        base = torch.linspace(-0.2, 0.2, vocab_size)
        self.logit_bias = torch.nn.Parameter(base + bias_shift)
        self.device = torch.device("cpu")
        self.last_generate_kwargs: dict[str, Any] | None = None
        self.last_input_ids: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.last_labels: torch.Tensor | None = None
        self.grad_enabled_during_forward: bool | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        self.last_labels = labels
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        vocab_size = self.logit_bias.shape[0]
        logits = self.logit_bias.view(1, 1, vocab_size).expand(
            input_ids.shape[0],
            input_ids.shape[1],
            vocab_size,
        )
        token_signal = F.one_hot(input_ids % vocab_size, num_classes=vocab_size).float()
        return SimpleNamespace(logits=logits + 0.05 * token_signal)

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.last_generate_kwargs = dict(kwargs)
        input_ids = kwargs["input_ids"]
        num_return_sequences = int(kwargs.get("num_return_sequences", 1))
        expanded_input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        completion = torch.tensor(
            [[self.logit_bias.shape[0] - 2, self.logit_bias.shape[0] - 1]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).repeat(expanded_input_ids.shape[0], 1)
        sequences = torch.cat([expanded_input_ids, completion], dim=1)
        scores = [
            self.logit_bias.view(1, -1).repeat(expanded_input_ids.shape[0], 1),
            (self.logit_bias + 0.1).view(1, -1).repeat(expanded_input_ids.shape[0], 1),
        ]
        if kwargs.get("return_dict_in_generate"):
            return SimpleNamespace(sequences=sequences, scores=scores)
        return sequences

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: list[torch.Tensor],
        normalize_logits: bool,
    ) -> torch.Tensor:
        del normalize_logits
        generated_tokens = sequences[:, -len(scores):]
        per_step = []
        for step_index, step_scores in enumerate(scores):
            token_logprobs = F.log_softmax(step_scores, dim=-1)
            step_ids = generated_tokens[:, step_index].unsqueeze(-1)
            per_step.append(torch.gather(token_logprobs, dim=-1, index=step_ids).squeeze(-1))
        return torch.stack(per_step, dim=1)

    def to(self, device: torch.device | str) -> "TinyCausalLM":
        self.device = torch.device(device)
        return super().to(self.device)


class TinyActor:
    """Actor wrapper matching the backend contract used in trainer tests."""

    def __init__(self, bias_shift: float = 0.0) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_length=128)
        self.tokenizer = TinyTokenizer()
        self.model = TinyCausalLM(self.tokenizer.vocab_size, bias_shift=bias_shift)
        self.generation_defaults: dict[str, Any] = {}
        self.last_generate_kwargs: dict[str, Any] | None = None
        self._batch_call_index = 0

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.last_generate_kwargs = dict(kwargs)
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[SimpleNamespace]:
        self.last_generate_kwargs = dict(kwargs)
        call_index = self._batch_call_index
        self._batch_call_index += 1
        outputs: list[SimpleNamespace] = []
        for prompt in prompts:
            prompt_token_ids = self.tokenizer._encode(prompt, max_length=self.config.max_length)
            response_text = f"generated::{prompt}::{call_index}"
            response_token_ids = self.tokenizer._encode(
                response_text,
                max_length=self.config.max_length,
            )[:4]
            response_token_logprobs = [
                -0.1 - 0.01 * call_index - 0.001 * token_index
                for token_index in range(len(response_token_ids))
            ]
            outputs.append(
                SimpleNamespace(
                    text=response_text,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    log_prob=float(sum(response_token_logprobs)),
                )
            )
        return outputs

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[SimpleNamespace]]:
        self.last_generate_kwargs = dict(kwargs)
        grouped_outputs: list[list[SimpleNamespace]] = []
        for prompt in prompts:
            prompt_token_ids = self.tokenizer._encode(prompt, max_length=self.config.max_length)
            prompt_outputs: list[SimpleNamespace] = []
            for candidate_index in range(group_size):
                response_text = f"generated::{prompt}::{candidate_index}"
                response_token_ids = self.tokenizer._encode(
                    response_text,
                    max_length=self.config.max_length,
                )[:4]
                response_token_logprobs = [
                    -0.1 - 0.01 * candidate_index - 0.001 * token_index
                    for token_index in range(len(response_token_ids))
                ]
                prompt_outputs.append(
                    SimpleNamespace(
                        text=response_text,
                        prompt_token_ids=prompt_token_ids,
                        response_token_ids=response_token_ids,
                        response_token_logprobs=response_token_logprobs,
                        log_prob=float(sum(response_token_logprobs)),
                    )
                )
            grouped_outputs.append(prompt_outputs)
        return grouped_outputs

    def set_generation_defaults(self, **kwargs: Any) -> None:
        self.generation_defaults = dict(kwargs)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


class TinyTrainingBackend:
    """Local training backend fake for GRPO tests."""

    def __init__(self, learning_rate: float = 1e-3) -> None:
        self.actor = TinyActor(bias_shift=0.25)
        self.actor.train()
        self.optimizer = torch.optim.SGD(self.actor.model.parameters(), lr=learning_rate)

    def sync_weights_to(self, serving_backend: "TinyServingBackend") -> None:
        serving_backend.actor.model.load_state_dict(self.actor.model.state_dict())


class TinyServingBackend:
    """Local serving backend fake for GRPO tests."""

    def __init__(self) -> None:
        self.actor = TinyActor(bias_shift=0.1)
        self.actor.eval()


class TinyReferenceModel:
    """Reference model fake for GRPO tests."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = TinyCausalLM(bias_shift=0.0)


def make_rollout_fn(response_suffix: str = "response", repeat: int = 1):
    """Create deterministic rollout outputs for a batch of prompts."""

    call_index = 0

    def rollout_fn(prompts: list[Prompt], actor: Any) -> list[RolloutOutput]:
        nonlocal call_index
        outputs: list[RolloutOutput] = []
        for prompt in prompts:
            prompt_token_ids = actor.tokenizer._encode(prompt.text, max_length=actor.config.max_length)
            response = (
                f"{response_suffix} "
                + ("detail " * repeat)
                + prompt.text
                + f"::{call_index}"
            )
            response_token_ids = actor.tokenizer._encode(
                response,
                max_length=actor.config.max_length,
            )[:4]
            response_token_logprobs = [
                -0.05 - 0.01 * call_index - 0.001 * token_index
                for token_index in range(len(response_token_ids))
            ]
            outputs.append(
                RolloutOutput(
                    text=response,
                    log_prob=float(sum(response_token_logprobs)),
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    conversation=Conversation(
                        messages=[
                            Message(role="user", content=prompt.text),
                            Message(role="assistant", content=response),
                        ]
                    ),
                )
            )
        call_index += 1
        return outputs

    return rollout_fn


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Simple deterministic reward based on response length."""
    return RewardOutput(reward=len(rollout.text) / 50.0)
