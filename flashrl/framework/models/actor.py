"""Actor model (policy model) wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

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
    metadata: dict[str, Any]


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
        self._live_rollout_debug_callback: Any = None
        self._live_rollout_debug_context: dict[str, Any] = {}
        self._live_rollout_candidate_index: int | None = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                self.tokenizer.pad_token_id = int(eos_token_id)
        self._configure_generation_special_tokens()

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
        if getattr(self.config, "debug_live_rollout", False):
            return self._generate_grouped_sequential(prompts, group_size, **kwargs)

        return self._generate_grouped_fast(prompts, group_size, **kwargs)

    def _generate_grouped_fast(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[GeneratedSample]]:
        """Generate grouped candidates in the default fast batched path."""
        generation_kwargs = {
            **self.generation_defaults,
            **kwargs,
            "num_return_sequences": group_size,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        self._apply_generation_special_token_kwargs(generation_kwargs)

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

            transition_scores = self._extract_generation_logprobs(outputs)
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
                    metadata = self._build_generation_metadata(
                        response_token_ids=response_token_ids,
                        generation_kwargs=generation_kwargs,
                    )
                    candidates.append(
                        GeneratedSample(
                            text=text,
                            prompt_token_ids=list(prompt_token_ids),
                            response_token_ids=response_token_ids,
                            response_token_logprobs=response_token_logprobs,
                            log_prob=float(sum(response_token_logprobs)),
                            metadata=metadata,
                        )
                    )
                grouped_outputs.append(candidates)
            return grouped_outputs
        finally:
            self.tokenizer.padding_side = original_padding_side

    def _generate_grouped_sequential(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[GeneratedSample]]:
        """Generate grouped candidates sequentially for live debug streaming."""
        grouped_outputs: list[list[GeneratedSample]] = []
        for prompt_index, prompt in enumerate(prompts):
            candidates: list[GeneratedSample] = []
            for candidate_index in range(group_size):
                effective_candidate_index = (
                    self._live_rollout_candidate_index
                    if group_size == 1 and self._live_rollout_candidate_index is not None
                    else candidate_index
                )
                candidates.append(
                    self._generate_debug_sample(
                        prompt=prompt,
                        prompt_index=prompt_index,
                        candidate_index=effective_candidate_index,
                        **kwargs,
                    )
                )
            grouped_outputs.append(candidates)
        return grouped_outputs

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text from prompts."""
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def set_generation_defaults(self, **kwargs: Any) -> None:
        """Set default generation kwargs used when rollout code does not pass them explicitly."""
        self.generation_defaults = dict(kwargs)

    def set_live_rollout_debug(
        self,
        callback: Any,
        context: dict[str, Any],
    ) -> None:
        """Install a live-rollout debug callback and step-scoped context."""
        self._live_rollout_debug_callback = callback
        self._live_rollout_debug_context = dict(context)
        self._live_rollout_candidate_index = None

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        """Update the current framework-owned candidate index for debug streaming."""
        self._live_rollout_candidate_index = candidate_index

    def clear_live_rollout_debug(self) -> None:
        """Clear the current live-rollout debug callback and context."""
        self._live_rollout_debug_callback = None
        self._live_rollout_debug_context = {}
        self._live_rollout_candidate_index = None

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
            use_cache=False,
        )
        return outputs.logits

    def _configure_generation_special_tokens(self) -> None:
        """Keep generation config aligned with tokenizer special-token ids."""
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = int(eos_token_id)
            self.tokenizer.pad_token_id = pad_token_id

        if pad_token_id is None:
            return

        if hasattr(self.model, "generation_config") and getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = int(pad_token_id)
        if hasattr(self.model, "config") and getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = int(pad_token_id)

    def _apply_generation_special_token_kwargs(self, generation_kwargs: dict[str, Any]) -> None:
        """Pass explicit special-token ids to generate() to avoid runtime inference warnings."""
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs.setdefault("pad_token_id", int(pad_token_id))
        if eos_token_id is not None:
            generation_kwargs.setdefault("eos_token_id", int(eos_token_id))

    def _extract_generation_logprobs(self, outputs: Any) -> torch.Tensor:
        """Extract log probabilities from HuggingFace generation output.

        Uses HuggingFace's native compute_transition_scores() when available,
        which is optimized. Falls back to efficient computation only when needed.

        Args:
            outputs: HuggingFace model.generate() output with output_scores=True

        Returns:
            Log probabilities for each generated token (batch_size, num_tokens)
        """
        # Use HuggingFace's native method (optimized) when available
        if hasattr(self.model, "compute_transition_scores"):
            return self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            ).detach().cpu()

        # Fallback: No scores available (shouldn't happen with output_scores=True)
        if not outputs.scores:
            return torch.zeros(
                (outputs.sequences.shape[0], 0),
                dtype=torch.float32,
            )

        # Manual computation (only when native method unavailable)
        # Uses F.cross_entropy which only computes target token log probs
        transition_scores = []
        generated_tokens = outputs.sequences[:, -len(outputs.scores):]
        for step_index, step_scores in enumerate(outputs.scores):
            step_ids = generated_tokens[:, step_index]
            neg_log_probs = F.cross_entropy(step_scores, step_ids, reduction='none')
            transition_scores.append(-neg_log_probs)

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

    def _generate_debug_sample(
        self,
        *,
        prompt: str,
        prompt_index: int,
        candidate_index: int,
        **kwargs: Any,
    ) -> GeneratedSample:
        """Generate one sample with live text streaming and timing capture."""
        generation_kwargs = {
            **self.generation_defaults,
            **kwargs,
            "num_return_sequences": 1,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        self._apply_generation_special_token_kwargs(generation_kwargs)

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            prompt_token_ids = inputs["input_ids"][0][
                inputs["attention_mask"][0].to(dtype=torch.bool)
            ].tolist()

            debug_payload = {
                **self._live_rollout_debug_context,
                "prompt_index": prompt_index,
                "candidate_index": candidate_index,
                "prompt_text": prompt,
                "prompt_preview": prompt,
            }
            self._emit_live_rollout_debug("start", debug_payload)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_started_at = time.perf_counter()
            first_text_at: float | None = None
            outputs_holder: dict[str, Any] = {}
            error_holder: dict[str, BaseException] = {}

            def run_generate() -> None:
                try:
                    with torch.no_grad():
                        outputs_holder["outputs"] = self.model.generate(
                            **inputs,
                            streamer=streamer,
                            **generation_kwargs,
                        )
                except BaseException as exc:  # pragma: no cover - propagated after join
                    error_holder["error"] = exc

            thread = Thread(target=run_generate, daemon=True)
            thread.start()

            for fragment in streamer:
                if not fragment:
                    continue
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                self._emit_live_rollout_debug(
                    "chunk",
                    {
                        **debug_payload,
                        "text": fragment,
                    },
                )

            thread.join()
            if "error" in error_holder:
                raise error_holder["error"]

            outputs = outputs_holder["outputs"]
            transition_scores = self._extract_generation_logprobs(outputs)
            prompt_width = inputs["input_ids"].shape[1]
            response_tokens = outputs.sequences[:, prompt_width:]
            response_token_ids, response_token_logprobs = self._trim_generated_response(
                response_tokens[0].tolist(),
                transition_scores[0].tolist() if len(transition_scores) else [],
            )
            generation_seconds = time.perf_counter() - generation_started_at
            ttft_seconds = (
                (first_text_at - generation_started_at)
                if first_text_at is not None
                else generation_seconds
            )
            if response_token_ids:
                tpot_seconds = (generation_seconds - ttft_seconds) / max(len(response_token_ids) - 1, 1)
            else:
                tpot_seconds = 0.0

            text = self.tokenizer.batch_decode(
                [response_token_ids],
                skip_special_tokens=True,
            )[0]
            metadata = {
                "ttft_seconds": float(ttft_seconds),
                "tpot_seconds": float(tpot_seconds),
                "generation_seconds": float(generation_seconds),
            }
            metadata.update(
                self._build_generation_metadata(
                    response_token_ids=response_token_ids,
                    generation_kwargs=generation_kwargs,
                )
            )
            self._emit_live_rollout_debug(
                "done",
                {
                    **debug_payload,
                    **metadata,
                    "response_preview": text,
                },
            )
            return GeneratedSample(
                text=text,
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                response_token_logprobs=response_token_logprobs,
                log_prob=float(sum(response_token_logprobs)),
                metadata=metadata,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

    def _emit_live_rollout_debug(self, kind: str, payload: dict[str, Any]) -> None:
        """Emit a live-rollout debug callback event when configured."""
        if self._live_rollout_debug_callback is None:
            return
        self._live_rollout_debug_callback(kind, payload)

    def _build_generation_metadata(
        self,
        *,
        response_token_ids: list[int],
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Infer finish metadata from one generated sample."""
        metadata: dict[str, Any] = {
            "response_token_count": len(response_token_ids),
        }
        finish_reason = self._infer_finish_reason(
            response_token_ids=response_token_ids,
            generation_kwargs=generation_kwargs,
        )
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        return metadata

    def _infer_finish_reason(
        self,
        *,
        response_token_ids: list[int],
        generation_kwargs: dict[str, Any],
    ) -> str | None:
        """Infer whether generation stopped normally or hit the length limit."""
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and response_token_ids and response_token_ids[-1] == int(eos_token_id):
            return "stop"

        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 0) or 0)
        if max_new_tokens > 0 and len(response_token_ids) >= max_new_tokens:
            return "length"
        return None

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
