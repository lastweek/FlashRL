"""Shared learner-side optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from flashrl.framework.data_models import LearnerBatch
from flashrl.framework.memory import (
    capture_memory_snapshot,
    memory_pressure_tags,
    release_device_cache,
)
from flashrl.framework.observability import StageResult, timed_call
from flashrl.framework.controller.grpo.loss_variants import (
    LossAssemblyResult,
    assemble_grpo_loss,
    get_current_policy_log_probs,
)
from flashrl.framework.utils import mean

if TYPE_CHECKING:
    from flashrl.framework.config import GrpoConfig
    from flashrl.framework.training.base import ActorTrainingBackend, OptimizationResult, TrainingBackend


# LossAssemblyResult and assemble_grpo_loss are now imported from
# flashrl.framework.training.loss_variants to support multiple variants


def reference_logits(
    reference: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen reference model without gradients."""
    with torch.no_grad():
        if hasattr(reference, "forward_logits"):
            return reference.forward_logits(input_ids, attention_mask)
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        try:
            signature = inspect.signature(reference.model.forward)
        except (TypeError, ValueError):
            supports_use_cache = True
        else:
            supports_use_cache = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD or name == "use_cache"
                for name, parameter in signature.parameters.items()
            )
        if supports_use_cache:
            forward_kwargs["use_cache"] = False
        return reference.model(**forward_kwargs).logits


def backward_step(optimizer: torch.optim.Optimizer, loss: torch.Tensor):
    """Return the backward closure used in timing."""

    def run() -> None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
        loss.backward()

    return run


def prepare_learner_inputs(
    learner_batch: LearnerBatch,
    actor: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
    """Build padded learner tensors directly from rollout-provided token ids."""
    pad_token_id = getattr(actor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    full_sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    full_lengths: list[int] = []
    rollout_response_log_probs: list[list[float] | None] = []
    for prompt_ids, response_ids, response_logprobs in zip(
        learner_batch.prompt_token_ids,
        learner_batch.response_token_ids,
        learner_batch.response_token_logprobs,
        strict=True,
    ):
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        response_ids = [int(token_id) for token_id in response_ids]
        response_logprobs = [float(value) for value in response_logprobs]
        full_sequence = prompt_ids + response_ids
        if not full_sequence:
            raise ValueError("GRPO training requires at least one prompt token per rollout.")
        full_sequences.append(full_sequence)
        prompt_lengths.append(len(prompt_ids))
        full_lengths.append(len(full_sequence))
        if response_ids and len(response_ids) != len(response_logprobs):
            rollout_response_log_probs.append(None)
        else:
            rollout_response_log_probs.append(response_logprobs)

    batch_size = len(full_sequences)
    max_length = max(full_lengths, default=0)
    input_ids = torch.full(
        (batch_size, max_length),
        fill_value=int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device,
    )
    for index, token_ids in enumerate(full_sequences):
        length = len(token_ids)
        input_ids[index, :length] = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1

    # Clean up temporary list that is no longer needed
    del full_sequences

    prompt_length_tensor = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
    if any(sample_log_probs is None for sample_log_probs in rollout_response_log_probs):
        computed_log_probs = compute_rollout_log_probs_from_actor(
            actor=actor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )
        rollout_response_log_probs = [
            computed if sample_log_probs is None else sample_log_probs
            for sample_log_probs, computed in zip(
                rollout_response_log_probs,
                computed_log_probs,
                strict=True,
            )
        ]
    # Clean up prompt_lengths list after converting to tensor
    del prompt_lengths
    return (
        input_ids,
        attention_mask,
        prompt_length_tensor,
        full_lengths,
        [list(sample_log_probs) for sample_log_probs in rollout_response_log_probs],
    )


def compute_rollout_log_probs_from_actor(
    actor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
) -> list[list[float]]:
    """Derive rollout-policy token logprobs from the synced training actor."""
    with torch.no_grad():
        if hasattr(actor, "compute_log_probs"):
            actor_logits = actor.compute_log_probs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        else:
            outputs = actor.model(
                input_ids=input_ids.to(actor.device),
                attention_mask=attention_mask.to(actor.device),
                labels=input_ids.to(actor.device),
            )
            actor_logits = outputs.logits

    token_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
    shift_ids = input_ids[:, 1:]
    gathered = torch.gather(
        token_log_probs,
        dim=-1,
        index=shift_ids.unsqueeze(-1),
    ).squeeze(-1)
    shift_mask = attention_mask[:, 1:].to(dtype=torch.bool)

    computed: list[list[float]] = []
    for index, prompt_length in enumerate(prompt_lengths):
        response_mask = torch.zeros_like(shift_mask[index], dtype=torch.bool)
        prompt_start = max(int(prompt_length) - 1, 0)
        response_mask[prompt_start:] = shift_mask[index, prompt_start:]
        computed.append(gathered[index, response_mask].detach().cpu().tolist())
    return computed


# assemble_grpo_loss function is now unified and configurable via GrpoConfig
# The optimize_grpo_batch function below uses this unified interface.


@dataclass
class OptimizationStageError(RuntimeError):
    """Wrap one learner-stage failure with stage and memory context."""

    stage_name: str
    message: str
    memory_snapshot: dict[str, Any]
    reason_tags: list[str]

    def __post_init__(self) -> None:
        self.args = (self.message,)

    def __str__(self) -> str:
        return self.message


def _timed_stage(
    *,
    stage_name: str,
    device: Any,
    operation,
    shared_device_pressure: bool = False,
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    """Run one learner substage and capture before/after memory."""
    before_memory = capture_memory_snapshot(device)
    started_at = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        failure_memory = capture_memory_snapshot(device)
        raise OptimizationStageError(
            stage_name=stage_name,
            message=str(exc),
            memory_snapshot=failure_memory,
            reason_tags=memory_pressure_tags(
                exc,
                snapshot=failure_memory,
                shared_device_pressure=shared_device_pressure,
            ),
        ) from exc
    after_memory = capture_memory_snapshot(device)
    return result, time.perf_counter() - started_at, before_memory, after_memory


def optimize_grpo_batch(
    *,
    actor_backend: "ActorTrainingBackend",
    reference_backend: "TrainingBackend | None",
    grpo_config: "GrpoConfig",
    learner_batch: LearnerBatch,
) -> "OptimizationResult":
    """Run one full GRPO learner step across peer actor/reference backends."""
    from flashrl.framework.training.base import OptimizationResult

    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    prompt_lengths: torch.Tensor | None = None
    full_lengths: list[int] = []
    rollout_response_log_probs: list[list[float]] = []
    actor_logits: torch.Tensor | None = None
    ref_logits: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    training_old_response_log_probs: list[list[float]] | None = None
    current_policy_log_probs: torch.Tensor | None = None
    resolved_grpo_config = grpo_config.get_resolved_config()

    try:
        (
            input_ids,
            attention_mask,
            prompt_lengths,
            full_lengths,
            rollout_response_log_probs,
        ), prepare_seconds, prepare_memory_before, prepare_memory_after = _timed_stage(
            stage_name="prepare_inputs",
            device=actor_backend.device,
            operation=lambda: actor_backend.prepare_inputs(learner_batch),
        )
        full_tokens_total = int(sum(full_lengths))
        response_tokens_total = int(sum(len(tokens) for tokens in learner_batch.response_token_ids))

        actor_logits, actor_forward_seconds, actor_forward_memory_before, actor_forward_memory_after = _timed_stage(
            stage_name="actor_forward",
            device=actor_backend.device,
            operation=lambda: actor_backend.forward_logits(input_ids, attention_mask),
        )

        if resolved_grpo_config.entropy_coefficient == 0.0:
            (
                (current_policy_log_probs, _, _),
                logprob_seconds,
            ) = timed_call(
                lambda: get_current_policy_log_probs(
                    input_ids,
                    attention_mask,
                    actor_logits,
                )
            )
            actor_forward_seconds += logprob_seconds
            del actor_logits
            actor_logits = None
            actor_forward_memory_after = capture_memory_snapshot(actor_backend.device)

        # Compute π^train_old logprobs BEFORE optimizer update
        # This is critical for GLM-5 IcePop token gate.
        if grpo_config.enable_icepop_token_gate:
            training_old_response_log_probs, _ = timed_call(
                lambda: compute_rollout_log_probs_from_actor(
                    actor=actor_backend.model_copy,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_lengths=prompt_lengths,
                )
            )

        reference_active = reference_backend is not None and grpo_config.kl_coefficient > 0.0
        if reference_active:
            ref_logits, reference_forward_seconds, reference_memory_before, reference_memory_after = _timed_stage(
                stage_name="reference_forward",
                device=getattr(reference_backend, "device", actor_backend.device),
                operation=lambda: reference_logits(reference_backend, input_ids, attention_mask),
            )
        else:
            reference_forward_seconds = 0.0
            reference_memory_before = {}
            reference_memory_after = {}

        advantages = torch.tensor(
            learner_batch.advantages,
            dtype=torch.float32,
            device=actor_backend.device,
        )
        loss_result, loss_seconds, loss_memory_before, loss_memory_after = _timed_stage(
            stage_name="loss_assembly",
            device=actor_backend.device,
            operation=lambda: assemble_grpo_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
                actor_logits=actor_logits,
                ref_logits=ref_logits,
                rollout_response_log_probs=rollout_response_log_probs,  # π^infer_old
                training_response_log_probs=training_old_response_log_probs,  # π^train_old
                advantages=advantages,
                config=grpo_config,  # Pass full config instead of individual parameters
                current_policy_log_probs=current_policy_log_probs,
            ),
        )

        _, backward_seconds, backward_memory_before, backward_memory_after = _timed_stage(
            stage_name="backward",
            device=actor_backend.device,
            operation=actor_backend.backward_step(loss_result.loss),
        )
        _, optimizer_seconds, optimizer_memory_before, optimizer_memory_after = _timed_stage(
            stage_name="optimizer",
            device=actor_backend.device,
            operation=actor_backend.optimizer_step,
        )
        learning_rate = float(actor_backend.optimizer.param_groups[0]["lr"])

        stages = [
            StageResult(
                name="prepare_inputs",
                seconds=prepare_seconds,
                metrics={
                    "full_tokens_mean": mean(full_lengths),
                    "full_tokens_max": max(full_lengths, default=0),
                    "response_tokens_total": response_tokens_total,
                    "memory": {
                        "before": prepare_memory_before,
                        "after": prepare_memory_after,
                    },
                },
            ),
            StageResult(
                name="actor_forward",
                seconds=actor_forward_seconds,
                metrics={
                    "full_tokens_total": full_tokens_total,
                    "memory": {
                        "before": actor_forward_memory_before,
                        "after": actor_forward_memory_after,
                    },
                },
            ),
        ]
        if reference_active:
            stages.append(
                StageResult(
                    name="reference_forward",
                    seconds=reference_forward_seconds,
                    metrics={
                        "full_tokens_total": full_tokens_total,
                        "memory": {
                            "before": reference_memory_before,
                            "after": reference_memory_after,
                        },
                    },
                )
            )
        stages.extend(
            [
                StageResult(
                    name="loss_assembly",
                    seconds=loss_seconds,
                    metrics={
                        "loss": float(loss_result.loss.item()),
                        "policy_loss": float(loss_result.policy_loss.item()),
                        "kl_divergence": float(loss_result.kl_divergence.item()),
                        "response_tokens_total": loss_result.response_tokens_total,
                        "importance_sampling_ratio_mean": loss_result.importance_sampling_ratio_mean,
                        "importance_sampling_ratio_std": loss_result.importance_sampling_ratio_std,
                        "importance_sampling_ratio_min": loss_result.importance_sampling_ratio_min,
                        "importance_sampling_ratio_max": loss_result.importance_sampling_ratio_max,
                        "clip_fraction": loss_result.clip_fraction,
                        "memory": {
                            "before": loss_memory_before,
                            "after": loss_memory_after,
                        },
                    },
                ),
                StageResult(
                    name="backward",
                    seconds=backward_seconds,
                    metrics={
                        "loss": float(loss_result.loss.item()),
                        "memory": {
                            "before": backward_memory_before,
                            "after": backward_memory_after,
                        },
                    },
                ),
                StageResult(
                    name="optimizer",
                    seconds=optimizer_seconds,
                    metrics={
                        "learning_rate": learning_rate,
                        "memory": {
                            "before": optimizer_memory_before,
                            "after": optimizer_memory_after,
                        },
                    },
                ),
            ]
        )

        return OptimizationResult(
            loss=float(loss_result.loss.item()),
            policy_loss=float(loss_result.policy_loss.item()),
            kl_divergence=float(loss_result.kl_divergence.item()),
            learning_rate=learning_rate,
            response_tokens_total=loss_result.response_tokens_total,
            reference_active=reference_active,
            stages=stages,
        )
    finally:
        if input_ids is not None:
            del input_ids
        if attention_mask is not None:
            del attention_mask
        if prompt_lengths is not None:
            del prompt_lengths
        if actor_logits is not None:
            del actor_logits
        if ref_logits is not None:
            del ref_logits
        if advantages is not None:
            del advantages
        if training_old_response_log_probs is not None:
            del training_old_response_log_probs
        if current_policy_log_probs is not None:
            del current_policy_log_probs
        release_device_cache(actor_backend.device)
