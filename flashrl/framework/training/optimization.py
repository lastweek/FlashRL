"""Shared learner-side optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from flashrl.framework.data_models import LearnerBatch
from flashrl.framework.observability import StageResult, timed_call
from flashrl.framework.training.loss_variants import (
    LossAssemblyResult,
    assemble_grpo_loss,
)

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
        return reference.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def backward_step(optimizer: torch.optim.Optimizer, loss: torch.Tensor):
    """Return the backward closure used in timing."""

    def run() -> None:
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


# assemble_grpo_loss function is now in flashrl.framework.training.grpo.loss_variants
# and supports multiple variants. The optimize_grpo_batch function below
# will use the imported version automatically.


def optimize_grpo_batch(
    *,
    actor_backend: "ActorTrainingBackend",
    reference_backend: "TrainingBackend | None",
    grpo_config: "GrpoConfig",
    learner_batch: LearnerBatch,
) -> "OptimizationResult":
    """Run one full GRPO learner step across peer actor/reference backends."""
    from flashrl.framework.training.base import OptimizationResult

    (
        input_ids,
        attention_mask,
        prompt_lengths,
        full_lengths,
        rollout_response_log_probs,
    ), prepare_seconds = timed_call(lambda: actor_backend.prepare_inputs(learner_batch))
    full_tokens_total = int(sum(full_lengths))
    response_tokens_total = int(sum(len(tokens) for tokens in learner_batch.response_token_ids))

    actor_logits, actor_forward_seconds = timed_call(
        lambda: actor_backend.forward_logits(input_ids, attention_mask)
    )

    reference_active = reference_backend is not None and grpo_config.kl_coefficient > 0.0
    if reference_active:
        ref_logits, reference_forward_seconds = timed_call(
            lambda: reference_logits(reference_backend, input_ids, attention_mask)
        )
    else:
        ref_logits = None
        reference_forward_seconds = 0.0

    advantages = torch.tensor(
        learner_batch.advantages,
        dtype=torch.float32,
        device=actor_backend.device,
    )
    loss_result, loss_seconds = timed_call(
        lambda: assemble_grpo_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            actor_logits=actor_logits,
            ref_logits=ref_logits,
            rollout_response_log_probs=rollout_response_log_probs,
            advantages=advantages,
            kl_coefficient=grpo_config.kl_coefficient,
            clip_ratio=grpo_config.clip_ratio,
        )
    )

    _, backward_seconds = timed_call(actor_backend.backward_step(loss_result.loss))
    _, optimizer_seconds = timed_call(actor_backend.optimizer_step)
    learning_rate = float(actor_backend.optimizer.param_groups[0]["lr"])

    stages = [
        StageResult(
            name="prepare_inputs",
            seconds=prepare_seconds,
            metrics={
                "full_tokens_mean": _mean(full_lengths),
                "full_tokens_max": max(full_lengths, default=0),
                "response_tokens_total": response_tokens_total,
            },
        ),
        StageResult(
            name="actor_forward",
            seconds=actor_forward_seconds,
            metrics={"full_tokens_total": full_tokens_total},
        ),
    ]
    if reference_active:
        stages.append(
            StageResult(
                name="reference_forward",
                seconds=reference_forward_seconds,
                metrics={"full_tokens_total": full_tokens_total},
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
                },
            ),
            StageResult(
                name="backward",
                seconds=backward_seconds,
                metrics={"loss": float(loss_result.loss.item())},
            ),
            StageResult(
                name="optimizer",
                seconds=optimizer_seconds,
                metrics={"learning_rate": learning_rate},
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


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
