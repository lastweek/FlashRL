"""Shared learner-side optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from flashrl.framework.data_models import LearnerBatch


@dataclass
class LossAssemblyResult:
    """Full response-only GRPO loss assembly output."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_divergence: torch.Tensor
    response_tokens_total: int
    importance_sampling_ratio_mean: float
    importance_sampling_ratio_std: float
    importance_sampling_ratio_min: float
    importance_sampling_ratio_max: float
    clip_fraction: float

    def __iter__(self):
        """Preserve tuple-style unpacking in tests."""
        yield self.loss
        yield self.policy_loss
        yield self.kl_divergence
        yield self.response_tokens_total


def reference_logits(
    reference: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen reference model without gradients."""
    with torch.no_grad():
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


def assemble_grpo_loss(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    actor_logits: torch.Tensor,
    ref_logits: torch.Tensor | None,
    rollout_response_log_probs: list[list[float]],
    advantages: torch.Tensor,
    kl_coefficient: float,
    clip_ratio: float,
) -> LossAssemblyResult:
    """Assemble the response-only GRPO objective and count response tokens."""
    shift_ids = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()

    actor_token_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
    log_pi_theta = torch.gather(
        actor_token_log_probs,
        dim=-1,
        index=shift_ids.unsqueeze(-1),
    ).squeeze(-1)

    response_mask = torch.zeros_like(shift_mask)
    for index, prompt_length in enumerate(prompt_lengths.tolist()):
        prompt_start = max(int(prompt_length) - 1, 0)
        response_mask[index, prompt_start:] = shift_mask[index, prompt_start:]
    response_token_count = response_mask.sum().clamp(min=1)
    response_tokens_total = int(response_mask.sum().item())

    if len(rollout_response_log_probs) != response_mask.shape[0]:
        raise ValueError("rollout_response_log_probs must match the batch size.")

    log_pi_old = torch.zeros(
        response_mask.shape,
        dtype=log_pi_theta.dtype,
        device=log_pi_theta.device,
    )
    response_mask_bool = response_mask.to(dtype=torch.bool)
    for index, sample_log_probs in enumerate(rollout_response_log_probs):
        expected_tokens = int(response_mask_bool[index].sum().item())
        if len(sample_log_probs) != expected_tokens:
            raise ValueError(
                "Each rollout_response_log_probs entry must match that sample's "
                "response-token count."
            )
        if expected_tokens == 0:
            continue
        sample_tensor = torch.tensor(
            sample_log_probs,
            dtype=log_pi_theta.dtype,
            device=log_pi_theta.device,
        )
        log_pi_old[index, response_mask_bool[index]] = sample_tensor

    sample_advantages = advantages.to(
        device=log_pi_theta.device,
        dtype=log_pi_theta.dtype,
    ).unsqueeze(-1)
    expanded_advantages = sample_advantages.expand_as(response_mask)
    ratio = torch.exp(log_pi_theta - log_pi_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate_unclipped = ratio * expanded_advantages
    surrogate_clipped = clipped_ratio * expanded_advantages
    surrogate_objective = torch.minimum(surrogate_unclipped, surrogate_clipped)
    masked_surrogate = surrogate_objective * response_mask
    policy_loss = -masked_surrogate.sum() / response_token_count

    ratio_values = ratio[response_mask_bool]
    if ratio_values.numel() > 0:
        importance_sampling_ratio_mean = float(ratio_values.mean().item())
        importance_sampling_ratio_std = float(ratio_values.std(unbiased=False).item())
        importance_sampling_ratio_min = float(ratio_values.min().item())
        importance_sampling_ratio_max = float(ratio_values.max().item())
        clip_fraction = float(
            ((ratio_values < (1.0 - clip_ratio)) | (ratio_values > (1.0 + clip_ratio)))
            .float()
            .mean()
            .item()
        )
    else:
        importance_sampling_ratio_mean = 0.0
        importance_sampling_ratio_std = 0.0
        importance_sampling_ratio_min = 0.0
        importance_sampling_ratio_max = 0.0
        clip_fraction = 0.0

    if ref_logits is not None:
        reference_token_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        log_pi_ref = torch.gather(
            reference_token_log_probs,
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)
        reference_log_gap = log_pi_ref - log_pi_theta
        kl_terms = torch.exp(reference_log_gap) - reference_log_gap - 1.0
        masked_kl = kl_terms * response_mask
        kl_divergence = masked_kl.sum() / response_token_count
    else:
        kl_divergence = torch.zeros(
            (),
            device=log_pi_theta.device,
            dtype=log_pi_theta.dtype,
        )

    loss = policy_loss + kl_coefficient * kl_divergence
    return LossAssemblyResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_divergence=kl_divergence,
        response_tokens_total=response_tokens_total,
        importance_sampling_ratio_mean=importance_sampling_ratio_mean,
        importance_sampling_ratio_std=importance_sampling_ratio_std,
        importance_sampling_ratio_min=importance_sampling_ratio_min,
        importance_sampling_ratio_max=importance_sampling_ratio_max,
        clip_fraction=clip_fraction,
    )
