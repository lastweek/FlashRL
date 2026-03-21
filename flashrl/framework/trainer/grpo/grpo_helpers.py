"""GRPO-specific helpers for training step orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from flashrl.framework.data_models import RewardOutput
from flashrl.framework.utils import mean


STAGE_ORDER = (
    "rollout",
    "reward",
    "advantage",
    "prepare_inputs",
    "actor_forward",
    "reference_forward",
    "loss_assembly",
    "backward",
    "optimizer",
    "publish_weights",
    "sync",
)


@dataclass(frozen=True)
class StepContext:
    """Stable metadata shared by all events in a training step."""

    step: int
    epoch: int
    total_epochs: int
    batch_index: int
    batches_in_epoch: int
    batch_size: int
    prompt_count: int
    group_size: int
    dataset_prompt_start: int
    dataset_prompt_end: int
    dataset_prompt_count: int
    planned_prompts_per_step: int
    planned_samples_per_step: int

    def payload(self) -> dict[str, int]:
        """Return the event payload fields shared by all step-stage logs."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "batch_index": self.batch_index,
            "batches_in_epoch": self.batches_in_epoch,
            "batch_size": self.batch_size,
            "prompt_count": self.prompt_count,
            "group_size": self.group_size,
            "dataset_prompt_start": self.dataset_prompt_start,
            "dataset_prompt_end": self.dataset_prompt_end,
            "dataset_prompt_count": self.dataset_prompt_count,
            "planned_prompts_per_step": self.planned_prompts_per_step,
            "planned_samples_per_step": self.planned_samples_per_step,
            "completions_per_prompt": self.group_size,
            "planned_completions_per_step": self.planned_samples_per_step,
            "samples_this_step": self.batch_size,
            "completions_this_step": self.batch_size,
        }


def compute_advantages(
    rewards: list[RewardOutput],
    *,
    prompt_count: int,
    group_size: int,
) -> torch.Tensor:
    """Compute GRPO advantages within each prompt group."""
    reward_values = torch.tensor(
        [reward.reward for reward in rewards],
        dtype=torch.float32,
    )
    expected_samples = prompt_count * group_size
    if reward_values.numel() != expected_samples:
        raise ValueError(
            "Reward count must match prompt_count * group_size for GRPO "
            f"(expected {expected_samples}, got {reward_values.numel()})."
        )
    if reward_values.numel() == 0:
        return reward_values
    grouped_rewards = reward_values.view(prompt_count, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    group_stds = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped_rewards - group_means) / (group_stds + 1e-8)).reshape(-1)


def batch_items(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    """Yield fixed-size slices from one list."""
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def prompt_batch_size(batch_size: int, group_size: int) -> int:
    """Return the number of unique prompts consumed by each grouped GRPO step."""
    return batch_size // group_size


def reward_rate_stats(rewards: list[RewardOutput]) -> dict[str, float]:
    """Aggregate common boolean reward metadata into per-step rates."""
    mappings = (
        ("accuracy_pass", "accuracy_pass_rate"),
        ("format_pass", "format_pass_rate"),
        ("truncated", "truncation_rate"),
    )
    stats: dict[str, float] = {}
    for source_key, target_key in mappings:
        values = [
            float(bool(reward.metadata[source_key]))
            for reward in rewards
            if source_key in reward.metadata
        ]
        if values:
            stats[target_key] = mean(values)
    return stats


def mean_payload_metrics(
    payloads: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, float]:
    """Average selected float payload metrics across a list of step payloads."""
    result: dict[str, float] = {}
    for key in keys:
        values = [float(payload[key]) for payload in payloads if key in payload]
        if values:
            result[key] = mean(values)
    return result


def accumulate_totals(target: dict[str, float], update: dict[str, float]) -> None:
    """Accumulate a flat mapping of numeric totals."""
    for key, value in update.items():
        target[key] = target.get(key, 0.0) + float(value)
