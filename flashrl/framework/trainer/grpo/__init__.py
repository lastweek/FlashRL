"""GRPO (Group Relative Policy Optimization) trainer and utilities."""

from __future__ import annotations

from flashrl.framework.trainer.grpo.grpo_helpers import (
    STAGE_ORDER,
    StepContext,
    accumulate_totals,
    batch_items,
    compute_advantages,
    mean_payload_metrics,
    prompt_batch_size,
    reward_rate_stats,
)
from flashrl.framework.trainer.grpo.loss_variants import (
    LossAssemblyResult,
    assemble_grpo_loss,
)

__all__ = [
    "StepContext",
    "compute_advantages",
    "prompt_batch_size",
    "reward_rate_stats",
    "mean_payload_metrics",
    "accumulate_totals",
    "batch_items",
    "STAGE_ORDER",
    "LossAssemblyResult",
    "assemble_grpo_loss",
]
