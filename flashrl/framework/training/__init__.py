"""Training backends and factory helpers."""

from __future__ import annotations

from flashrl.framework.config import GrpoConfig, TrainingConfig
from flashrl.framework.training.base import OptimizationResult, TrainingBackend
from flashrl.framework.training.fsdp2 import FSDP2TrainingBackend
from flashrl.framework.training.huggingface import HuggingFaceTrainingBackend


def create_training_backend(
    config: TrainingConfig,
    *,
    learning_rate: float,
    grpo_config: GrpoConfig,
    reference_enabled: bool = False,
    reference_device: str | None = None,
) -> TrainingBackend:
    """Construct the requested training backend implementation."""
    if config.backend == "huggingface":
        return HuggingFaceTrainingBackend(
            config,
            learning_rate=learning_rate,
            grpo_config=grpo_config,
            reference_enabled=reference_enabled,
            reference_device=reference_device,
        )
    if config.backend == "fsdp2":
        return FSDP2TrainingBackend(
            config,
            learning_rate=learning_rate,
            grpo_config=grpo_config,
            reference_enabled=reference_enabled,
            reference_device=reference_device,
        )
    raise ValueError(f"Unsupported training backend: {config.backend}")


__all__ = [
    "TrainingBackend",
    "OptimizationResult",
    "HuggingFaceTrainingBackend",
    "FSDP2TrainingBackend",
    "create_training_backend",
]
