"""Training backends and factory helpers."""

from __future__ import annotations

from flashrl.framework.config import TrainingConfig
from flashrl.framework.training.base import ActorTrainingBackend, OptimizationResult, TrainingBackend
from flashrl.framework.training.fsdp2 import FSDP2ReferenceBackend, FSDP2TrainingBackend
from flashrl.framework.training.huggingface import (
    HuggingFaceReferenceBackend,
    HuggingFaceTrainingBackend,
)


def create_training_backend(
    config: TrainingConfig,
    *,
    role: str,
    learning_rate: float | None = None,
) -> TrainingBackend:
    """Construct one role-specific training backend implementation."""
    if role == "actor":
        if learning_rate is None:
            raise ValueError("learning_rate is required when creating the actor backend.")
        if config.backend == "huggingface":
            return HuggingFaceTrainingBackend(config, learning_rate=learning_rate)
        if config.backend == "fsdp2":
            return FSDP2TrainingBackend(config, learning_rate=learning_rate)
        raise ValueError(f"Unsupported actor backend: {config.backend}")

    if role == "reference":
        if config.backend == "huggingface":
            return HuggingFaceReferenceBackend(config)
        if config.backend == "fsdp2":
            return FSDP2ReferenceBackend(config)
        raise ValueError(f"Unsupported reference backend: {config.backend}")

    raise ValueError(f"Unsupported training backend role: {role}")


__all__ = [
    "TrainingBackend",
    "ActorTrainingBackend",
    "OptimizationResult",
    "HuggingFaceTrainingBackend",
    "FSDP2TrainingBackend",
    "create_training_backend",
]
