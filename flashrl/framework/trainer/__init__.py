"""Trainer abstractions and implementations."""

from flashrl.framework.trainer.base import BaseTrainer
from flashrl.framework.trainer.grpo import GRPOTrainer

__all__ = ["BaseTrainer", "GRPOTrainer"]
