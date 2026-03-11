"""Base trainer abstraction."""

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.config import TrainerConfig
from flashrl.framework.data_models import TrainingBatch
from flashrl.framework.run_logger import RunLogger


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    def __init__(
        self,
        config: TrainerConfig,
        run_logger: RunLogger | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Trainer configuration.
            run_logger: Optional run-scoped logger.
        """
        self.config = config
        self.run_logger = run_logger
        self.current_epoch = 0
        self.total_steps = 0

    def attach_run_logger(self, run_logger: RunLogger | None) -> None:
        """Attach or clear the current run-scoped logger."""
        self.run_logger = run_logger

    def reset_state(self) -> None:
        """Reset per-run trainer state."""
        self.current_epoch = 0
        self.total_steps = 0

    @abstractmethod
    def train(self, dataset: Any) -> None:
        """Train on the given dataset.

        Args:
            dataset: Training data.
        """
        pass

    @abstractmethod
    def step(self, batch: TrainingBatch) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Training batch.

        Returns:
            Dictionary of metrics (e.g., loss, kl_divergence).
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        pass
