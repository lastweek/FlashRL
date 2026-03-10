"""Base trainer abstraction."""

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.config import TrainerConfig
from flashrl.framework.data_models import TrainingBatch


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    def __init__(self, config: TrainerConfig) -> None:
        """Initialize the trainer.

        Args:
            config: Trainer configuration.
        """
        self.config = config
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
