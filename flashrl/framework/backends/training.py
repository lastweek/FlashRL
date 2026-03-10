"""Training backend for model optimization."""

import torch

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.actor import ActorModel


class TrainingBackend:
    """Backend for model training.

    Owns the training model, optimizer, and checkpointing.
    Always in training mode, has separate weights from serving backend.
    """

    def __init__(self, config: ModelConfig, learning_rate: float = 1e-5) -> None:
        """Initialize training backend.

        Args:
            config: Model configuration.
            learning_rate: Learning rate for optimizer.
        """
        self.config = config
        self.actor = ActorModel(config)
        self.actor.train()  # Always in training mode
        self.optimizer = torch.optim.Adam(
            self.actor.model.parameters(),
            lr=learning_rate,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save(self.actor.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        self.actor.model.load_state_dict(torch.load(path, weights_only=False))

    def sync_weights_to(self, serving_backend: "ServingBackend") -> None:
        """Sync training weights to serving backend.

        Args:
            serving_backend: Serving backend to sync weights to.
        """
        serving_backend.actor.model.load_state_dict(
            self.actor.model.state_dict()
        )
