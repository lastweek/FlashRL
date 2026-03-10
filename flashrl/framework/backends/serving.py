"""Serving backend for model generation."""

from typing import Any

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.actor import ActorModel


class ServingBackend:
    """Backend for model serving (generation).

    Owns the serving model with separate weights from training backend.
    Always in eval mode, optimized for generation.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize serving backend.

        Args:
            config: Model configuration.
        """
        self.config = config
        self.actor = ActorModel(config)
        self.actor.eval()  # Always in eval mode

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses from prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts.
        """
        return self.actor.generate(prompts, **kwargs)
