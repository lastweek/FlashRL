"""Serving backend for model generation."""

from typing import Any

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import set_num_threads


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

        # Set CPU thread limit before loading model
        set_num_threads(config.num_threads)

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

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        """Generate one structured sample per prompt."""
        return self.actor.generate_batch(prompts, **kwargs)

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[Any]]:
        """Generate grouped candidates from prompts."""
        return self.actor.generate_grouped(prompts, group_size, **kwargs)

    def set_live_rollout_debug(self, callback: Any, context: dict[str, Any]) -> None:
        """Install a serving-side live-rollout debug callback when enabled."""
        if not getattr(self.config, "debug_live_rollout", False):
            return
        self.actor.set_live_rollout_debug(callback, context)

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        """Update the current framework-owned candidate index for debug streaming."""
        if not getattr(self.config, "debug_live_rollout", False):
            return
        self.actor.set_live_rollout_candidate_index(candidate_index)

    def clear_live_rollout_debug(self) -> None:
        """Clear any serving-side live-rollout debug hooks."""
        self.actor.clear_live_rollout_debug()
