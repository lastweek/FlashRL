"""Shared serving backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.config import ServingConfig
from flashrl.framework.models.actor import ActorModel


class ServingBackend(ABC):
    """Shared serving surface used by the rest of the framework."""

    config: ServingConfig
    device: Any
    generation_defaults: dict[str, Any]

    @abstractmethod
    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text responses from prompts."""

    @abstractmethod
    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        """Generate one structured sample per prompt."""

    @abstractmethod
    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[Any]]:
        """Generate prompt-major grouped candidates."""

    @abstractmethod
    def set_generation_defaults(self, **kwargs: Any) -> None:
        """Update default generation kwargs used by rollout code."""

    def set_live_rollout_debug(self, callback: Any, context: dict[str, Any]) -> None:
        """Install a serving-side live-rollout debug callback when supported."""
        del callback, context

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        """Update the framework-owned candidate index for debug streaming."""
        del candidate_index

    def clear_live_rollout_debug(self) -> None:
        """Clear any serving-side live-rollout debug hooks."""
        return None

    @abstractmethod
    def sync_from_training_actor(self, training_actor: ActorModel) -> None:
        """Refresh the serving copy from the training actor."""

    @abstractmethod
    def close(self) -> None:
        """Release any backend-owned resources."""
