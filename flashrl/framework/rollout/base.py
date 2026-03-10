"""Base rollout abstraction."""

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    Conversation,
)


class BaseRollout(ABC):
    """Abstract base class for rollout generation."""

    def __init__(self, config: RolloutConfig) -> None:
        """Initialize the rollout generator.

        Args:
            config: Rollout configuration.
        """
        self.config = config

    @abstractmethod
    def generate(self, prompts: list[Prompt]) -> list[RolloutOutput]:
        """Generate rollouts from prompts.

        Args:
            prompts: List of input prompts.

        Returns:
            List of rollout outputs.
        """
        pass

    @abstractmethod
    def generate_conversation(
        self,
        prompt: Prompt,
        max_turns: int = 10,
    ) -> Conversation:
        """Generate a multi-turn conversation.

        Args:
            prompt: Initial prompt.
            max_turns: Maximum number of turns to generate.

        Returns:
            Generated conversation.
        """
        pass
