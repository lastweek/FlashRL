"""Base reward abstraction."""

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.config import RewardConfig
from flashrl.framework.data_models import (
    RewardOutput,
    RolloutOutput,
    Conversation,
)


class BaseReward(ABC):
    """Abstract base class for reward computation."""

    def __init__(self, config: RewardConfig) -> None:
        """Initialize the reward function.

        Args:
            config: Reward configuration.
        """
        self.config = config

    @abstractmethod
    def compute(self, rollout: RolloutOutput) -> RewardOutput:
        """Compute reward for a single rollout.

        Args:
            rollout: Rollout output.

        Returns:
            Reward output.
        """
        pass

    @abstractmethod
    def compute_batch(self, rollouts: list[RolloutOutput]) -> list[RewardOutput]:
        """Compute rewards for a batch of rollouts.

        Args:
            rollouts: List of rollout outputs.

        Returns:
            List of reward outputs.
        """
        pass

    @abstractmethod
    def compute_conversation_reward(
        self,
        conversation: Conversation,
    ) -> RewardOutput:
        """Compute reward for a multi-turn conversation.

        Args:
            conversation: Conversation to evaluate.

        Returns:
            Reward output.
        """
        pass
