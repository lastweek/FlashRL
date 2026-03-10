"""Simple reward function for testing."""

import random

from flashrl.framework.config import RewardConfig
from flashrl.framework.reward.base import BaseReward
from flashrl.framework.data_models import (
    RewardOutput,
    RolloutOutput,
    Conversation,
)


class SimpleReward(BaseReward):
    """Simple reward function for testing.

    Returns dummy rewards based on simple heuristics.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialize simple reward function.

        Args:
            config: Reward configuration.
        """
        if config is None:
            from flashrl.framework.config import RewardConfig

            config = RewardConfig()
        super().__init__(config)

    def compute(self, rollout: RolloutOutput) -> RewardOutput:
        """Compute a dummy reward for a rollout.

        Args:
            rollout: Rollout output.

        Returns:
            Reward output with dummy reward.
        """
        # Simple heuristic: longer responses get higher reward
        reward = min(len(rollout.text) / 100.0, 10.0)
        return RewardOutput(reward=reward)

    def compute_batch(self, rollouts: list[RolloutOutput]) -> list[RewardOutput]:
        """Compute rewards for a batch of rollouts.

        Args:
            rollouts: List of rollout outputs.

        Returns:
            List of reward outputs.
        """
        return [self.compute(r) for r in rollouts]

    def compute_conversation_reward(
        self,
        conversation: Conversation,
    ) -> RewardOutput:
        """Compute reward for a conversation.

        Args:
            conversation: Conversation to evaluate.

        Returns:
            Reward output.
        """
        # Simple heuristic: more messages = higher reward
        reward = min(len(conversation.messages), 10.0)
        return RewardOutput(reward=reward)
