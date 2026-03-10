"""User-defined reward computation.

Allows users to pass simple functions instead of creating classes.
"""

from typing import Callable

from flashrl.framework.config import RewardConfig
from flashrl.framework.reward.base import BaseReward
from flashrl.framework.data_models import (
    RewardOutput,
    RolloutOutput,
    Conversation,
)


class UserDefinedReward(BaseReward):
    """Reward function that wraps a user-provided function.

    This allows users to pass simple functions for reward computation
    instead of creating full classes that inherit from BaseReward.

    Example:
        def my_reward_fn(rollout):
            score = len(rollout.text) / 100.0
            return RewardOutput(reward=score)

        reward = UserDefinedReward(my_reward_fn, config)
        reward_output = reward.compute(rollout)
    """

    def __init__(
        self,
        reward_fn: Callable[[RolloutOutput], RewardOutput],
        config: RewardConfig,
    ) -> None:
        """Initialize UserDefinedReward.

        Args:
            reward_fn: User-provided function that computes rewards.
                Takes RolloutOutput and returns RewardOutput.
            config: Reward configuration.
        """
        super().__init__(config)
        self.reward_fn = reward_fn

    def compute(self, rollout: RolloutOutput) -> RewardOutput:
        """Compute reward using user function.

        Args:
            rollout: Rollout output.

        Returns:
            Reward output.
        """
        return self.reward_fn(rollout)

    def compute_batch(self, rollouts: list[RolloutOutput]) -> list[RewardOutput]:
        """Compute rewards for a batch of rollouts.

        Args:
            rollouts: List of rollout outputs.

        Returns:
            List of reward outputs.
        """
        return [self.reward_fn(r) for r in rollouts]

    def compute_conversation_reward(
        self,
        conversation: Conversation,
    ) -> RewardOutput:
        """Compute reward for a multi-turn conversation.

        For simple reward functions, this creates a basic RolloutOutput
        from the conversation and calls the user function.

        Args:
            conversation: Conversation to evaluate.

        Returns:
            Reward output.
        """
        # Create a basic RolloutOutput from the conversation
        text = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation.messages])
        basic_rollout = RolloutOutput(
            text=text,
            log_prob=0.0,
            conversation=conversation,
        )
        return self.reward_fn(basic_rollout)
