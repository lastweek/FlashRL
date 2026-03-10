"""Simple rollout generator for testing."""

import random

from flashrl.framework.config import RolloutConfig
from flashrl.framework.rollout.base import BaseRollout
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    Conversation,
    Message,
)


class SimpleRollout(BaseRollout):
    """Simple rollout generator for testing.

    Generates dummy rollouts without actually calling a model.
    """

    def __init__(self, config: RolloutConfig | None = None) -> None:
        """Initialize simple rollout generator.

        Args:
            config: Rollout configuration.
        """
        if config is None:
            from flashrl.framework.config import RolloutConfig

            config = RolloutConfig()
        super().__init__(config)

    def generate(self, prompts: list[Prompt]) -> list[RolloutOutput]:
        """Generate dummy rollouts from prompts.

        Args:
            prompts: List of input prompts.

        Returns:
            List of rollout outputs with dummy data.
        """
        rollouts = []

        for prompt in prompts:
            # Generate dummy response
            response = f"Response to: {prompt.text[:50]}..."

            # Create conversation
            message = Message(role="assistant", content=response)
            conversation = Conversation(messages=[message])

            rollout = RolloutOutput(
                text=response,
                log_prob=random.random(),
                conversation=conversation,
            )
            rollouts.append(rollout)

        return rollouts

    def generate_conversation(
        self,
        prompt: Prompt,
        max_turns: int = 10,
    ) -> Conversation:
        """Generate a multi-turn conversation (not implemented).

        Args:
            prompt: Initial prompt.
            max_turns: Maximum number of turns.

        Returns:
            Empty conversation (placeholder).
        """
        return Conversation(messages=[])
