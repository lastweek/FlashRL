"""User-defined rollout generator.

Allows users to pass simple functions instead of creating classes.
"""

from typing import Callable

from flashrl.framework.config import RolloutConfig
# Removed BaseRollout inheritance - using direct class
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    Conversation,
)
from flashrl.framework.models.actor import ActorModel


class UserDefinedRollout:
    """Rollout generator that wraps a user-provided function.

    This allows users to pass simple functions for rollout generation
    without needing to inherit from base classes.

    Example:
        def my_rollout_fn(prompts, actor):
            return actor.generate([p.text for p in prompts])

        rollout = UserDefinedRollout(my_rollout_fn, actor, config)
        rollouts = rollout.generate(prompts)
    """

    def __init__(
        self,
        rollout_fn: Callable[[list[Prompt], ActorModel], list[RolloutOutput]],
        actor: ActorModel,
        config: RolloutConfig,
    ) -> None:
        """Initialize UserDefinedRollout.

        Args:
            rollout_fn: User-provided function that generates rollouts.
                Takes (list[Prompt], ActorModel) and returns list[RolloutOutput].
            actor: Actor model to use for generation.
            config: Rollout configuration.
        """
        self.config = config
        self.rollout_fn = rollout_fn
        self.actor = actor

    def generate(self, prompts: list[Prompt]) -> list[RolloutOutput]:
        """Generate rollouts using user function.

        Args:
            prompts: List of input prompts.

        Returns:
            List of rollout outputs.
        """
        return self.rollout_fn(prompts, self.actor)

    def generate_conversation(
        self,
        prompt: Prompt,
        max_turns: int = 10,
    ) -> Conversation:
        """Generate a multi-turn conversation.

        For simple rollout functions, this defaults to single-turn.

        Args:
            prompt: Initial prompt.
            max_turns: Maximum number of turns (ignored for single-turn).

        Returns:
            Generated conversation.
        """
        # Generate single turn
        rollouts = self.generate([prompt])
        if rollouts:
            return rollouts[0].conversation
        else:
            return Conversation(messages=[])
