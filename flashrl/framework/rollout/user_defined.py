"""User-defined rollout generator."""

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
        self._apply_generation_defaults()

    def _generation_kwargs(self) -> dict[str, int | float | bool]:
        """Return generation defaults derived from rollout config."""
        return {
            "max_new_tokens": getattr(self.config, "max_new_tokens", 512),
            "temperature": getattr(self.config, "temperature", 1.0),
            "top_p": getattr(self.config, "top_p", 0.9),
            "top_k": getattr(self.config, "top_k", 0),
            "do_sample": getattr(self.config, "do_sample", True),
        }

    def _apply_generation_defaults(self) -> None:
        """Apply generation defaults to the actor when supported."""
        generation_kwargs = self._generation_kwargs()
        if hasattr(self.actor, "set_generation_defaults"):
            self.actor.set_generation_defaults(**generation_kwargs)
        else:
            setattr(self.actor, "generation_defaults", generation_kwargs)

    def generate(self, prompts: list[Prompt]) -> list[RolloutOutput]:
        """Generate rollouts using user function.

        Args:
            prompts: List of input prompts.

        Returns:
            List of rollout outputs.
        """
        self._apply_generation_defaults()
        return self.rollout_fn(prompts, self.actor)

    def generate_grouped(
        self,
        prompts: list[Prompt],
        group_size: int,
    ) -> tuple[list[Prompt], list[RolloutOutput], list[int], list[int]]:
        """Expand prompts into prompt-major groups and generate one rollout per sample."""
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")

        expanded_prompts: list[Prompt] = []
        prompt_indices: list[int] = []
        candidate_indices: list[int] = []
        for prompt_index, prompt in enumerate(prompts):
            for candidate_index in range(group_size):
                expanded_prompts.append(prompt.model_copy(deep=True))
                prompt_indices.append(prompt_index)
                candidate_indices.append(candidate_index)

        rollouts = self.generate(expanded_prompts)
        if len(rollouts) != len(expanded_prompts):
            raise ValueError(
                "Grouped rollout must return exactly prompt_count * group_size samples "
                f"(expected {len(expanded_prompts)}, got {len(rollouts)})."
            )
        return expanded_prompts, rollouts, prompt_indices, candidate_indices

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
