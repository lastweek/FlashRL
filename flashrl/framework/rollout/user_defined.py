"""User-defined rollout generator."""

from typing import Callable

from flashrl.framework.config import RolloutConfig
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
        def my_rollout_fn(prompts, actor, group_size):
            return actor.generate_grouped([p.text for p in prompts], group_size)

        rollout = UserDefinedRollout(my_rollout_fn, actor, config)
        rollouts = rollout.generate(prompts)
    """

    def __init__(
        self,
        rollout_fn: Callable[[list[Prompt], ActorModel, int], list[list[RolloutOutput]]],
        actor: ActorModel,
        config: RolloutConfig,
    ) -> None:
        """Initialize UserDefinedRollout.

        Args:
            rollout_fn: User-provided function that generates rollouts.
                Takes (list[Prompt], ActorModel, group_size) and returns
                prompt-major grouped RolloutOutput values.
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
        """Generate exactly one rollout per prompt."""
        grouped_rollouts = self._generate_nested(prompts, group_size=1)
        return [candidates[0] for candidates in grouped_rollouts]

    def generate_grouped(
        self,
        prompts: list[Prompt],
        group_size: int,
    ) -> tuple[list[Prompt], list[RolloutOutput], list[int], list[int]]:
        """Generate prompt-major grouped rollouts and flatten them for the trainer."""
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")

        grouped_rollouts = self._generate_nested(prompts, group_size=group_size)

        flat_prompts: list[Prompt] = []
        flat_rollouts: list[RolloutOutput] = []
        prompt_indices: list[int] = []
        candidate_indices: list[int] = []
        for prompt_index, (prompt, candidates) in enumerate(zip(prompts, grouped_rollouts, strict=True)):
            for candidate_index, rollout in enumerate(candidates):
                flat_prompts.append(prompt)
                flat_rollouts.append(rollout)
                prompt_indices.append(prompt_index)
                candidate_indices.append(candidate_index)

        return flat_prompts, flat_rollouts, prompt_indices, candidate_indices

    def _generate_nested(
        self,
        prompts: list[Prompt],
        group_size: int,
    ) -> list[list[RolloutOutput]]:
        """Generate grouped rollouts and validate prompt-major grouped shape."""
        self._apply_generation_defaults()
        grouped_rollouts = self.rollout_fn(prompts, self.actor, group_size)
        if len(grouped_rollouts) != len(prompts):
            raise ValueError(
                "Grouped rollout must return exactly one candidate list per input prompt "
                f"(expected {len(prompts)}, got {len(grouped_rollouts)})."
            )

        validated: list[list[RolloutOutput]] = []
        for prompt_index, candidates in enumerate(grouped_rollouts):
            if len(candidates) != group_size:
                raise ValueError(
                    "Grouped rollout must return exactly group_size candidates per prompt "
                    f"(prompt_index={prompt_index}, expected {group_size}, got {len(candidates)})."
                )
            validated_candidates: list[RolloutOutput] = []
            for rollout in candidates:
                self._validate_rollout_output(rollout)
                validated_candidates.append(rollout)
            validated.append(validated_candidates)
        return validated

    def _validate_rollout_output(self, rollout: RolloutOutput) -> None:
        """Validate that rollout outputs contain the token data required by GRPO."""
        if not rollout.prompt_token_ids:
            raise ValueError("RolloutOutput.prompt_token_ids must be populated for GRPO training.")
        if len(rollout.response_token_logprobs) != len(rollout.response_token_ids):
            raise ValueError(
                "RolloutOutput.response_token_logprobs must match RolloutOutput.response_token_ids."
            )

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
            Generated conversation. This path still uses the grouped rollout
            hook, but always requests a single candidate.
        """
        del max_turns
        self._apply_generation_defaults()
        grouped_rollouts = self.rollout_fn([prompt], self.actor, 1)
        if grouped_rollouts and grouped_rollouts[0]:
            rollout = grouped_rollouts[0][0]
            self._validate_rollout_output(rollout)
            return rollout.conversation
        return Conversation(messages=[])
