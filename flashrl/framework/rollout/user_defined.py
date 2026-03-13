"""User-defined rollout generator."""

from typing import Callable

from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    Conversation,
)
from flashrl.framework.serving import ServingBackend


class UserDefinedRollout:
    """Rollout generator that wraps a user-provided function.

    This allows users to pass simple functions for rollout generation
    without needing to inherit from base classes.

    Example:
        def my_rollout_fn(prompts, serving_backend):
            samples = serving_backend.generate_batch([p.text for p in prompts])
            return [...]

        rollout = UserDefinedRollout(my_rollout_fn, serving_backend, config)
        rollouts = rollout.generate(prompts)
    """

    def __init__(
        self,
        rollout_fn: Callable[[list[Prompt], ServingBackend], list[RolloutOutput]],
        serving_backend: ServingBackend,
        config: RolloutConfig,
    ) -> None:
        """Initialize UserDefinedRollout.

        Args:
            rollout_fn: User-provided function that generates rollouts.
                Takes (list[Prompt], ServingBackend) and returns exactly one
                RolloutOutput per input prompt.
            serving_backend: Serving backend to use for generation.
            config: Rollout configuration.
        """
        self.config = config
        self.rollout_fn = rollout_fn
        self.serving_backend = serving_backend
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
        """Apply generation defaults to the serving backend."""
        generation_kwargs = self._generation_kwargs()
        self.serving_backend.set_generation_defaults(**generation_kwargs)

    def generate(self, prompts: list[Prompt]) -> list[RolloutOutput]:
        """Generate exactly one rollout per prompt."""
        return self._generate_once(prompts)

    def generate_grouped(
        self,
        prompts: list[Prompt],
        group_size: int,
    ) -> tuple[list[Prompt], list[RolloutOutput], list[int], list[int]]:
        """Generate prompt-major grouped rollouts and flatten them for the trainer."""
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")

        grouped_rollouts: list[list[RolloutOutput]] = [[] for _ in prompts]
        try:
            for candidate_index in range(group_size):
                self.serving_backend.set_live_rollout_candidate_index(candidate_index)
                rollouts = self._generate_once(prompts)
                for prompt_index, rollout in enumerate(rollouts):
                    grouped_rollouts[prompt_index].append(rollout)
        finally:
            self.serving_backend.set_live_rollout_candidate_index(None)

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

    def _generate_once(
        self,
        prompts: list[Prompt],
    ) -> list[RolloutOutput]:
        """Generate one rollout per prompt and validate the flat output shape."""
        self._apply_generation_defaults()
        rollouts = self.rollout_fn(prompts, self.serving_backend)
        if len(rollouts) != len(prompts):
            raise ValueError(
                "Rollout must return exactly one output per input prompt "
                f"(expected {len(prompts)}, got {len(rollouts)})."
            )

        validated: list[RolloutOutput] = []
        for rollout in rollouts:
            self._validate_rollout_output(rollout)
            validated.append(rollout)
        return validated

    def _validate_rollout_output(self, rollout: RolloutOutput) -> None:
        """Validate that rollout outputs contain the token data required by GRPO."""
        if not rollout.prompt_token_ids:
            raise ValueError("RolloutOutput.prompt_token_ids must be populated for GRPO training.")
        if rollout.response_token_logprobs and (
            len(rollout.response_token_logprobs) != len(rollout.response_token_ids)
        ):
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
            Generated conversation.
        """
        del max_turns
        self._apply_generation_defaults()
        rollouts = self.rollout_fn([prompt], self.serving_backend)
        if rollouts:
            if len(rollouts) != 1:
                raise ValueError(
                    "Rollout must return exactly one output per input prompt "
                    f"(expected 1, got {len(rollouts)})."
                )
            rollout = rollouts[0]
            self._validate_rollout_output(rollout)
            return rollout.conversation
        return Conversation(messages=[])
