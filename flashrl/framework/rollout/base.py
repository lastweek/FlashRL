"""Internal rollout-generator base and factory helpers."""

from __future__ import annotations

from typing import Any

from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import (
    Conversation,
    Prompt,
    RolloutOutput,
    WeightVersionInfo,
)
from flashrl.framework.serving import ServingBackend


class BaseRolloutGenerator:
    """Internal base for grouped rollout generation."""

    def __init__(
        self,
        *,
        serving_backend: ServingBackend,
        config: RolloutConfig,
    ) -> None:
        self.serving_backend = serving_backend
        self.config = config
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
        self.serving_backend.set_generation_defaults(**self._generation_kwargs())

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
        rollouts = self._call_rollout(prompts)
        if len(rollouts) != len(prompts):
            raise ValueError(
                "Rollout must return exactly one output per input prompt "
                f"(expected {len(prompts)}, got {len(rollouts)})."
            )

        validated: list[RolloutOutput] = []
        weight_version = self._current_weight_version()
        for rollout in rollouts:
            self._stamp_weight_version(rollout, weight_version)
            self._validate_rollout_output(rollout)
            validated.append(rollout)
        return validated

    def _call_rollout(
        self,
        prompts: list[Prompt],
    ) -> list[RolloutOutput]:
        """Generate one raw rollout per prompt before validation."""
        raise NotImplementedError

    def _current_weight_version(self) -> WeightVersionInfo | None:
        getter = getattr(self.serving_backend, "current_weight_version", None)
        if getter is None or not callable(getter):
            return None
        try:
            weight_version = getter()
        except Exception:
            return None
        if isinstance(weight_version, WeightVersionInfo):
            return weight_version
        return None

    def _stamp_weight_version(
        self,
        rollout: RolloutOutput,
        weight_version: WeightVersionInfo | None,
    ) -> None:
        if weight_version is None:
            return
        rollout.metadata = {
            **dict(rollout.metadata),
            "weight_version": weight_version.model_dump(),
        }

    def _validate_rollout_output(self, rollout: RolloutOutput) -> None:
        """Validate that rollout outputs contain the token data required by GRPO."""
        if rollout.assistant_turns:
            for turn in rollout.assistant_turns:
                if not turn.prompt_token_ids:
                    raise ValueError(
                        "AssistantTurn.prompt_token_ids must be populated for GRPO training."
                    )
                if turn.response_token_logprobs and (
                    len(turn.response_token_logprobs) != len(turn.response_token_ids)
                ):
                    raise ValueError(
                        "AssistantTurn.response_token_logprobs must match "
                        "AssistantTurn.response_token_ids."
                    )
            return

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
        """Generate a multi-turn conversation for diagnostic helpers."""
        del max_turns
        self._apply_generation_defaults()
        rollouts = self._call_rollout([prompt])
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


def build_rollout_generator(
    *,
    rollout_fn: Any,
    serving_backend: ServingBackend,
    config: RolloutConfig,
) -> BaseRolloutGenerator:
    """Normalize the public rollout hook into one internal rollout generator."""
    from flashrl.framework.agent import Agent
    from flashrl.framework.rollout.agent import AgentRolloutGenerator
    from flashrl.framework.rollout.function import FunctionRolloutGenerator

    if isinstance(rollout_fn, Agent):
        return AgentRolloutGenerator(
            agent=rollout_fn,
            serving_backend=serving_backend,
            config=config,
        )
    if callable(rollout_fn):
        return FunctionRolloutGenerator(
            rollout_fn=rollout_fn,
            serving_backend=serving_backend,
            config=config,
        )
    raise TypeError(
        "rollout_fn must be a callable or flashrl.framework.agent.Agent "
        f"(got {type(rollout_fn).__name__})."
    )
