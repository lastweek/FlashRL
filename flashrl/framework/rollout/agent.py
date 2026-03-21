"""Internal rollout generator for whitebox ``Agent`` rollouts."""

from __future__ import annotations

from flashrl.framework.agent import Agent
from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import Prompt, RolloutOutput
from flashrl.framework.rollout.base import BaseRolloutGenerator
from flashrl.framework.serving import ServingBackend


class AgentRolloutGenerator(BaseRolloutGenerator):
    """Run an ``Agent`` directly without routing through the function adapter."""

    def __init__(
        self,
        *,
        agent: Agent,
        serving_backend: ServingBackend,
        config: RolloutConfig,
    ) -> None:
        self.agent = agent
        super().__init__(serving_backend=serving_backend, config=config)

    def _call_rollout(
        self,
        prompts: list[Prompt],
    ) -> list[RolloutOutput]:
        return self.agent.run_batch(prompts, self.serving_backend)

    def generate_grouped(
        self,
        prompts: list[Prompt],
        group_size: int,
    ) -> tuple[list[Prompt], list[RolloutOutput], list[int], list[int]]:
        """Delegate grouped whitebox scheduling directly to ``Agent.run_grouped``."""
        self._apply_generation_defaults()
        flat_prompts, flat_rollouts, prompt_indices, candidate_indices = self.agent.run_grouped(
            prompts,
            self.serving_backend,
            group_size,
        )
        validated: list[RolloutOutput] = []
        weight_version = self._current_weight_version()
        for rollout in flat_rollouts:
            self._stamp_weight_version(rollout, weight_version)
            self._validate_rollout_output(rollout)
            validated.append(rollout)
        return flat_prompts, validated, prompt_indices, candidate_indices
