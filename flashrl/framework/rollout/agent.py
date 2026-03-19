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
