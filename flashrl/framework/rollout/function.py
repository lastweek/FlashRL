"""Internal rollout generator for plain function-based rollouts."""

from __future__ import annotations

from typing import Callable

from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import Prompt, RolloutOutput
from flashrl.framework.rollout.base import BaseRolloutGenerator
from flashrl.framework.serving import ServingBackend


class FunctionRolloutGenerator(BaseRolloutGenerator):
    """Adapt a plain ``rollout_fn(prompts, serving_backend)`` into the internal protocol."""

    def __init__(
        self,
        *,
        rollout_fn: Callable[[list[Prompt], ServingBackend], list[RolloutOutput]],
        serving_backend: ServingBackend,
        config: RolloutConfig,
    ) -> None:
        self.rollout_fn = rollout_fn
        super().__init__(serving_backend=serving_backend, config=config)

    def _call_rollout(
        self,
        prompts: list[Prompt],
    ) -> list[RolloutOutput]:
        return self.rollout_fn(prompts, self.serving_backend)
