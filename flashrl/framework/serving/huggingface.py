"""In-process Hugging Face serving backend."""

from __future__ import annotations

from typing import Callable
from typing import Any

from flashrl.framework.config import ServingConfig
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import set_num_threads
from flashrl.framework.serving.base import ServingBackend


class HuggingFaceServingBackend(ServingBackend):
    """In-process Hugging Face serving backend."""

    def __init__(
        self,
        config: ServingConfig,
        startup_logger: Callable[[str], None] | None = None,
    ) -> None:
        del startup_logger
        self.config = config

        # Set CPU thread limit before loading model.
        set_num_threads(config.num_threads)

        self._actor = ActorModel(config)
        self._actor.eval()
        self.device = self._actor.device
        self.generation_defaults: dict[str, Any] = {}

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[Any]]:
        return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs: Any) -> None:
        self.generation_defaults = dict(kwargs)
        self._actor.set_generation_defaults(**kwargs)

    def set_live_rollout_debug(self, callback: Any, context: dict[str, Any]) -> None:
        if not self.config.debug_live_rollout:
            return
        self._actor.set_live_rollout_debug(callback, context)

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        if not self.config.debug_live_rollout:
            return
        self._actor.set_live_rollout_candidate_index(candidate_index)

    def clear_live_rollout_debug(self) -> None:
        self._actor.clear_live_rollout_debug()

    def sync_from_training_actor(self, training_actor: ActorModel) -> None:
        self._actor.model.load_state_dict(training_actor.model.state_dict())

    def close(self) -> None:
        return None
