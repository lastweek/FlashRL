"""In-process Hugging Face serving backend."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Callable
from typing import Any

from flashrl.framework.config import ServingConfig
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import set_num_threads
from flashrl.framework.serving.base import ServingBackend


class HuggingFaceServingBackend(ServingBackend):
    """In-process Hugging Face serving backend.

    Supports weight loading without restart via in-memory state dict loading.
    This is the optimal approach for in-process backends.
    """

    def __init__(
        self,
        config: ServingConfig,
        startup_logger: Callable[[str], None] | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        del startup_logger, log_dir
        self.config = config

        # Set CPU thread limit before loading model.
        set_num_threads(config.num_threads)

        self._actor = ActorModel(config)
        self._actor.eval()
        self.device = self._actor.device
        self.generation_defaults: dict[str, Any] = {}
        self._lifecycle_lock = threading.RLock()
        self._initialize_weight_version(model_source=self.config.model_name, origin="startup")

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        with self._lifecycle_lock:
            return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        with self._lifecycle_lock:
            return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[Any]]:
        with self._lifecycle_lock:
            return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs: Any) -> None:
        with self._lifecycle_lock:
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

    def sync_from_training_actor(
        self,
        training_actor: ActorModel,
        *,
        source_training_step: int | None = None,
        source_epoch: int | None = None,
        origin: str = "sync",
    ) -> WeightVersionInfo:
        with self._lifecycle_lock:
            version_id = int(getattr(self, "_next_weight_version_id", 1))
            model_source = f"in_memory://{self.config.model_name}/version-{version_id}"
            self._begin_pending_weight_version(
                model_source=model_source,
                source_training_step=source_training_step,
                source_epoch=source_epoch,
                origin=origin,
            )
            try:
                self._actor.model.load_state_dict(training_actor.model.state_dict())
            except Exception as exc:
                self._clear_pending_weight_version(
                    sync_healthy=True,
                    last_sync_error=str(exc),
                )
                raise
            return self._commit_pending_weight_version()

    def close(self) -> None:
        with self._lifecycle_lock:
            return None
