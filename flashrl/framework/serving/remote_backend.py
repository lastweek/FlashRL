"""ServingBackend adapter over the remote serving client."""

from __future__ import annotations

from typing import Any

from flashrl.framework.config import RolloutConfig, ServingConfig
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.models import GenerateGroupedRequest, WeightVersionRef
from flashrl.framework.distributed.serving_client import ServingClient
from flashrl.framework.serving.base import ServingBackend


class RemoteServingBackend(ServingBackend):
    """Remote serving backend used by rollout pods in platform mode."""

    def __init__(
        self,
        *,
        config: ServingConfig,
        client: ServingClient,
    ) -> None:
        self.config = config
        self.device = "remote"
        self.generation_defaults: dict[str, Any] = {}
        self._client = client
        self._required_weight_version: WeightVersionRef | None = None
        self._active_weight_version = self._load_active_weight_version()

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        grouped = self.generate_grouped(prompts, group_size=1, **kwargs)
        return [samples[0] for samples in grouped]

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[Any]]:
        generation_config = RolloutConfig.model_validate(
            {
                **self.generation_defaults,
                **kwargs,
            }
        )
        response = self._client.generate_grouped(
            GenerateGroupedRequest(
                prompts=list(prompts),
                group_size=group_size,
                generation_config=generation_config,
                required_weight_version=self._required_weight_version,
            )
        )
        self._active_weight_version = response.active_weight_version
        return [list(group) for group in response.grouped_samples]

    def set_generation_defaults(self, **kwargs: Any) -> None:
        self.generation_defaults.update(kwargs)

    def set_required_weight_version(self, weight_version: WeightVersionRef | None) -> None:
        self._required_weight_version = (
            weight_version.model_copy(deep=True) if weight_version is not None else None
        )

    def sync_from_training_actor(
        self,
        training_actor,
        *,
        source_training_step: int | None = None,
        source_epoch: int | None = None,
        origin: str = "sync",
    ) -> WeightVersionInfo:
        del training_actor, source_training_step, source_epoch, origin
        raise NotImplementedError("RemoteServingBackend cannot sync local training weights directly.")

    def current_weight_version(self) -> WeightVersionInfo:
        active = self._active_weight_version
        if active is None:
            raise RuntimeError("Remote serving backend has no active weight version yet.")
        return active.model_copy(deep=True)

    def close(self) -> None:
        return None

    def _load_active_weight_version(self) -> WeightVersionInfo | None:
        try:
            status = self._client.status().status
        except Exception:
            return None
        return status.active_weight_version
