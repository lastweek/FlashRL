"""HTTP/JSON adapters for distributed FlashRL protocols."""

from __future__ import annotations

import json
from typing import Any, TypeVar
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel

from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    RewardBatchRequest,
    RewardBatchResponse,
    RolloutBatchRequest,
    RolloutBatchResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    StatusResponse,
)


T = TypeVar("T", bound=BaseModel)


class _HttpJsonClient:
    """Small standard-library JSON RPC client."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)

    def _post(self, path: str, payload: BaseModel, model: type[T]) -> T:
        request = urllib_request.Request(
            f"{self._base_url}{path}",
            data=payload.model_dump_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, model)

    def _get(self, path: str, model: type[T]) -> T:
        request = urllib_request.Request(
            f"{self._base_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        return self._send(request, model)

    def _send(self, request: urllib_request.Request, model: type[T]) -> T:
        try:
            with urllib_request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code} from {request.full_url}: {detail}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Request failed for {request.full_url}: {exc}") from exc

        parsed: Any = json.loads(body or "{}")
        return model.model_validate(parsed)


class HttpRolloutClient(_HttpJsonClient):
    """HTTP rollout client."""

    def rollout_batch(self, request: RolloutBatchRequest) -> RolloutBatchResponse:
        return self._post("/v1/rollout-batches", request, RolloutBatchResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)


class HttpRewardClient(_HttpJsonClient):
    """HTTP reward client."""

    def reward_batch(self, request: RewardBatchRequest) -> RewardBatchResponse:
        return self._post("/v1/reward-batches", request, RewardBatchResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)


class HttpLearnerClient(_HttpJsonClient):
    """HTTP learner client."""

    def optimize_step(self, request: OptimizeStepRequest) -> OptimizeStepResponse:
        return self._post("/v1/optimize-steps", request, OptimizeStepResponse)

    def save_checkpoint(self, request: SaveCheckpointRequest) -> SaveCheckpointResponse:
        return self._post("/v1/checkpoints/save", request, SaveCheckpointResponse)

    def load_checkpoint(self, request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        return self._post("/v1/checkpoints/load", request, LoadCheckpointResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)


class HttpServingClient(_HttpJsonClient):
    """HTTP serving client."""

    def activate_weight_version(
        self,
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        return self._post("/v1/activate-weight-version", request, ActivateWeightVersionResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)
