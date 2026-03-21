"""Remote reward client."""

from __future__ import annotations

from flashrl.framework.distributed.client_common import _HttpJsonClient
from flashrl.framework.distributed.models import (
    RewardBatchRequest,
    RewardBatchResponse,
    StatusResponse,
)


class RewardClient(_HttpJsonClient):
    """Remote reward client."""

    def reward_batch(self, request: RewardBatchRequest) -> RewardBatchResponse:
        return self._post("/v1/reward-batches", request, RewardBatchResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)
