"""Remote rollout client."""

from __future__ import annotations

from flashrl.framework.distributed.client_common import _HttpJsonClient
from flashrl.framework.distributed.models import (
    RolloutBatchRequest,
    RolloutBatchResponse,
    StatusResponse,
)


class RolloutClient(_HttpJsonClient):
    """Remote rollout client."""

    def rollout_batch(self, request: RolloutBatchRequest) -> RolloutBatchResponse:
        return self._post("/v1/rollout-batches", request, RolloutBatchResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)
