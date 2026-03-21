"""Remote learner client."""

from __future__ import annotations

from flashrl.framework.distributed.client_common import _HttpJsonClient
from flashrl.framework.distributed.models import (
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    StatusResponse,
)
class LearnerClient(_HttpJsonClient):
    """Remote learner client."""

    def optimize_step(self, request: OptimizeStepRequest) -> OptimizeStepResponse:
        return self._post("/v1/optimize-steps", request, OptimizeStepResponse)

    def save_checkpoint(self, request: SaveCheckpointRequest) -> SaveCheckpointResponse:
        return self._post("/v1/checkpoints/save", request, SaveCheckpointResponse)

    def load_checkpoint(self, request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        return self._post("/v1/checkpoints/load", request, LoadCheckpointResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)
