"""Remote serving client."""

from __future__ import annotations

from flashrl.framework.distributed.client_common import _HttpJsonClient
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    GenerateGroupedRequest,
    GenerateGroupedResponse,
    StatusResponse,
)
class ServingClient(_HttpJsonClient):
    """Remote serving client."""

    def generate_grouped(self, request: GenerateGroupedRequest) -> GenerateGroupedResponse:
        return self._post("/v1/generate-grouped", request, GenerateGroupedResponse)

    def activate_weight_version(
        self,
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        return self._post("/v1/activate-weight-version", request, ActivateWeightVersionResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)
