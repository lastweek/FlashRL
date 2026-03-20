"""FastAPI app for the serving service."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.distributed.local import LocalServingClient
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    GenerateGroupedRequest,
    GenerateGroupedResponse,
)
from flashrl.framework.services.common import install_common_routes


def create_serving_app(client: LocalServingClient) -> FastAPI:
    """Create a serving RPC app around one local serving adapter."""
    app = FastAPI(title="FlashRL Serving Service")
    install_common_routes(
        app,
        status_getter=lambda: client.status().status,
        kind="ServingService",
        name="serving",
    )

    @app.post("/v1/generate-grouped")
    def generate_grouped(request: GenerateGroupedRequest) -> GenerateGroupedResponse:
        return client.generate_grouped(request)

    @app.post("/v1/activate-weight-version")
    def activate_weight_version(
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        return client.activate_weight_version(request)

    return app
