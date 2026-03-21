"""FastAPI app for the rollout service."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.distributed.models import RolloutBatchRequest, RolloutBatchResponse
from flashrl.framework.distributed.rollout_service import RolloutService
from flashrl.framework.distributed.server_common import install_common_routes


def create_rollout_service_app(service: RolloutService) -> FastAPI:
    """Create a rollout RPC app around one local rollout service."""
    app = FastAPI(title="FlashRL Rollout Service")
    install_common_routes(
        app,
        status_getter=lambda: service.status().status,
        kind="RolloutService",
        name="rollout",
        drainable=True,
    )

    @app.post("/v1/rollout-batches")
    def rollout_batches(request: RolloutBatchRequest) -> RolloutBatchResponse:
        return service.rollout_batch(request)

    return app
