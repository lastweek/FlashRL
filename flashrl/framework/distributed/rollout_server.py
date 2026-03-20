"""FastAPI app for the rollout service."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.distributed.models import RolloutBatchRequest, RolloutBatchResponse
from flashrl.framework.distributed.rollout_client import LocalRolloutClient
from flashrl.framework.distributed.server_common import install_common_routes


def create_rollout_app(client: LocalRolloutClient) -> FastAPI:
    """Create a rollout RPC app around one local rollout adapter."""
    app = FastAPI(title="FlashRL Rollout Service")
    install_common_routes(
        app,
        status_getter=lambda: client.status().status,
        kind="RolloutService",
        name="rollout",
        drainable=True,
    )

    @app.post("/v1/rollout-batches")
    def rollout_batches(request: RolloutBatchRequest) -> RolloutBatchResponse:
        return client.rollout_batch(request)

    return app
