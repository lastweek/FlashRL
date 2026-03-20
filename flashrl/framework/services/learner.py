"""FastAPI app for the learner service."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.distributed.local import LocalLearnerClient
from flashrl.framework.distributed.models import (
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
)
from flashrl.framework.services.common import install_common_routes


def create_learner_app(client: LocalLearnerClient) -> FastAPI:
    """Create a learner RPC app around one local learner adapter."""
    app = FastAPI(title="FlashRL Learner Service")
    install_common_routes(
        app,
        status_getter=lambda: client.status().status,
        kind="LearnerService",
        name="learner",
    )

    @app.post("/v1/optimize-steps")
    def optimize_steps(request: OptimizeStepRequest) -> OptimizeStepResponse:
        return client.optimize_step(request)

    @app.post("/v1/checkpoints/save")
    def save_checkpoint(request: SaveCheckpointRequest) -> SaveCheckpointResponse:
        return client.save_checkpoint(request)

    @app.post("/v1/checkpoints/load")
    def load_checkpoint(request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        return client.load_checkpoint(request)

    return app
