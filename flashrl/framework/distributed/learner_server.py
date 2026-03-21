"""FastAPI app for the learner service."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.distributed.learner_service import LearnerService
from flashrl.framework.distributed.models import (
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
)
from flashrl.framework.distributed.server_common import install_common_routes


def create_learner_service_app(service: LearnerService) -> FastAPI:
    """Create a learner RPC app around one local learner service."""
    app = FastAPI(title="FlashRL Learner Service")
    install_common_routes(
        app,
        status_getter=lambda: service.status().status,
        kind="LearnerService",
        name="learner",
    )

    @app.post("/v1/optimize-steps")
    def optimize_steps(request: OptimizeStepRequest) -> OptimizeStepResponse:
        return service.optimize_step(request)

    @app.post("/v1/checkpoints/save")
    def save_checkpoint(request: SaveCheckpointRequest) -> SaveCheckpointResponse:
        return service.save_checkpoint(request)

    @app.post("/v1/checkpoints/load")
    def load_checkpoint(request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        return service.load_checkpoint(request)

    return app
