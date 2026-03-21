"""In-process rollout service implementation and HTTP app builder."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.http_common import install_common_routes
from flashrl.framework.distributed.models import (
    ComponentStatus,
    RolloutBatchRequest,
    RolloutBatchResponse,
    StatusResponse,
)
from flashrl.framework.rollout.base import BaseRolloutGenerator


class RolloutService:
    """In-process rollout service over one rollout generator."""

    def __init__(self, rollout_generator: BaseRolloutGenerator) -> None:
        self._rollout_generator = rollout_generator

    def rollout_batch(self, request: RolloutBatchRequest) -> RolloutBatchResponse:
        original_config = self._rollout_generator.config
        serving_backend = getattr(self._rollout_generator, "serving_backend", None)
        required_version_setter = getattr(serving_backend, "set_required_weight_version", None)
        if request.rollout_config is not None:
            self._rollout_generator.config = request.rollout_config.model_copy(deep=True)
        if callable(required_version_setter):
            required_version_setter(request.required_weight_version)
        try:
            prompts, rollouts, prompt_indices, candidate_indices = (
                self._rollout_generator.generate_grouped(request.prompts, request.group_size)
            )
        finally:
            self._rollout_generator.config = original_config
            if callable(required_version_setter):
                required_version_setter(None)

        weight_version = None
        if rollouts:
            metadata = dict(getattr(rollouts[0], "metadata", {}) or {})
            payload = metadata.get("weight_version")
            if isinstance(payload, dict):
                weight_version = WeightVersionInfo.model_validate(payload)

        return RolloutBatchResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            rollouts=rollouts,
            prompt_indices=prompt_indices,
            candidate_indices=candidate_indices,
            prompt_count=len(request.prompts),
            group_size=request.group_size,
            weight_version=weight_version,
            metrics={"sample_count": len(rollouts)},
        )

    def status(self) -> StatusResponse:
        return StatusResponse(
            status=ComponentStatus(
                name="rollout",
                phase="Ready",
                healthy=True,
                ready_replica_count=1,
                desired_replica_count=1,
            )
        )


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
