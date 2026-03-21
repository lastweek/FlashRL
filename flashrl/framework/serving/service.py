"""Domain-owned serving service implementation and HTTP app builder."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.http_common import install_common_routes
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    ComponentStatus,
    GenerateGroupedRequest,
    GenerateGroupedResponse,
    GeneratedSamplePayload,
    StatusResponse,
)
from flashrl.framework.serving.base import ServingBackend


class ServingService:
    """In-process serving service over one serving backend."""

    def __init__(self, serving_backend: ServingBackend) -> None:
        self._serving_backend = serving_backend

    def generate_grouped(self, request: GenerateGroupedRequest) -> GenerateGroupedResponse:
        active = self._serving_backend.current_weight_version()
        if (
            request.required_weight_version is not None
            and active.version_id != request.required_weight_version.version_id
        ):
            raise RuntimeError(
                "Serving backend active version does not match required_weight_version."
            )
        if request.generation_config is not None:
            self._serving_backend.set_generation_defaults(
                **request.generation_config.model_dump()
            )
        grouped_samples = self._serving_backend.generate_grouped(
            request.prompts,
            request.group_size,
        )
        return GenerateGroupedResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            grouped_samples=[
                [
                    request_model
                    for request_model in [
                        GeneratedSamplePayload.from_sample(sample)
                        for sample in prompt_samples
                    ]
                ]
                for prompt_samples in grouped_samples
            ],
            active_weight_version=active,
            metrics={"prompt_count": len(request.prompts)},
        )

    def activate_weight_version(
        self,
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        active = self._safe_current_weight_version()
        if active is None or active.version_id != request.weight_version.version_id:
            activate = getattr(self._serving_backend, "activate_weight_version_ref", None)
            if not callable(activate):
                raise RuntimeError(
                    "Serving backend does not support activate_weight_version_ref()."
                )
            active = activate(request.weight_version)

        status = self.status().status
        status.active_weight_version = active
        status.metadata.setdefault("desiredWeightVersion", request.weight_version.version_id)
        return ActivateWeightVersionResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            desired_weight_version=request.weight_version,
            active_weight_version=active,
            ready_replica_count=max(status.ready_replica_count, 1),
            total_replica_count=max(status.desired_replica_count, 1),
            converged=active.version_id == request.weight_version.version_id,
            status=status,
        )

    def status(self) -> StatusResponse:
        active = self._safe_current_weight_version()
        desired_replicas = int(getattr(self._serving_backend.config, "num_replicas", 1))
        return StatusResponse(
            status=ComponentStatus(
                name="serving",
                phase="Ready" if active is not None else "Starting",
                healthy=active is not None,
                ready_replica_count=desired_replicas if active is not None else 0,
                desired_replica_count=desired_replicas,
                active_weight_version=active,
            )
        )

    def _safe_current_weight_version(self) -> WeightVersionInfo | None:
        getter = getattr(self._serving_backend, "current_weight_version", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None


def create_serving_service_app(service: ServingService) -> FastAPI:
    """Create a serving RPC app around one local serving service."""
    app = FastAPI(title="FlashRL Serving Service")
    install_common_routes(
        app,
        status_getter=lambda: service.status().status,
        kind="ServingService",
        name="serving",
        drainable=True,
    )

    @app.post("/v1/generate-grouped")
    def generate_grouped(request: GenerateGroupedRequest) -> GenerateGroupedResponse:
        return service.generate_grouped(request)

    @app.post("/v1/activate-weight-version")
    def activate_weight_version(
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        return service.activate_weight_version(request)

    return app
