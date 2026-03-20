"""Serving RPC clients."""

from __future__ import annotations

from pathlib import Path

import torch

from flashrl.framework.admin import utc_now_iso
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.client_common import _HttpJsonClient
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    ComponentStatus,
    GenerateGroupedRequest,
    GenerateGroupedResponse,
    GeneratedSamplePayload,
    StatusResponse,
    WeightVersionRef,
)
from flashrl.framework.serving.base import ServingBackend


def _resolve_weight_artifact_path(artifact_uri: str) -> Path:
    path = Path(artifact_uri)
    if path.is_dir():
        candidate = path / "model_state.pt"
        if candidate.exists():
            return candidate
    return path


class LocalServingClient:
    """In-process controller adapter over one serving backend."""

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


class HttpServingClient(_HttpJsonClient):
    """HTTP serving client."""

    def generate_grouped(self, request: GenerateGroupedRequest) -> GenerateGroupedResponse:
        return self._post("/v1/generate-grouped", request, GenerateGroupedResponse)

    def activate_weight_version(
        self,
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        return self._post("/v1/activate-weight-version", request, ActivateWeightVersionResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)


def activate_huggingface_serving_backend_from_ref(
    serving_backend: ServingBackend,
    weight_version: WeightVersionRef,
) -> WeightVersionInfo:
    """Activate a local file-based weight artifact on a serving backend."""
    weight_path = _resolve_weight_artifact_path(weight_version.artifact_uri)
    state_dict = torch.load(weight_path, weights_only=False)
    serving_backend._actor.model.load_state_dict(state_dict)  # type: ignore[attr-defined]
    activated = weight_version.to_info(activated_at=utc_now_iso())
    serving_backend._active_weight_version = activated
    serving_backend._pending_weight_version = None
    serving_backend._next_weight_version_id = max(
        int(getattr(serving_backend, "_next_weight_version_id", 1)),
        weight_version.version_id + 1,
    )
    serving_backend._last_successful_sync_at = serving_backend._active_weight_version.activated_at
    serving_backend._sync_healthy = True
    serving_backend._last_sync_error = None
    return serving_backend.current_weight_version()
