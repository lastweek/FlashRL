"""Domain-owned learner service implementation and HTTP app builder."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from fastapi import FastAPI
import torch

from flashrl.framework.config import GrpoConfig
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.http_common import install_common_routes
from flashrl.framework.distributed.models import (
    ComponentStatus,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    StageResultPayload,
    StatusResponse,
    WeightVersionRef,
)
from flashrl.framework.observability import StageResult
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.training.base import ActorTrainingBackend, ReferenceTrainingBackend
from flashrl.framework.training.optimization import optimize_grpo_batch


def _stage_payload(stage: StageResult) -> StageResultPayload:
    return StageResultPayload(
        name=stage.name,
        seconds=float(stage.seconds),
        metrics=dict(stage.metrics),
    )


def _readable_weight_version(weight_version: WeightVersionInfo | None) -> WeightVersionRef:
    if weight_version is None:
        return WeightVersionRef(version_id=0, artifact_uri="", origin="startup")
    return WeightVersionRef(
        version_id=weight_version.version_id,
        artifact_uri=str(weight_version.model_source),
        source_training_step=weight_version.source_training_step,
        source_epoch=weight_version.source_epoch,
        origin=weight_version.origin,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class LearnerService:
    """In-process learner service over actor/reference training backends."""

    def __init__(
        self,
        actor_backend: ActorTrainingBackend,
        reference_backend: ReferenceTrainingBackend | None,
        *,
        grpo_config: GrpoConfig,
        serving_backend: ServingBackend | None = None,
        publish_dir: str | Path | None = None,
        synchronize_serving: bool = True,
    ) -> None:
        self._actor_backend = actor_backend
        self._reference_backend = reference_backend
        self._grpo_config = grpo_config
        self._serving_backend = serving_backend
        self._synchronize_serving = synchronize_serving
        self._step_results: dict[int, OptimizeStepResponse] = {}
        root_dir = Path(publish_dir) if publish_dir is not None else None
        if root_dir is None and not synchronize_serving:
            root_dir = Path(tempfile.mkdtemp(prefix="flashrl-published-weights-"))
        self._publish_dir = root_dir

    def reset_state(self) -> None:
        """Clear per-run exactly-once caches when a new local run starts."""
        self._step_results.clear()

    def optimize_step(self, request: OptimizeStepRequest) -> OptimizeStepResponse:
        if request.step_id is not None and request.step_id in self._step_results:
            return self._step_results[int(request.step_id)].model_copy(deep=True)

        result = optimize_grpo_batch(
            actor_backend=self._actor_backend,
            reference_backend=self._reference_backend,
            grpo_config=self._grpo_config,
            learner_batch=request.learner_batch,
        )
        weight_version = self._publish_weight_version(request)
        response = OptimizeStepResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            loss=result.loss,
            policy_loss=result.policy_loss,
            kl_divergence=result.kl_divergence,
            learning_rate=result.learning_rate,
            response_tokens_total=result.response_tokens_total,
            reference_active=result.reference_active,
            stages=[_stage_payload(stage) for stage in result.stages],
            weight_version=weight_version,
            metrics={},
        )
        if request.step_id is not None:
            self._step_results[int(request.step_id)] = response.model_copy(deep=True)
        return response

    def save_checkpoint(self, request: SaveCheckpointRequest) -> SaveCheckpointResponse:
        Path(request.path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "controller_state": dict(request.controller_state),
            "backend_states": {
                "actor": self._actor_backend.export_state(),
                "reference": (
                    self._reference_backend.export_state()
                    if self._reference_backend is not None
                    else None
                ),
            },
            "checkpoint_metadata": dict(request.checkpoint_metadata),
        }
        torch.save(payload, request.path)
        return SaveCheckpointResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            path=request.path,
        )

    def load_checkpoint(self, request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        checkpoint = torch.load(request.path, weights_only=False)
        backend_states = checkpoint.get("backend_states")
        if not isinstance(backend_states, dict):
            raise ValueError("Checkpoint is missing backend_states.")
        actor_state = backend_states.get("actor")
        if not isinstance(actor_state, dict):
            raise ValueError("Checkpoint is missing backend_states.actor.")
        self._actor_backend.load_state(actor_state)

        reference_state = backend_states.get("reference")
        if self._reference_backend is None:
            if reference_state is not None:
                raise ValueError(
                    "Checkpoint contains reference backend state, but no reference backend is configured."
                )
        else:
            if not isinstance(reference_state, dict):
                raise ValueError(
                    "Checkpoint is missing backend_states.reference for a configured reference backend."
                )
            self._reference_backend.load_state(reference_state)

        checkpoint_metadata = checkpoint.get("checkpoint_metadata")
        if not isinstance(checkpoint_metadata, dict):
            checkpoint_metadata = None

        if self._synchronize_serving and self._serving_backend is not None:
            serving_metadata = checkpoint_metadata.get("serving") if checkpoint_metadata else None
            if isinstance(serving_metadata, dict):
                restore_weight_version_state = getattr(
                    self._serving_backend,
                    "restore_weight_version_state",
                    None,
                )
                if callable(restore_weight_version_state):
                    restore_weight_version_state(serving_metadata.get("weight_version_state"))
            controller_state = dict(checkpoint.get("controller_state", {}))
            self._actor_backend.sync_weights_to(
                self._serving_backend,
                source_training_step=int(controller_state.get("total_steps", 0)),
                source_epoch=int(controller_state.get("epoch", 0)) + 1,
                origin="resume",
            )

        return LoadCheckpointResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            controller_state=dict(checkpoint.get("controller_state", {})),
            checkpoint_metadata=checkpoint_metadata,
        )

    def status(self) -> StatusResponse:
        world_size = int(getattr(self._actor_backend, "world_size", 1))
        return StatusResponse(
            status=ComponentStatus(
                name="learner",
                phase="Ready",
                healthy=True,
                ready_replica_count=world_size,
                desired_replica_count=world_size,
                metadata={
                    "backend": self._actor_backend.backend_name,
                    "optimizer": self._actor_backend.optimizer_name,
                },
            )
        )

    def _publish_weight_version(self, request: OptimizeStepRequest) -> WeightVersionRef:
        if self._synchronize_serving:
            if self._serving_backend is None:
                raise RuntimeError(
                    "synchronize_serving=True requires a serving_backend on LearnerService."
                )
            activated = self._actor_backend.sync_weights_to(
                self._serving_backend,
                source_training_step=request.step_id,
                source_epoch=request.epoch,
                origin="sync",
            )
            return _readable_weight_version(activated)

        if self._publish_dir is None:
            raise RuntimeError("No publish_dir is configured for learner weight publication.")
        version_id = self._next_version_id()
        version_dir = self._publish_dir / f"version-{version_id:08d}"
        version_dir.mkdir(parents=True, exist_ok=True)
        weight_path = version_dir / "model_state.pt"
        torch.save(self._actor_backend.model.state_dict(), weight_path)
        return WeightVersionRef(
            version_id=version_id,
            artifact_uri=str(weight_path),
            checksum=_sha256_file(weight_path),
            size_bytes=int(weight_path.stat().st_size),
            source_training_step=request.step_id,
            source_epoch=request.epoch,
            origin="sync",
        )

    def _next_version_id(self) -> int:
        existing = sorted(self._publish_dir.glob("version-*")) if self._publish_dir is not None else []
        if not existing:
            return 1
        latest = existing[-1].name.rsplit("-", 1)[-1]
        try:
            return int(latest) + 1
        except ValueError:
            return len(existing) + 1


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
