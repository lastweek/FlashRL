"""Local in-process adapters for distributed FlashRL protocols."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any

import torch

from flashrl.framework.admin import utc_now_iso
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    ComponentStatus,
    GenerateGroupedRequest,
    GenerateGroupedResponse,
    GeneratedSamplePayload,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    RewardBatchRequest,
    RewardBatchResponse,
    RolloutBatchRequest,
    RolloutBatchResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    StageResultPayload,
    StatusResponse,
    WeightVersionRef,
)
from flashrl.framework.observability import StageResult
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.base import BaseRolloutGenerator
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.config import GrpoConfig
from flashrl.framework.training import ActorTrainingBackend, ReferenceTrainingBackend
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


def _resolve_weight_artifact_path(artifact_uri: str) -> Path:
    path = Path(artifact_uri)
    if path.is_dir():
        candidate = path / "model_state.pt"
        if candidate.exists():
            return candidate
    return path


class LocalRolloutClient:
    """In-process controller adapter over ``UserDefinedRollout``."""

    def __init__(self, rollout_generator: BaseRolloutGenerator) -> None:
        self._rollout_generator = rollout_generator

    def rollout_batch(self, request: RolloutBatchRequest) -> RolloutBatchResponse:
        original_config = self._rollout_generator.config
        if request.rollout_config is not None:
            self._rollout_generator.config = request.rollout_config.model_copy(deep=True)
        try:
            prompts, rollouts, prompt_indices, candidate_indices = (
                self._rollout_generator.generate_grouped(request.prompts, request.group_size)
            )
        finally:
            self._rollout_generator.config = original_config

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


class LocalRewardClient:
    """In-process controller adapter over ``UserDefinedReward``."""

    def __init__(self, reward: UserDefinedReward) -> None:
        self._reward = reward

    def reward_batch(self, request: RewardBatchRequest) -> RewardBatchResponse:
        rewards = self._reward.compute_batch(request.rollouts)
        reward_values = [reward.reward for reward in rewards]
        reward_mean = sum(reward_values) / len(reward_values) if reward_values else 0.0
        return RewardBatchResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            rewards=rewards,
            metrics={"reward_mean": reward_mean, "sample_count": len(rewards)},
        )

    def status(self) -> StatusResponse:
        return StatusResponse(
            status=ComponentStatus(
                name="reward",
                phase="Ready",
                healthy=True,
                ready_replica_count=1,
                desired_replica_count=1,
            )
        )


class LocalServingClient:
    """In-process controller adapter over one ``ServingBackend``."""

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


class LocalLearnerClient:
    """In-process controller adapter over actor/reference learner backends."""

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
                    "synchronize_serving=True requires a serving_backend on LocalLearnerClient."
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
