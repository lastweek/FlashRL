"""Shared transport models for distributed FlashRL components."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from flashrl.framework.admin import utc_now_iso
from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import (
    LearnerBatch,
    Prompt,
    RewardOutput,
    RolloutOutput,
    WeightVersionInfo,
)


def _default_trace_id() -> str:
    return uuid4().hex


class RpcMessage(BaseModel):
    """Common request/response metadata for all service calls."""

    api_version: str = "v1"
    job_id: str = "local"
    run_id: str = "local"
    step_id: int | None = None
    trace_id: str = Field(default_factory=_default_trace_id)
    sent_at: str = Field(default_factory=utc_now_iso)
    deadline_ms: int | None = None


class RpcError(BaseModel):
    """Portable RPC error payload."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class WeightVersionRef(BaseModel):
    """Published learner-owned weight artifact reference."""

    version_id: int
    artifact_uri: str
    checksum: str = ""
    size_bytes: int = 0
    source_training_step: int | None = None
    source_epoch: int | None = None
    origin: Literal["startup", "sync", "resume"] = "sync"

    def to_info(self, *, activated_at: str | None = None) -> WeightVersionInfo:
        """Convert one artifact reference into a serving-side status view."""
        return WeightVersionInfo(
            version_id=self.version_id,
            source_training_step=self.source_training_step,
            source_epoch=self.source_epoch,
            activated_at=activated_at,
            model_source=self.artifact_uri,
            origin=self.origin,
        )


class ComponentStatus(BaseModel):
    """Shared status view used by services and platform aggregation."""

    name: str
    phase: str
    healthy: bool
    ready_replica_count: int = 0
    desired_replica_count: int = 0
    active_weight_version: WeightVersionInfo | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageResultPayload(BaseModel):
    """JSON transport form of one measured training stage."""

    name: str
    seconds: float
    metrics: dict[str, Any] = Field(default_factory=dict)


class GeneratedSamplePayload(BaseModel):
    """Stable serving output shape used by distributed rollout services."""

    text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_token_logprobs: list[float]
    log_prob: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: Any) -> "GeneratedSamplePayload":
        """Normalize one serving backend sample into a wire payload."""
        return cls(
            text=str(getattr(sample, "text", "")),
            prompt_token_ids=[int(token_id) for token_id in getattr(sample, "prompt_token_ids", [])],
            response_token_ids=[
                int(token_id) for token_id in getattr(sample, "response_token_ids", [])
            ],
            response_token_logprobs=[
                float(value)
                for value in getattr(sample, "response_token_logprobs", [])
            ],
            log_prob=float(getattr(sample, "log_prob", 0.0)),
            metadata=dict(getattr(sample, "metadata", {}) or {}),
        )


class GenerateGroupedRequest(RpcMessage):
    """Serving RPC request for grouped text generation."""

    prompts: list[str]
    group_size: int = Field(ge=1)
    generation_config: RolloutConfig | None = None
    required_weight_version: WeightVersionRef | None = None


class GenerateGroupedResponse(RpcMessage):
    """Serving RPC response for grouped text generation."""

    grouped_samples: list[list[GeneratedSamplePayload]]
    active_weight_version: WeightVersionInfo | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class RolloutBatchRequest(RpcMessage):
    """Controller-to-rollout batch generation request."""

    prompts: list[Prompt]
    group_size: int = Field(ge=1)
    rollout_config: RolloutConfig | None = None
    required_weight_version: WeightVersionRef | None = None


class RolloutBatchResponse(RpcMessage):
    """Controller-facing rollout batch response."""

    rollouts: list[RolloutOutput]
    prompt_indices: list[int] = Field(default_factory=list)
    candidate_indices: list[int] = Field(default_factory=list)
    prompt_count: int = 0
    group_size: int = 1
    weight_version: WeightVersionInfo | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class RewardBatchRequest(RpcMessage):
    """Controller-to-reward scoring request."""

    rollouts: list[RolloutOutput]


class RewardBatchResponse(RpcMessage):
    """Controller-facing reward batch response."""

    rewards: list[RewardOutput]
    metrics: dict[str, Any] = Field(default_factory=dict)


class OptimizeStepRequest(RpcMessage):
    """Controller-to-learner optimization request."""

    learner_batch: LearnerBatch
    epoch: int
    rollout_weight_version: WeightVersionInfo | None = None


class OptimizeStepResponse(RpcMessage):
    """Learner optimization response with a published weight artifact."""

    loss: float
    policy_loss: float
    kl_divergence: float
    learning_rate: float
    response_tokens_total: int
    reference_active: bool
    stages: list[StageResultPayload] = Field(default_factory=list)
    weight_version: WeightVersionRef
    metrics: dict[str, Any] = Field(default_factory=dict)


class ActivateWeightVersionRequest(RpcMessage):
    """Controller-to-serving activation request."""

    weight_version: WeightVersionRef


class ActivateWeightVersionResponse(RpcMessage):
    """Serving activation response."""

    desired_weight_version: WeightVersionRef
    active_weight_version: WeightVersionInfo | None = None
    ready_replica_count: int = 0
    total_replica_count: int = 0
    converged: bool = False
    status: ComponentStatus | None = None


class SaveCheckpointRequest(RpcMessage):
    """Learner checkpoint save request."""

    path: str
    controller_state: dict[str, Any] = Field(default_factory=dict)
    checkpoint_metadata: dict[str, Any] = Field(default_factory=dict)


class SaveCheckpointResponse(RpcMessage):
    """Learner checkpoint save response."""

    path: str


class LoadCheckpointRequest(RpcMessage):
    """Learner checkpoint load request."""

    path: str


class LoadCheckpointResponse(RpcMessage):
    """Learner checkpoint load response."""

    controller_state: dict[str, Any] = Field(default_factory=dict)
    checkpoint_metadata: dict[str, Any] | None = None


class StatusResponse(RpcMessage):
    """Generic component status response."""

    status: ComponentStatus
