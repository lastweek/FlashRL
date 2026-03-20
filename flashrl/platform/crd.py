"""FlashRLJob CRD models and manifest helpers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flashrl.framework.config import CheckpointingConfig, GrpoConfig, ServingConfig, TrainerConfig, TrainingConfig


class DatasetSpec(BaseModel):
    """Dataset reference resolved by the controller."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    format: Literal["jsonl", "json", "parquet", "huggingface"] = "jsonl"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplicaRange(BaseModel):
    """Elastic replica bounds for one service pool."""

    model_config = ConfigDict(extra="forbid")

    min: int = Field(default=1, ge=1)
    max: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ReplicaRange":
        if self.max < self.min:
            raise ValueError("replicas.max must be >= replicas.min.")
        return self


class ResourceRequirementsSpec(BaseModel):
    """Minimal Kubernetes resource requests/limits."""

    model_config = ConfigDict(extra="forbid")

    requests: dict[str, str] = Field(default_factory=dict)
    limits: dict[str, str] = Field(default_factory=dict)


class WorkloadSpec(BaseModel):
    """Per-component image and pod policy."""

    model_config = ConfigDict(extra="forbid")

    image: str
    replicas: ReplicaRange | None = None
    resources: ResourceRequirementsSpec = Field(default_factory=ResourceRequirementsSpec)
    env: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class ArtifactStorageSpec(BaseModel):
    """Object storage location for checkpoints and published weights."""

    model_config = ConfigDict(extra="forbid")

    uriPrefix: str
    secretRef: str | None = None


class StorageSpec(BaseModel):
    """Durable storage references for one job."""

    model_config = ConfigDict(extra="forbid")

    checkpoints: ArtifactStorageSpec
    weights: ArtifactStorageSpec


class ObservabilitySpec(BaseModel):
    """Platform-level observability toggles."""

    model_config = ConfigDict(extra="forbid")

    adminEnabled: bool = True
    metricsEnabled: bool = True


class FrameworkSpec(BaseModel):
    """Framework-side run semantics embedded in the CRD."""

    model_config = ConfigDict(extra="forbid")

    actor: TrainingConfig
    reference: TrainingConfig | None = None
    serving: ServingConfig
    trainer: TrainerConfig
    grpo: GrpoConfig

    @model_validator(mode="after")
    def validate_platform_mode(self) -> "FrameworkSpec":
        if self.grpo.kl_coefficient > 0.0 and self.reference is None:
            raise ValueError("framework.reference is required when grpo.kl_coefficient > 0.")
        if self.grpo.kl_coefficient <= 0.0 and self.reference is not None:
            raise ValueError("framework.reference must be omitted when grpo.kl_coefficient <= 0.")
        if self.serving.runtime_python is not None:
            raise ValueError("framework.serving.runtime_python is invalid in platform mode.")
        return self


class ProgressStatus(BaseModel):
    """Controller progress persisted in CRD status."""

    model_config = ConfigDict(extra="forbid")

    currentEpoch: int = 0
    currentStep: int = 0
    lastCompletedStep: int = 0


class WeightVersionStatus(BaseModel):
    """Desired and active weight-version status."""

    model_config = ConfigDict(extra="forbid")

    desired: dict[str, Any] | None = None
    active: dict[str, Any] | None = None


class CheckpointStatus(BaseModel):
    """Latest checkpoint URI and timestamp."""

    model_config = ConfigDict(extra="forbid")

    latestUri: str | None = None
    lastSavedAt: str | None = None


class ComponentStatus(BaseModel):
    """Per-component readiness tracked in the CRD."""

    model_config = ConfigDict(extra="forbid")

    phase: str = "Pending"
    readyReplicas: int = 0
    desiredReplicas: int = 0
    worldSize: int | None = None
    activeWeightVersion: int | None = None


class ConditionStatus(BaseModel):
    """Kubernetes-style CRD condition."""

    model_config = ConfigDict(extra="forbid")

    type: str
    status: Literal["True", "False", "Unknown"]
    reason: str
    lastTransitionTime: str
    message: str | None = None


class FlashRLJobSpec(BaseModel):
    """Top-level FlashRLJob spec."""

    model_config = ConfigDict(extra="forbid")

    suspend: bool = False
    framework: FrameworkSpec
    dataset: DatasetSpec
    controller: WorkloadSpec
    learner: WorkloadSpec
    servingPool: WorkloadSpec
    rollout: WorkloadSpec
    reward: WorkloadSpec
    storage: StorageSpec
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)

    @model_validator(mode="after")
    def validate_workloads(self) -> "FlashRLJobSpec":
        if self.controller.replicas is not None and (
            self.controller.replicas.min != 1 or self.controller.replicas.max != 1
        ):
            raise ValueError("controller replicas must stay fixed at 1 in v1.")
        return self


class FlashRLJobStatus(BaseModel):
    """Top-level FlashRLJob status."""

    model_config = ConfigDict(extra="forbid")

    observedGeneration: int = 0
    phase: str = "Pending"
    startedAt: str | None = None
    finishedAt: str | None = None
    progress: ProgressStatus = Field(default_factory=ProgressStatus)
    weightVersion: WeightVersionStatus = Field(default_factory=WeightVersionStatus)
    checkpoint: CheckpointStatus = Field(default_factory=CheckpointStatus)
    components: dict[str, ComponentStatus] = Field(default_factory=dict)
    conditions: list[ConditionStatus] = Field(default_factory=list)
    lastError: str | None = None


class FlashRLJob(BaseModel):
    """Validated FlashRLJob custom resource."""

    model_config = ConfigDict(extra="forbid")

    apiVersion: Literal["platform.flashrl.dev/v1alpha1"] = "platform.flashrl.dev/v1alpha1"
    kind: Literal["FlashRLJob"] = "FlashRLJob"
    metadata: dict[str, Any]
    spec: FlashRLJobSpec
    status: FlashRLJobStatus = Field(default_factory=FlashRLJobStatus)

    @property
    def name(self) -> str:
        """Return the Kubernetes object name."""
        return str(self.metadata["name"])


def flashrljob_crd_manifest() -> dict[str, Any]:
    """Return the CustomResourceDefinition manifest for FlashRLJob."""
    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {"name": "flashrljobs.platform.flashrl.dev"},
        "spec": {
            "group": "platform.flashrl.dev",
            "scope": "Namespaced",
            "names": {
                "plural": "flashrljobs",
                "singular": "flashrljob",
                "kind": "FlashRLJob",
                "shortNames": ["frj"],
            },
            "versions": [
                {
                    "name": "v1alpha1",
                    "served": True,
                    "storage": True,
                    "schema": {
                        "openAPIV3Schema": {
                            "type": "object",
                            "properties": {
                                "apiVersion": {"type": "string"},
                                "kind": {"type": "string"},
                                "metadata": {"type": "object"},
                                "spec": {"type": "object"},
                                "status": {"type": "object"},
                            },
                            "required": ["apiVersion", "kind", "metadata", "spec"],
                        }
                    },
                    "subresources": {"status": {}},
                }
            ],
        },
    }
