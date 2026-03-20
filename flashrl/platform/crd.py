"""FlashRLJob CRD models and manifest helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flashrl.framework.config import BuilderSpec
from flashrl.framework.config import (
    CheckpointingConfig,
    GrpoConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)


ELASTIC_COMPONENTS = ("serving", "rollout", "reward")
FIXED_COMPONENTS = ("controller", "learner")


class DatasetSpec(BaseModel):
    """Dataset reference resolved by the controller."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hook", "uri"] = "hook"
    uri: str | None = None
    format: Literal["jsonl", "json", "parquet", "huggingface"] = "jsonl"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_source(self) -> "DatasetSpec":
        if self.type == "uri" and not self.uri:
            raise ValueError("dataset.uri is required when dataset.type='uri'.")
        if self.type == "hook" and self.uri is not None:
            raise ValueError("dataset.uri must be omitted when dataset.type='hook'.")
        return self


class UserCodeSpec(BaseModel):
    """Structured builder hooks resolved inside runtime images."""

    model_config = ConfigDict(extra="forbid")

    dataset: BuilderSpec | None = None
    rollout: BuilderSpec
    reward: BuilderSpec


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


class AutoscalingSpec(BaseModel):
    """Operator-driven scaling policy for one elastic workload."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    targetInflightPerReplica: int = Field(default=1, ge=1)
    targetP95LatencySeconds: float | None = Field(default=None, ge=0.0)
    scaleUpStep: int = Field(default=1, ge=1)
    scaleDownStep: int = Field(default=1, ge=1)
    scaleUpCooldownSeconds: int = Field(default=30, ge=0)
    scaleDownCooldownSeconds: int = Field(default=60, ge=0)
    scaleDownStabilizationSeconds: int = Field(default=300, ge=0)


class FailurePolicySpec(BaseModel):
    """Bounded operator recovery policy for one workload."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["replace-pod", "restart-workload"] = "replace-pod"
    readinessTimeoutSeconds: int = Field(default=90, ge=1)
    backoffSeconds: int = Field(default=30, ge=0)
    maxRecoveryAttempts: int = Field(default=5, ge=0)


class ResourceRequirementsSpec(BaseModel):
    """Minimal Kubernetes resource requests/limits."""

    model_config = ConfigDict(extra="forbid")

    requests: dict[str, str] = Field(default_factory=dict)
    limits: dict[str, str] = Field(default_factory=dict)


class WorkloadSpec(BaseModel):
    """Per-component pod policy."""

    model_config = ConfigDict(extra="forbid")

    replicas: ReplicaRange | None = None
    autoscaling: AutoscalingSpec | None = None
    failurePolicy: FailurePolicySpec | None = None
    resources: ResourceRequirementsSpec = Field(default_factory=ResourceRequirementsSpec)
    env: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class ImageSpec(BaseModel):
    """Runtime image references used across workload families."""

    model_config = ConfigDict(extra="forbid")

    runtime: str
    serving: str
    training: str
    pullPolicy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"


class ArtifactStorageSpec(BaseModel):
    """Object storage location for checkpoints and published weights."""

    model_config = ConfigDict(extra="forbid")

    uriPrefix: str
    secretRef: str | None = None


class PersistentVolumeClaimSpec(BaseModel):
    """PVC policy for shared in-cluster job storage."""

    model_config = ConfigDict(extra="forbid")

    create: bool = True
    claimName: str | None = None
    size: str = "5Gi"
    storageClassName: str | None = None
    accessModes: list[Literal["ReadWriteOnce", "ReadWriteMany", "ReadOnlyMany"]] = Field(
        default_factory=lambda: ["ReadWriteOnce"]
    )

    @model_validator(mode="after")
    def validate_claim(self) -> "PersistentVolumeClaimSpec":
        if not self.create and not self.claimName:
            raise ValueError("sharedStorage.claim.claimName is required when create=false.")
        return self


class SharedStorageSpec(BaseModel):
    """Shared PVC-mounted storage used for weights and checkpoints."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mountPath: str = "/var/lib/flashrl/shared"
    checkpointsSubPath: str = "checkpoints"
    weightsSubPath: str = "weights"
    claim: PersistentVolumeClaimSpec = Field(default_factory=PersistentVolumeClaimSpec)

    @model_validator(mode="after")
    def validate_paths(self) -> "SharedStorageSpec":
        if not str(self.mountPath).startswith("/"):
            raise ValueError("sharedStorage.mountPath must be an absolute container path.")
        for field_name in ("checkpointsSubPath", "weightsSubPath"):
            raw_value = str(getattr(self, field_name))
            if raw_value.startswith("/"):
                raise ValueError(f"sharedStorage.{field_name} must be a relative path.")
            if ".." in PurePosixPath(raw_value).parts:
                raise ValueError(f"sharedStorage.{field_name} must not contain '..'.")
        return self


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


class WorkloadStatus(BaseModel):
    """Per-workload status tracked in the CRD."""

    model_config = ConfigDict(extra="forbid")

    phase: str = "Pending"
    readyReplicas: int = 0
    availableReplicas: int = 0
    desiredReplicas: int = 0
    worldSize: int | None = None
    activeWeightVersion: dict[str, Any] | None = None
    restartCount: int = 0
    recoveryAttempts: int = 0
    lastScaleAt: str | None = None
    lastObservedAt: str | None = None
    lastError: str | None = None
    lowLoadSince: str | None = None
    unreadySince: str | None = None
    lastRecoveryAt: str | None = None


class ConditionStatus(BaseModel):
    """Kubernetes-style CRD condition."""

    model_config = ConfigDict(extra="forbid")

    type: str
    status: Literal["True", "False", "Unknown"]
    reason: str
    lastTransitionTime: str
    message: str | None = None


def _default_failure_policy(component: str) -> FailurePolicySpec:
    if component == "controller":
        return FailurePolicySpec(
            mode="replace-pod",
            readinessTimeoutSeconds=120,
            backoffSeconds=30,
            maxRecoveryAttempts=3,
        )
    if component == "learner":
        return FailurePolicySpec(
            mode="restart-workload",
            readinessTimeoutSeconds=180,
            backoffSeconds=60,
            maxRecoveryAttempts=3,
        )
    return FailurePolicySpec(
        mode="replace-pod",
        readinessTimeoutSeconds=90,
        backoffSeconds=30,
        maxRecoveryAttempts=5,
    )


def _default_autoscaling(component: str, replicas: ReplicaRange | None) -> AutoscalingSpec:
    if component in FIXED_COMPONENTS:
        return AutoscalingSpec(enabled=False)
    bounds = replicas or ReplicaRange()
    return AutoscalingSpec(enabled=bounds.max > bounds.min)


class FlashRLJobSpec(BaseModel):
    """Top-level FlashRLJob spec."""

    model_config = ConfigDict(extra="forbid")

    suspend: bool = False
    framework: FrameworkSpec
    dataset: DatasetSpec
    images: ImageSpec
    userCode: UserCodeSpec
    sharedStorage: SharedStorageSpec = Field(default_factory=SharedStorageSpec)
    controller: WorkloadSpec = Field(default_factory=WorkloadSpec)
    learner: WorkloadSpec = Field(default_factory=WorkloadSpec)
    serving: WorkloadSpec = Field(default_factory=WorkloadSpec)
    rollout: WorkloadSpec = Field(default_factory=WorkloadSpec)
    reward: WorkloadSpec = Field(default_factory=WorkloadSpec)
    storage: StorageSpec
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)

    @model_validator(mode="after")
    def validate_workloads(self) -> "FlashRLJobSpec":
        actor_world_size = int(self.framework.actor.dp_size)

        self.controller.replicas = self.controller.replicas or ReplicaRange(min=1, max=1)
        if self.controller.replicas.min != 1 or self.controller.replicas.max != 1:
            raise ValueError("controller replicas must stay fixed at 1 in v1.")

        if self.learner.replicas is not None:
            raise ValueError("learner replicas are derived from framework.actor.dp_size in platform mode.")

        for component_name in ELASTIC_COMPONENTS:
            workload = getattr(self, component_name)
            workload.replicas = workload.replicas or ReplicaRange()

        component_defaults = {
            "controller": self.controller,
            "learner": self.learner,
            "serving": self.serving,
            "rollout": self.rollout,
            "reward": self.reward,
        }
        for component_name, workload in component_defaults.items():
            if workload.failurePolicy is None:
                workload.failurePolicy = _default_failure_policy(component_name)
            if workload.autoscaling is None:
                workload.autoscaling = _default_autoscaling(component_name, workload.replicas)

        if self.controller.autoscaling.enabled:
            raise ValueError("controller autoscaling is invalid in v1.")
        if self.learner.autoscaling.enabled:
            raise ValueError("learner autoscaling is invalid in v1.")
        if self.learner.failurePolicy.mode != "restart-workload":
            raise ValueError("learner failurePolicy.mode must be restart-workload in v1.")
        if self.controller.failurePolicy.mode != "replace-pod":
            raise ValueError("controller failurePolicy.mode must be replace-pod in v1.")

        self.learner.failurePolicy = self.learner.failurePolicy or _default_failure_policy("learner")
        self.controller.failurePolicy = self.controller.failurePolicy or _default_failure_policy("controller")
        self.learner.autoscaling = AutoscalingSpec(enabled=False)
        self.controller.autoscaling = AutoscalingSpec(enabled=False)

        if self.dataset.type == "hook" and self.userCode.dataset is None:
            raise ValueError("userCode.dataset is required when dataset.type='hook'.")
        if self.dataset.type == "uri" and self.userCode.dataset is not None:
            raise ValueError("userCode.dataset must be omitted when dataset.type='uri'.")
        if self.sharedStorage.enabled and self.dataset.type == "uri":
            if self.dataset.uri is not None and str(self.dataset.uri).startswith(("s3://", "gs://", "hf://")):
                raise ValueError(
                    "dataset.type='uri' with sharedStorage.enabled currently requires a mounted file path, not object storage."
                )
        if actor_world_size < 1:
            raise ValueError("framework.actor.dp_size must be >= 1.")
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
    components: dict[str, WorkloadStatus] = Field(default_factory=dict)
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


def _resolve_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic ``$defs`` references into one CRD-safe schema."""
    definitions = deepcopy(schema.pop("$defs", {}))
    resolved_definitions: dict[str, dict[str, Any]] = {}
    resolving: set[str] = set()

    def _resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = str(node["$ref"])
            if not ref.startswith("#/$defs/"):
                return {key: _resolve(value) for key, value in node.items() if key != "$ref"}
            target_name = ref.rsplit("/", 1)[-1]
            if target_name in resolved_definitions:
                merged = deepcopy(resolved_definitions[target_name])
            elif target_name in resolving:
                merged = {"type": "object", "x-kubernetes-preserve-unknown-fields": True}
            else:
                resolving.add(target_name)
                target = deepcopy(definitions[target_name])
                merged = _resolve(target)
                resolving.remove(target_name)
                resolved_definitions[target_name] = deepcopy(merged)
            for key, value in node.items():
                if key == "$ref":
                    continue
                merged[key] = _resolve(value)
            return merged
        return {
            key: _resolve(value)
            for key, value in node.items()
            if key not in {"title", "$defs"}
        }

    return _resolve(schema)


def flashrljob_openapi_schema() -> dict[str, Any]:
    """Return a dereferenced OpenAPI schema for the FlashRLJob CRD."""
    schema = FlashRLJob.model_json_schema(mode="validation")
    resolved = _resolve_local_refs(schema)
    metadata_schema = resolved.setdefault("properties", {}).setdefault("metadata", {"type": "object"})
    metadata_schema.clear()
    metadata_schema.update(
        {
            "type": "object",
            "x-kubernetes-preserve-unknown-fields": True,
        }
    )
    return resolved


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
                        "openAPIV3Schema": flashrljob_openapi_schema(),
                    },
                    "subresources": {"status": {}},
                }
            ],
        },
    }
