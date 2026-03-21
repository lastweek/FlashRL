"""Platform-side config models and RunConfig -> FlashRLJob translation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from flashrl.framework.config import (
    BaseConfig,
    BuilderSpec,
    CheckpointingConfig,
    FlashRLConfig,
    RunConfig,
    _expand_env_vars,
    load_yaml_mapping,
)
from flashrl.platform.k8s.job import (
    DatasetSpec,
    FlashRLJob,
    FlashRLJobSpec,
    ImageSpec,
    ObservabilitySpec,
    SharedStorageSpec,
    StorageSpec,
    UserCodeSpec,
    WorkloadSpec,
)


class PlatformJobConfig(BaseConfig):
    """Kubernetes object metadata for one submitted FlashRL job."""

    name: str
    namespace: str = "default"
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class PlatformImageConfig(BaseConfig):
    """Runtime image references for cluster execution."""

    runtime: str
    serving: str
    training: str
    pullPolicy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"


class PlatformUserCodeConfig(BaseConfig):
    """Optional platform-side hook overrides."""

    dataset: BuilderSpec | None = None
    rollout: BuilderSpec | None = None
    reward: BuilderSpec | None = None


class PlatformConfig(BaseConfig):
    """Cluster-only configuration used at submit time."""

    job: PlatformJobConfig
    images: PlatformImageConfig
    storage: StorageSpec
    suspend: bool = False
    controller: WorkloadSpec = Field(default_factory=WorkloadSpec)
    learner: WorkloadSpec = Field(default_factory=WorkloadSpec)
    serving: WorkloadSpec = Field(default_factory=WorkloadSpec)
    rollout: WorkloadSpec = Field(default_factory=WorkloadSpec)
    reward: WorkloadSpec = Field(default_factory=WorkloadSpec)
    dataset: DatasetSpec | None = None
    userCode: PlatformUserCodeConfig | None = None
    sharedStorage: SharedStorageSpec = Field(default_factory=SharedStorageSpec)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlatformConfig":
        """Load one platform-only config or the platform section from a combined config."""
        data = load_yaml_mapping(path)
        if "platform" in data:
            platform = data.get("platform")
            if not isinstance(platform, dict):
                raise ValueError("config.platform must be a mapping.")
            data = platform
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlatformConfig":
        """Load one platform-only config or the platform section from a combined config."""
        expanded = dict(data)
        expanded = _expand_env_vars(expanded)
        if "platform" in expanded:
            platform = expanded.get("platform")
            if not isinstance(platform, dict):
                raise ValueError("config.platform must be a mapping.")
            expanded = platform
        return cls.model_validate(expanded)


def load_flashrl_config(path: str | Path) -> FlashRLConfig:
    """Load one combined FlashRL config file."""
    return FlashRLConfig.from_yaml(path)


def _normalize_builder(binding: str | BuilderSpec | None) -> BuilderSpec | None:
    if binding is None:
        return None
    if isinstance(binding, BuilderSpec):
        return binding.model_copy(deep=True)
    return BuilderSpec.model_validate({"import": str(binding), "kwargs": {}})


def _resolve_user_code(
    run_config: RunConfig,
    platform_config: PlatformConfig,
) -> tuple[DatasetSpec, UserCodeSpec]:
    hooks = run_config.hooks
    overrides = platform_config.userCode or PlatformUserCodeConfig()

    dataset_builder = overrides.dataset
    rollout_builder = overrides.rollout
    reward_builder = overrides.reward

    if hooks is not None:
        dataset_builder = dataset_builder or _normalize_builder(hooks.dataset_fn)
        rollout_builder = rollout_builder or _normalize_builder(hooks.rollout_fn)
        reward_builder = reward_builder or _normalize_builder(hooks.reward_fn)

    dataset = platform_config.dataset.model_copy(deep=True) if platform_config.dataset is not None else None
    if dataset is None:
        if dataset_builder is None:
            raise ValueError(
                "Platform submit requires either platform.dataset or hooks.dataset_fn / userCode.dataset."
            )
        dataset = DatasetSpec(type="hook")

    if rollout_builder is None:
        raise ValueError(
            "Platform submit requires hooks.rollout_fn or platform userCode.rollout."
        )
    if reward_builder is None:
        raise ValueError(
            "Platform submit requires hooks.reward_fn or platform userCode.reward."
        )

    return (
        dataset,
        UserCodeSpec(
            dataset=dataset_builder.model_copy(deep=True) if dataset_builder is not None else None,
            rollout=rollout_builder.model_copy(deep=True),
            reward=reward_builder.model_copy(deep=True),
        ),
    )


def build_flashrl_job(
    *,
    run_config: RunConfig,
    platform_config: PlatformConfig,
) -> FlashRLJob:
    """Translate one local run config plus one platform config into a FlashRLJob."""
    dataset, user_code = _resolve_user_code(run_config, platform_config)
    metadata: dict[str, Any] = {
        "name": platform_config.job.name,
        "namespace": platform_config.job.namespace,
    }
    if platform_config.job.labels:
        metadata["labels"] = dict(platform_config.job.labels)
    if platform_config.job.annotations:
        metadata["annotations"] = dict(platform_config.job.annotations)

    spec = FlashRLJobSpec(
        suspend=platform_config.suspend,
        framework={
            "actor": run_config.actor.model_dump(mode="json"),
            "reference": (
                run_config.reference.model_dump(mode="json")
                if run_config.reference is not None
                else None
            ),
            "serving": run_config.serving.model_dump(mode="json"),
            "controller": run_config.controller.model_dump(mode="json"),
            "grpo": run_config.grpo.model_dump(mode="json"),
            "logging": run_config.logging.model_dump(mode="json"),
            "metrics": run_config.metrics.model_dump(mode="json"),
            "admin": run_config.admin.model_dump(mode="json"),
        },
        dataset=dataset,
        images=ImageSpec.model_validate(platform_config.images.model_dump(mode="json")),
        userCode=user_code,
        sharedStorage=platform_config.sharedStorage.model_copy(deep=True),
        controller=platform_config.controller.model_copy(deep=True),
        learner=platform_config.learner.model_copy(deep=True),
        serving=platform_config.serving.model_copy(deep=True),
        rollout=platform_config.rollout.model_copy(deep=True),
        reward=platform_config.reward.model_copy(deep=True),
        storage=platform_config.storage.model_copy(deep=True),
        checkpointing=platform_config.checkpointing.model_copy(deep=True),
        observability=platform_config.observability.model_copy(deep=True),
    )
    return FlashRLJob(metadata=metadata, spec=spec)
