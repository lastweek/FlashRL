"""Configuration models for FlashRL components and YAML-driven runs."""

import os
from pathlib import Path
import re
from typing import Any, Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand `${VAR}` placeholders in config values."""
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        resolved = os.environ.get(variable)
        if resolved is None:
            raise ValueError(
                f"Missing required environment variable '{variable}' while loading config."
            )
        return resolved

    return _ENV_VAR_PATTERN.sub(replace, value)


class BaseConfig(BaseModel):
    """Base configuration class with common loading methods."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        data = _expand_env_vars(data)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Load config from dictionary."""
        return cls(**data)


class TrainerConfig(BaseConfig):
    """Configuration for the trainer."""

    learning_rate: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 10
    seed: int = 42
    shuffle_each_epoch: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseConfig):
    """Configuration for model loading."""

    model_name: str
    device: str | None = None  # None = auto-detect
    dtype: str = "float32"
    max_length: int = 2048
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    num_threads: int = 1  # Default to 1 CPU thread
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServingConfig(ModelConfig):
    """Configuration for the serving model copy."""

    backend: Literal["huggingface", "vllm"] = "huggingface"
    runtime_python: str | None = None
    num_replicas: int = Field(default=1, ge=1)
    vllm_args: list[str] = Field(default_factory=list)
    debug_live_rollout: bool = False


class RolloutConfig(BaseConfig):
    """Configuration for rollout generation."""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardConfig(BaseConfig):
    """Configuration for reward computation."""

    reward_model_name: str | None = None
    scale: float = 1.0
    normalize: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GrpoConfig(BaseConfig):
    """Configuration for grouped GRPO rollout and optimization."""

    model_config = ConfigDict(extra="forbid")

    group_size: int = Field(default=2, ge=2)
    clip_ratio: float = 0.2
    kl_coefficient: float = 0.0
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseConfig):
    """Configuration for run logging and terminal UX."""

    level: str = "INFO"
    log_dir: str | Path = "logs"
    log_every_steps: int = 1
    sample_every_steps: int = 10
    console: bool = True
    file: bool = True
    console_mode: Literal["compact", "verbose"] = "compact"
    rich_progress: bool = False


class MetricsConfig(BaseConfig):
    """Configuration for Prometheus/Grafana observability."""

    enabled: bool = True
    backend: Literal["pushgateway"] = "pushgateway"
    pushgateway_url: str = "http://localhost:9091"
    job_name: str = "flashrl"


class RuntimeConfig(BaseConfig):
    """Runtime options that sit outside the model/trainer sections."""

    reference_enabled: bool = False
    reference_device: str | None = None
    admin_enabled: bool = True
    admin_host: str = "127.0.0.1"
    admin_port: int = Field(default=0, ge=0, le=65535)


class HookConfig(BaseConfig):
    """Python import-string hooks used by YAML-driven runs."""

    rollout_fn: str
    reward_fn: str
    dataset_fn: str


class CommonConfig(BaseConfig):
    """Optional shared model defaults for both training and serving."""

    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None
    device: str | None = None
    dtype: str | None = None
    max_length: int | None = None
    load_in_8bit: bool | None = None
    trust_remote_code: bool | None = None
    metadata: dict[str, Any] | None = None


class TrainingSectionConfig(CommonConfig):
    """YAML training section: model-copy settings plus loop settings."""

    num_threads: int | None = None
    learning_rate: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 10
    seed: int = 42
    shuffle_each_epoch: bool = True


class ServingSectionConfig(CommonConfig):
    """YAML serving section: serving model-copy settings only."""

    num_threads: int | None = None
    backend: Literal["huggingface", "vllm"] = "huggingface"
    runtime_python: str | None = None
    num_replicas: int = Field(default=1, ge=1)
    vllm_args: list[str] = Field(default_factory=list)
    debug_live_rollout: bool = False


class RunConfig(BaseConfig):
    """Top-level YAML config for one FlashRL run."""

    model_config = ConfigDict(extra="forbid")

    common: CommonConfig | None = None
    training: TrainingSectionConfig
    serving: ServingSectionConfig
    grpo: GrpoConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    hooks: HookConfig
