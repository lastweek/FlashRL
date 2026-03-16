"""Configuration models for FlashRL components and YAML-driven runs."""

import os
from pathlib import Path
import re
from typing import Any, Literal
from enum import Enum

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class LossPreset(str, Enum):
    """GRPO loss preset options."""
    GRPO_NAIVE = "grpo_naive"
    PPO_CLIPPED = "ppo_clipped"
    DEEPSEEK_V3_2 = "deepseek_v3.2"
    KIMI_K2_5 = "kimi_k2.5"
    GLM_5 = "glm_5"
    MIMO_V2 = "mimo_v2"
    CUSTOM = "custom"


class ClippingMode(str, Enum):
    """Clipping mode options for GRPO loss."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HARD_MASK = "hard_mask"
    NONE = "none"


class KLMode(str, Enum):
    """KL divergence computation mode options."""
    NONE = "none"
    K1 = "k1"
    K3 = "k3"
    UNBIASED = "unbiased"


class AdvantageMode(str, Enum):
    """Advantage normalization mode options."""
    GROUP_CENTERED = "group_centered"
    GROUP_NORMALIZED = "group_normalized"


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

    model_config = ConfigDict(extra="forbid")

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
        data = _expand_env_vars(data)
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


class FSDP2Config(BaseConfig):
    """Configuration for the FSDP2 training backend."""

    model_config = ConfigDict(extra="forbid")

    reshard_after_forward: bool = True
    use_orig_params: bool = True
    cpu_offload: bool = False


class TrainingConfig(ModelConfig):
    """Configuration for the training model copy and backend."""

    backend: Literal["huggingface", "fsdp2"] = "huggingface"
    dp_size: int = Field(default=1, ge=1)
    fsdp2: FSDP2Config = Field(default_factory=FSDP2Config)

    @model_validator(mode="after")
    def validate_backend_shape(self) -> "TrainingConfig":
        """Reject invalid backend/config combinations early."""
        if self.backend == "huggingface" and self.dp_size != 1:
            raise ValueError("training.dp_size must be 1 when training.backend='huggingface'.")
        return self


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

    # Core GRPO parameters
    group_size: int = Field(default=2, ge=2)
    kl_coefficient: float = 0.0
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ===== Loss Configuration =====
    # Preset selection
    loss_preset: Literal[
        "grpo_naive",  # PPO-style symmetric clipping
        "ppo_clipped",  # Alias for grpo_naive
        "deepseek_v3.2",  # Token-level GRPO + reference KL + sequence masking
        "kimi_k2.5",  # Hard token clip + soft log-ratio penalty
        "glm_5",  # Train/infer mismatch gate + teacher distillation
        "mimo_v2",  # Teacher distillation + importance gating
        "custom",  # Use explicit config parameters
    ] = "grpo_naive"

    # Clipping mode
    clipping_mode: Literal["symmetric", "asymmetric", "hard_mask", "none"] = "symmetric"
    clip_ratio: float = 0.2  # For symmetric clipping
    clip_ratio_lower: float | None = None  # For asymmetric clipping
    clip_ratio_upper: float | None = None  # For asymmetric clipping
    clip_log_ratio_alpha: float | None = None  # For Kimi hard token mask
    clip_log_ratio_beta: float | None = None  # For Kimi hard token mask

    # KL divergence computation
    kl_mode: Literal["none", "k1", "k3", "unbiased"] = "k3"
    kl_target: float | None = None  # For hard KL constraint
    kl_hard_threshold: float | None = None  # Reject updates if KL exceeds

    # Log-ratio penalty (Kimi-style)
    log_ratio_penalty_coefficient: float = 0.0  # τ for soft quadratic penalty

    # IcePop token gate (GLM-5 train/infer mismatch, per-token)
    enable_icepop_token_gate: bool = False
    icepop_token_gate_beta: float = 2.0  # β for pop(ρ, 1/β, β)

    # Sequence-level off-policy masking (DeepSeek-V3.2)
    enable_off_policy_sequence_masking: bool = False
    off_policy_sequence_masking_delta: float = 2.0  # δ threshold

    # Importance weight gating (MiMo-style)
    enable_importance_gating: bool = False
    importance_epsilon_low: float | None = None  # ε_low
    importance_epsilon_high: float | None = None  # ε_high

    # Advantage normalization
    advantage_normalization: bool = True
    advantage_mode: Literal["group_centered", "group_normalized"] = "group_centered"

    # Entropy regularization
    entropy_coefficient: float = 0.0
    entropy_decay_rate: float = 1.0  # 1.0 = no decay

    # Cache for resolved preset (internal use only)
    resolved_config_cache: "GrpoConfig | None" = Field(default=None, exclude=True)

    def get_resolved_config(self) -> "GrpoConfig":
        """Get the preset-resolved configuration, cached for efficiency.

        This resolves the loss_preset to explicit parameters once and caches
        the result to avoid repeated computation during training.
        """
        if self.resolved_config_cache is None:
            from flashrl.framework.trainer.grpo.loss_variants import resolve_loss_preset

            self.resolved_config_cache = resolve_loss_preset(self)
        return self.resolved_config_cache


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


class TensorBoardMetricsConfig(BaseConfig):
    """Configuration for TensorBoard scalar logging."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class PushgatewayMetricsConfig(BaseConfig):
    """Configuration for Pushgateway-backed Prometheus metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    url: str = "http://localhost:9091"
    job_name: str = "flashrl"


class MetricsConfig(BaseConfig):
    """Configuration for run metrics sinks."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    tensorboard: TensorBoardMetricsConfig = Field(default_factory=TensorBoardMetricsConfig)
    pushgateway: PushgatewayMetricsConfig = Field(default_factory=PushgatewayMetricsConfig)


class CheckpointingConfig(BaseConfig):
    """Configuration for managed training checkpoints."""

    model_config = ConfigDict(extra="forbid")

    save_every_steps: int | None = Field(default=None, ge=1)
    save_on_run_end: bool = False
    directory: str | Path | None = None
    final_path: str | Path | None = None
    resume_from: str | Path | Literal["latest"] | None = None

    @model_validator(mode="after")
    def validate_resume_target(self) -> "CheckpointingConfig":
        """Require a managed directory when resuming from the latest checkpoint."""
        if self.resume_from == "latest" and self.directory is None:
            raise ValueError(
                "checkpointing.directory is required when checkpointing.resume_from='latest'."
            )
        return self


class AdminConfig(BaseConfig):
    """Admin server configuration."""

    model_config = ConfigDict(extra="forbid")

    admin_enabled: bool = True
    admin_host: str = "127.0.0.1"
    admin_port: int = Field(default=0, ge=0, le=65535)


class HookConfig(BaseConfig):
    """Python import-string hooks used by YAML-driven runs."""

    rollout_fn: str
    reward_fn: str
    dataset_fn: str


class RunConfig(BaseConfig):
    """Top-level YAML config for one FlashRL run."""

    model_config = ConfigDict(extra="forbid")

    actor: TrainingConfig
    reference: TrainingConfig | None = None
    serving: ServingConfig
    trainer: TrainerConfig
    grpo: GrpoConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    hooks: HookConfig | None = None

    @model_validator(mode="after")
    def validate_reference_policy(self) -> "RunConfig":
        """Require an explicit reference backend exactly when KL is enabled."""
        if self.grpo.kl_coefficient > 0.0 and self.reference is None:
            raise ValueError(
                "run_config.reference is required when grpo.kl_coefficient > 0."
            )
        if self.grpo.kl_coefficient <= 0.0 and self.reference is not None:
            raise ValueError(
                "run_config.reference must be omitted when grpo.kl_coefficient <= 0."
            )
        return self
