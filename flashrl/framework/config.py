"""Configuration models for FlashRL components."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class TrainerConfig(BaseModel):
    """Configuration for the trainer."""

    learning_rate: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 10
    kl_coefficient: float = 0.0
    gamma: float = 1.0  # Discount factor
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainerConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerConfig":
        """Load config from dictionary."""
        return cls(**data)


class ModelConfig(BaseModel):
    """Configuration for model loading."""

    model_name: str
    device: str | None = None  # None = auto-detect
    dtype: str = "float32"
    max_length: int = 2048
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    num_threads: int = 1  # Default to 1 CPU thread
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Load config from dictionary."""
        return cls(**data)


class RolloutConfig(BaseModel):
    """Configuration for rollout generation."""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    num_return_sequences: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RolloutConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutConfig":
        """Load config from dictionary."""
        return cls(**data)


class RewardConfig(BaseModel):
    """Configuration for reward computation."""

    reward_model_name: str | None = None
    scale: float = 1.0
    normalize: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RewardConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardConfig":
        """Load config from dictionary."""
        return cls(**data)


class LoggingConfig(BaseModel):
    """Configuration for run logging and terminal UX."""

    level: str = "INFO"
    log_dir: str | Path = ".flashrl-runs"
    log_every_steps: int = 1
    sample_every_steps: int = 10
    console: bool = True
    file: bool = True
    rich_progress: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoggingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        """Load config from dictionary."""
        return cls(**data)
