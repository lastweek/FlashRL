"""Configuration models for FlashRL components."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class with common loading methods."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
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
    kl_coefficient: float = 0.0
    gamma: float = 1.0  # Discount factor
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


class RolloutConfig(BaseConfig):
    """Configuration for rollout generation."""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    num_return_sequences: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardConfig(BaseConfig):
    """Configuration for reward computation."""

    reward_model_name: str | None = None
    scale: float = 1.0
    normalize: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseConfig):
    """Configuration for run logging and terminal UX."""

    level: str = "INFO"
    log_dir: str | Path = ".flashrl-runs"
    log_every_steps: int = 1
    sample_every_steps: int = 10
    console: bool = True
    file: bool = True
    rich_progress: bool = False
