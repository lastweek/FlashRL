"""Framework layer: Core RL training APIs."""

from flashrl.framework.backends.training import TrainingBackend
from flashrl.framework.backends.serving import ServingBackend
from flashrl.framework.config import (
    LoggingConfig,
    ModelConfig,
    RewardConfig,
    RolloutConfig,
    TrainerConfig,
)
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    ToolCall,
    ToolResult,
    TrainingBatch,
)

__all__ = [
    # Config
    "TrainerConfig",
    "ModelConfig",
    "RolloutConfig",
    "RewardConfig",
    "LoggingConfig",
    # Data models
    "Prompt",
    "Message",
    "Conversation",
    "ToolCall",
    "ToolResult",
    "RolloutOutput",
    "RewardOutput",
    "TrainingBatch",
    # Backends
    "TrainingBackend",
    "ServingBackend",
]
