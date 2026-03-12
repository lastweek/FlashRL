"""Framework layer: Core RL training APIs."""

from flashrl.framework.backends.training import TrainingBackend
from flashrl.framework.config import (
    CommonConfig,
    GrpoConfig,
    HookConfig,
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    RewardConfig,
    RunConfig,
    RuntimeConfig,
    RolloutConfig,
    ServingConfig,
    ServingSectionConfig,
    TrainingSectionConfig,
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
from flashrl.framework.flashrl import FlashRL
from flashrl.framework.serving import (
    HuggingFaceServingBackend,
    ServingBackend,
    VLLMMetalServingBackend,
    create_serving_backend,
)

__all__ = [
    "FlashRL",
    # Config
    "CommonConfig",
    "GrpoConfig",
    "TrainerConfig",
    "ModelConfig",
    "ServingConfig",
    "TrainingSectionConfig",
    "ServingSectionConfig",
    "RolloutConfig",
    "RewardConfig",
    "LoggingConfig",
    "MetricsConfig",
    "RuntimeConfig",
    "HookConfig",
    "RunConfig",
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
    "ServingBackend",
    "HuggingFaceServingBackend",
    "VLLMMetalServingBackend",
    "create_serving_backend",
    "TrainingBackend",
]
