"""Framework layer: Core RL training APIs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from flashrl.framework.config import (
    AdminConfig,
    CheckpointingConfig,
    ControllerConfig,
    FSDP2Config,
    GrpoConfig,
    HookConfig,
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    PushgatewayMetricsConfig,
    RewardConfig,
    RolloutConfig,
    RunConfig,
    ServingConfig,
    TensorBoardMetricsConfig,
    TrainingConfig,
)
from flashrl.framework.data_models import (
    AgentTrace,
    AgentTraceEvent,
    AssistantTurn,
    Conversation,
    LearnerBatch,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    ToolCall,
    ToolResult,
    TrainingBatch,
)
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    ComponentStatus,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    RewardBatchRequest,
    RewardBatchResponse,
    RolloutBatchRequest,
    RolloutBatchResponse,
    RpcError,
    RpcMessage,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    StageResultPayload,
    StatusResponse,
    WeightVersionRef,
)

if TYPE_CHECKING:
    from flashrl.framework.agent import (
        Agent,
        SubprocessToolRuntime,
        Tool,
    )
    from flashrl.framework.flashrl import FlashRL
    from flashrl.framework.serving import (
        HuggingFaceServingBackend,
        ServingBackend,
        VLLMServingBackend,
        create_serving_backend,
    )
    from flashrl.framework.training import (
        FSDP2TrainingBackend,
        HuggingFaceTrainingBackend,
        TrainingBackend,
        create_training_backend,
    )


_LAZY_EXPORTS = {
    "FlashRL": ("flashrl.framework.flashrl", "FlashRL"),
    "Agent": ("flashrl.framework.agent", "Agent"),
    "Tool": ("flashrl.framework.agent", "Tool"),
    "SubprocessToolRuntime": ("flashrl.framework.agent", "SubprocessToolRuntime"),
    "ServingBackend": ("flashrl.framework.serving", "ServingBackend"),
    "HuggingFaceServingBackend": ("flashrl.framework.serving", "HuggingFaceServingBackend"),
    "VLLMServingBackend": ("flashrl.framework.serving", "VLLMServingBackend"),
    "create_serving_backend": ("flashrl.framework.serving", "create_serving_backend"),
    "TrainingBackend": ("flashrl.framework.training", "TrainingBackend"),
    "HuggingFaceTrainingBackend": ("flashrl.framework.training", "HuggingFaceTrainingBackend"),
    "FSDP2TrainingBackend": ("flashrl.framework.training", "FSDP2TrainingBackend"),
    "create_training_backend": ("flashrl.framework.training", "create_training_backend"),
}

__all__ = [
    "FlashRL",
    # Config
    "AdminConfig",
    "CheckpointingConfig",
    "GrpoConfig",
    "ControllerConfig",
    "ModelConfig",
    "TrainingConfig",
    "ServingConfig",
    "FSDP2Config",
    "RolloutConfig",
    "RewardConfig",
    "LoggingConfig",
    "MetricsConfig",
    "TensorBoardMetricsConfig",
    "PushgatewayMetricsConfig",
    "HookConfig",
    "RunConfig",
    # Data models
    "Prompt",
    "Message",
    "Conversation",
    "AgentTraceEvent",
    "AgentTrace",
    "AssistantTurn",
    "LearnerBatch",
    "ToolCall",
    "ToolResult",
    "RolloutOutput",
    "RewardOutput",
    "TrainingBatch",
    # Distributed transport
    "RpcMessage",
    "RpcError",
    "WeightVersionRef",
    "ComponentStatus",
    "StageResultPayload",
    "RolloutBatchRequest",
    "RolloutBatchResponse",
    "RewardBatchRequest",
    "RewardBatchResponse",
    "OptimizeStepRequest",
    "OptimizeStepResponse",
    "ActivateWeightVersionRequest",
    "ActivateWeightVersionResponse",
    "SaveCheckpointRequest",
    "SaveCheckpointResponse",
    "LoadCheckpointRequest",
    "LoadCheckpointResponse",
    "StatusResponse",
    # Agent and tools
    "Agent",
    "Tool",
    "SubprocessToolRuntime",
    # Backends
    "ServingBackend",
    "HuggingFaceServingBackend",
    "VLLMServingBackend",
    "create_serving_backend",
    "TrainingBackend",
    "HuggingFaceTrainingBackend",
    "FSDP2TrainingBackend",
    "create_training_backend",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve heavyweight public exports on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive discovery tools."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))
