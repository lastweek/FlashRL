"""Core data models used across all components."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Input prompt for the system."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: list["ToolCall"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool invocation request."""

    name: str
    arguments: dict[str, Any]
    tool_id: str | None = None


class ToolResult(BaseModel):
    """Output from tool execution."""

    content: str
    error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """An ordered list of messages (multi-turn)."""

    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def last_user_message(self) -> Message | None:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None


class WeightVersionInfo(BaseModel):
    """One activated or pending serving-weight version."""

    version_id: int
    source_training_step: int | None = None
    source_epoch: int | None = None
    activated_at: str | None = None
    model_source: str
    origin: Literal["startup", "sync", "resume"]


class RolloutOutput(BaseModel):
    """Output from rollout generation."""

    text: str
    log_prob: float
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_token_logprobs: list[float]
    conversation: Conversation
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardOutput(BaseModel):
    """Reward signal and metadata."""

    reward: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingBatch(BaseModel):
    """Batch of data for a training step."""

    prompts: list[Prompt]
    conversations: list[Conversation]
    rollouts: list[RolloutOutput]
    rewards: list[RewardOutput]
    group_size: int = 1
    prompt_count: int = 0
    prompt_indices: list[int] = Field(default_factory=list)
    candidate_indices: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.prompts)


class LearnerBatch(BaseModel):
    """CPU-resident optimization batch handed from controller to training."""

    prompt_token_ids: list[list[int]]
    response_token_ids: list[list[int]]
    response_token_logprobs: list[list[float]]
    advantages: list[float]
    group_size: int = 1
    prompt_count: int = 0
    prompt_indices: list[int] = Field(default_factory=list)
    candidate_indices: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of rollout samples in the learner batch."""
        return len(self.prompt_token_ids)
