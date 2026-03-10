"""Test that all abstractions can be imported and instantiated."""


def test_import_data_models():
    """Test importing data models."""
    from flashrl.framework.data_models import (
        Prompt,
        Message,
        Conversation,
        ToolCall,
        ToolResult,
        RolloutOutput,
        RewardOutput,
        TrainingBatch,
    )
    assert Prompt is not None
    assert Message is not None
    assert Conversation is not None
    assert ToolCall is not None
    assert ToolResult is not None
    assert RolloutOutput is not None
    assert RewardOutput is not None
    assert TrainingBatch is not None


def test_import_config():
    """Test importing config models."""
    from flashrl.framework.config import (
        TrainerConfig,
        ModelConfig,
        RolloutConfig,
        RewardConfig,
    )
    assert TrainerConfig is not None
    assert ModelConfig is not None
    assert RolloutConfig is not None
    assert RewardConfig is not None


def test_import_base_abstractions():
    """Test importing base abstractions."""
    from flashrl.framework.trainer.base import BaseTrainer
    from flashrl.framework.rollout.base import BaseRollout
    from flashrl.framework.reward.base import BaseReward
    from flashrl.framework.tools.base import BaseToolExecutor

    assert BaseTrainer is not None
    assert BaseRollout is not None
    assert BaseReward is not None
    assert BaseToolExecutor is not None


def test_import_model_wrappers():
    """Test importing model wrappers."""
    from flashrl.framework.models.actor import ActorModel
    from flashrl.framework.models.critic import CriticModel
    from flashrl.framework.models.reference import ReferenceModel
    from flashrl.framework.models.device import get_device

    assert ActorModel is not None
    assert CriticModel is not None
    assert ReferenceModel is not None
    assert get_device is not None


def test_data_model_creation():
    """Test creating data model instances."""
    from flashrl.framework.data_models import (
        Prompt,
        Message,
        Conversation,
        ToolCall,
        ToolResult,
    )

    # Test Prompt
    prompt = Prompt(text="Hello, world!")
    assert prompt.text == "Hello, world!"
    assert prompt.metadata == {}

    # Test Message
    message = Message(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"

    # Test Conversation
    conversation = Conversation(messages=[message])
    assert len(conversation.messages) == 1
    assert conversation.last_user_message() == message

    # Test ToolCall
    tool_call = ToolCall(name="search", arguments={"query": "test"})
    assert tool_call.name == "search"
    assert tool_call.arguments == {"query": "test"}

    # Test ToolResult
    tool_result = ToolResult(content="Success")
    assert tool_result.content == "Success"
    assert tool_result.error is False


def test_config_creation():
    """Test creating config instances."""
    from flashrl.framework.config import (
        TrainerConfig,
        ModelConfig,
        RolloutConfig,
        RewardConfig,
    )

    # Test TrainerConfig
    trainer_config = TrainerConfig(learning_rate=1e-5, batch_size=32)
    assert trainer_config.learning_rate == 1e-5
    assert trainer_config.batch_size == 32

    # Test ModelConfig
    model_config = ModelConfig(model_name="gpt2")
    assert model_config.model_name == "gpt2"

    # Test RolloutConfig
    rollout_config = RolloutConfig(max_new_tokens=512)
    assert rollout_config.max_new_tokens == 512

    # Test RewardConfig
    reward_config = RewardConfig(scale=1.0)
    assert reward_config.scale == 1.0


def test_device_detection():
    """Test device detection."""
    from flashrl.framework.models.device import get_device
    import torch

    device = get_device()
    assert isinstance(device, torch.device)


def test_framework_imports():
    """Test that all framework components can be imported from flashrl.framework."""
    from flashrl.framework import (
        TrainerConfig,
        ModelConfig,
        RolloutConfig,
        RewardConfig,
        Prompt,
        Message,
        Conversation,
        ToolCall,
        ToolResult,
        RolloutOutput,
        RewardOutput,
        TrainingBatch,
    )
