"""Direct unit tests for core data models."""

from __future__ import annotations

import pytest

from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput, ToolCall, TrainingBatch

pytestmark = pytest.mark.unit


def test_conversation_add_message_and_last_user_message() -> None:
    """Conversation helpers should append messages and find the most recent user message."""
    conversation = Conversation(messages=[Message(role="system", content="be brief")])
    conversation.add_message(Message(role="user", content="first"))
    conversation.add_message(Message(role="assistant", content="reply"))
    conversation.add_message(Message(role="user", content="second"))

    assert [message.content for message in conversation.messages] == [
        "be brief",
        "first",
        "reply",
        "second",
    ]
    assert conversation.last_user_message() == Message(role="user", content="second")


def test_default_containers_are_not_shared_across_instances() -> None:
    """Mutable defaults should be independent across model instances."""
    first_prompt = Prompt(text="hello")
    second_prompt = Prompt(text="world")
    first_message = Message(role="assistant", content="one")
    second_message = Message(role="assistant", content="two")

    first_prompt.metadata["source"] = "first"
    first_message.tool_calls.append(ToolCall(name="search", arguments={"q": "x"}))

    assert second_prompt.metadata == {}
    assert second_message.tool_calls == []


def test_training_batch_len_and_grouping_fields_are_preserved() -> None:
    """TrainingBatch should expose sample count through __len__ and keep grouping metadata intact."""
    prompts = [Prompt(text="p0"), Prompt(text="p0"), Prompt(text="p1"), Prompt(text="p1")]
    rollouts = [
        RolloutOutput(
            text=f"r{index}",
            log_prob=0.0,
            prompt_token_ids=[1],
            response_token_ids=[2, 3],
            response_token_logprobs=[-0.1, -0.2],
            conversation=Conversation(messages=[]),
        )
        for index in range(4)
    ]
    rewards = [RewardOutput(reward=float(index)) for index in range(4)]
    batch = TrainingBatch(
        prompts=prompts,
        conversations=[rollout.conversation for rollout in rollouts],
        rollouts=rollouts,
        rewards=rewards,
        group_size=2,
        prompt_count=2,
        prompt_indices=[0, 0, 1, 1],
        candidate_indices=[0, 1, 0, 1],
    )

    assert len(batch) == 4
    assert batch.group_size == 2
    assert batch.prompt_count == 2
    assert batch.prompt_indices == [0, 0, 1, 1]
    assert batch.candidate_indices == [0, 1, 0, 1]
