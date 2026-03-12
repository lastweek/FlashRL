"""Unit tests for user-defined rollout and reward wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flashrl.framework.config import RewardConfig, RolloutConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from tests.conftest import TinyActor, make_rollout_fn

pytestmark = pytest.mark.unit


def test_user_defined_rollout_generate_applies_generation_defaults() -> None:
    """Rollout wrapper should push generation defaults onto the actor before generation."""
    actor = TinyActor()
    captured_defaults: list[dict[str, object]] = []

    def rollout_fn(
        prompts: list[Prompt],
        wrapped_actor: TinyActor,
        group_size: int,
    ) -> list[list[RolloutOutput]]:
        captured_defaults.append(dict(wrapped_actor.generation_defaults))
        return make_rollout_fn(response_suffix="rollout", repeat=1)(
            prompts,
            wrapped_actor,
            group_size,
        )

    rollout = UserDefinedRollout(
        rollout_fn=rollout_fn,
        actor=actor,
        config=RolloutConfig(
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.8,
            top_k=12,
            do_sample=True,
        ),
    )

    outputs = rollout.generate([Prompt(text="hello")])

    assert [output.text for output in outputs] == ["rollout detail hello::0"]
    assert captured_defaults == [
        {
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 12,
            "do_sample": True,
        }
    ]


def test_user_defined_rollout_generate_grouped_is_prompt_major_and_validates_count() -> None:
    """Grouped rollout should preserve prompt-major ordering and reject bad output counts."""
    actor = TinyActor()
    rollout = UserDefinedRollout(
        rollout_fn=make_rollout_fn(response_suffix="grouped", repeat=1),
        actor=actor,
        config=RolloutConfig(),
    )
    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]

    expanded_prompts, rollouts, prompt_indices, candidate_indices = rollout.generate_grouped(
        prompts,
        group_size=2,
    )

    assert [prompt.text for prompt in expanded_prompts] == [
        "prompt 0",
        "prompt 0",
        "prompt 1",
        "prompt 1",
    ]
    assert [rollout.text for rollout in rollouts] == [
        "grouped detail prompt 0::0",
        "grouped detail prompt 0::1",
        "grouped detail prompt 1::0",
        "grouped detail prompt 1::1",
    ]
    assert prompt_indices == [0, 0, 1, 1]
    assert candidate_indices == [0, 1, 0, 1]

    invalid = UserDefinedRollout(
        rollout_fn=lambda prompts, wrapped_actor, group_size: make_rollout_fn(
            response_suffix="invalid",
            repeat=1,
        )(prompts[:-1], wrapped_actor, group_size),
        actor=actor,
        config=RolloutConfig(),
    )
    with pytest.raises(ValueError, match="one candidate list per input prompt"):
        invalid.generate_grouped(prompts, group_size=2)


def test_user_defined_rollout_generate_grouped_calls_hook_once_with_unique_prompts() -> None:
    """Grouped rollout should pass unique prompts to the hook instead of duplicating them in Python."""
    actor = TinyActor()
    observed_calls: list[tuple[list[str], int]] = []

    def rollout_fn(
        prompts: list[Prompt],
        wrapped_actor: TinyActor,
        group_size: int,
    ) -> list[list[RolloutOutput]]:
        del wrapped_actor
        observed_calls.append(([prompt.text for prompt in prompts], group_size))
        return make_rollout_fn(response_suffix="grouped", repeat=1)(prompts, actor, group_size)

    rollout = UserDefinedRollout(
        rollout_fn=rollout_fn,
        actor=actor,
        config=RolloutConfig(),
    )

    rollout.generate_grouped(
        [Prompt(text="prompt 0"), Prompt(text="prompt 1")],
        group_size=3,
    )

    assert observed_calls == [(["prompt 0", "prompt 1"], 3)]


def test_user_defined_rollout_generate_conversation_uses_first_rollout_and_handles_empty() -> None:
    """Conversation generation should reuse the first rollout conversation or return an empty one."""
    actor = TinyActor()
    single = UserDefinedRollout(
        rollout_fn=make_rollout_fn(response_suffix="conv", repeat=1),
        actor=actor,
        config=RolloutConfig(),
    )
    empty = UserDefinedRollout(
        rollout_fn=lambda prompts, wrapped_actor, group_size: [[] for _ in prompts],
        actor=actor,
        config=RolloutConfig(),
    )

    conversation = single.generate_conversation(Prompt(text="hello"))
    empty_conversation = empty.generate_conversation(Prompt(text="hello"))

    assert conversation.messages[-1].content == "conv detail hello::0"
    assert empty_conversation.messages == []


def test_user_defined_reward_compute_batch_preserves_order() -> None:
    """compute_batch should preserve rollout order when mapping rewards."""
    reward = UserDefinedReward(
        reward_fn=lambda rollout: RewardOutput(reward=len(rollout.text)),
        config=RewardConfig(),
    )
    rollouts = [
        RolloutOutput(
            text="a",
            log_prob=0.0,
            prompt_token_ids=[1],
            response_token_ids=[2],
            response_token_logprobs=[-0.1],
            conversation=Conversation(messages=[]),
        ),
        RolloutOutput(
            text="abcd",
            log_prob=0.0,
            prompt_token_ids=[1],
            response_token_ids=[2, 3],
            response_token_logprobs=[-0.1, -0.2],
            conversation=Conversation(messages=[]),
        ),
    ]

    outputs = reward.compute_batch(rollouts)

    assert [output.reward for output in outputs] == [1, 4]


def test_user_defined_reward_compute_conversation_reward_builds_fallback_text() -> None:
    """Conversation reward should convert messages into the documented fallback rollout text."""
    captured_texts: list[str] = []

    def reward_fn(rollout: RolloutOutput) -> RewardOutput:
        captured_texts.append(rollout.text)
        return RewardOutput(reward=1.0)

    reward = UserDefinedReward(reward_fn=reward_fn, config=RewardConfig())
    conversation = Conversation(
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="world"),
        ]
    )

    result = reward.compute_conversation_reward(conversation)

    assert result.reward == pytest.approx(1.0)
    assert captured_texts == ["user: hello\nassistant: world"]
