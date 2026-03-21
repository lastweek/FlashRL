"""Unit tests for rollout generators and reward wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flashrl.framework.agent import Agent
from flashrl.framework.config import RewardConfig, RolloutConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.agent import AgentRolloutGenerator
from flashrl.framework.rollout.base import build_rollout_generator
from flashrl.framework.rollout.function import FunctionRolloutGenerator
from tests.conftest import TinyServingBackend, make_rollout_fn

pytestmark = pytest.mark.unit


def build_test_agent() -> Agent:
    """Build one minimal whitebox agent for rollout-generator tests."""

    def run(agent: Agent) -> None:
        sample = agent.generate(agent.prompt.text)
        agent.record_generation(sample)
        agent.finish(sample.text)

    return Agent(run_fn=run, max_steps=1)


def test_function_rollout_generator_generate_applies_generation_defaults() -> None:
    """Rollout wrapper should push generation defaults onto the serving backend."""
    serving_backend = TinyServingBackend()
    captured_defaults: list[dict[str, object]] = []

    def rollout_fn(
        prompts: list[Prompt],
        wrapped_backend: TinyServingBackend,
    ) -> list[RolloutOutput]:
        captured_defaults.append(dict(wrapped_backend.generation_defaults))
        return make_rollout_fn(response_suffix="rollout", repeat=1)(prompts, wrapped_backend)

    rollout = FunctionRolloutGenerator(
        rollout_fn=rollout_fn,
        serving_backend=serving_backend,
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


def test_function_rollout_generator_generate_grouped_is_prompt_major_and_validates_count() -> None:
    """Grouped rollout should preserve prompt-major ordering and reject bad output counts."""
    serving_backend = TinyServingBackend()
    rollout = FunctionRolloutGenerator(
        rollout_fn=make_rollout_fn(response_suffix="grouped", repeat=1),
        serving_backend=serving_backend,
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

    invalid = FunctionRolloutGenerator(
        rollout_fn=lambda prompts, wrapped_backend: make_rollout_fn(
            response_suffix="invalid",
            repeat=1,
        )(prompts[:-1], wrapped_backend),
        serving_backend=serving_backend,
        config=RolloutConfig(),
    )
    with pytest.raises(ValueError, match="one output per input prompt"):
        invalid.generate_grouped(prompts, group_size=2)


def test_function_rollout_generator_generate_grouped_repeats_unique_prompt_batches_per_candidate() -> None:
    """Grouped rollout should repeat the same unique-prompt batch once per candidate pass."""
    serving_backend = TinyServingBackend()
    observed_calls: list[list[str]] = []

    def rollout_fn(
        prompts: list[Prompt],
        wrapped_backend: TinyServingBackend,
    ) -> list[RolloutOutput]:
        del wrapped_backend
        observed_calls.append([prompt.text for prompt in prompts])
        return make_rollout_fn(response_suffix="grouped", repeat=1)(prompts, serving_backend)

    rollout = FunctionRolloutGenerator(
        rollout_fn=rollout_fn,
        serving_backend=serving_backend,
        config=RolloutConfig(),
    )

    rollout.generate_grouped(
        [Prompt(text="prompt 0"), Prompt(text="prompt 1")],
        group_size=3,
    )

    assert observed_calls == [
        ["prompt 0", "prompt 1"],
        ["prompt 0", "prompt 1"],
        ["prompt 0", "prompt 1"],
    ]


def test_function_rollout_generator_generate_conversation_uses_first_rollout_and_handles_empty() -> None:
    """Conversation generation should reuse the first rollout conversation or return an empty one."""
    serving_backend = TinyServingBackend()
    single = FunctionRolloutGenerator(
        rollout_fn=make_rollout_fn(response_suffix="conv", repeat=1),
        serving_backend=serving_backend,
        config=RolloutConfig(),
    )
    empty = FunctionRolloutGenerator(
        rollout_fn=lambda prompts, wrapped_backend: [],
        serving_backend=serving_backend,
        config=RolloutConfig(),
    )

    conversation = single.generate_conversation(Prompt(text="hello"))
    empty_conversation = empty.generate_conversation(Prompt(text="hello"))

    assert conversation.messages[-1].content == "conv detail hello::0"
    assert empty_conversation.messages == []


def test_agent_rollout_generator_generate_grouped_is_prompt_major_and_validates_outputs() -> None:
    """The direct Agent generator should preserve prompt-major grouping and validation."""
    serving_backend = TinyServingBackend()
    rollout = AgentRolloutGenerator(
        agent=build_test_agent(),
        serving_backend=serving_backend,
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
        "generated::prompt 0::0",
        "generated::prompt 0::1",
        "generated::prompt 1::0",
        "generated::prompt 1::1",
    ]
    assert prompt_indices == [0, 0, 1, 1]
    assert candidate_indices == [0, 1, 0, 1]
    assert all(rollout.metadata["weight_version"]["version_id"] == 0 for rollout in rollouts)


def test_build_rollout_generator_selects_function_or_agent_and_rejects_invalid() -> None:
    """FlashRL's rollout normalization helper should pick the correct internal adapter."""
    serving_backend = TinyServingBackend()

    function_rollout = build_rollout_generator(
        rollout_fn=make_rollout_fn(response_suffix="factory", repeat=1),
        serving_backend=serving_backend,
        config=RolloutConfig(),
    )
    agent_rollout = build_rollout_generator(
        rollout_fn=build_test_agent(),
        serving_backend=serving_backend,
        config=RolloutConfig(),
    )

    assert isinstance(function_rollout, FunctionRolloutGenerator)
    assert isinstance(agent_rollout, AgentRolloutGenerator)

    with pytest.raises(TypeError, match="callable or flashrl.framework.agent.Agent"):
        build_rollout_generator(
            rollout_fn=object(),
            serving_backend=serving_backend,
            config=RolloutConfig(),
        )


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
