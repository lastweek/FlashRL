"""Unit tests for the reasoning example reward design."""

from __future__ import annotations

import pytest

from examples.reasoning.train import (
    REASONING_PROMPTS,
    _extract_predicted_answer,
    _parse_expected_answer,
    reasoning_reward_fn,
)
from flashrl.framework.data_models import Conversation, Message, RolloutOutput

pytestmark = pytest.mark.unit


def make_rollout(prompt_text: str, response_text: str) -> RolloutOutput:
    """Build a minimal rollout for reasoning reward tests."""
    return RolloutOutput(
        text=response_text,
        log_prob=-0.1,
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[4, 5, 6],
        response_token_logprobs=[-0.1, -0.1, -0.1],
        conversation=Conversation(
            messages=[
                Message(role="user", content=prompt_text),
                Message(role="assistant", content=response_text),
            ]
        ),
    )


@pytest.mark.parametrize(
    ("prompt_text", "expected_answer"),
    [
        (REASONING_PROMPTS[0], 42),
        (REASONING_PROMPTS[1], 35),
        (REASONING_PROMPTS[2], 13),
        (REASONING_PROMPTS[3], 54),
        (
            "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I divide 24 by 3, what do I get?",
            8,
        ),
    ],
)
def test_reasoning_prompt_parser_supports_all_committed_patterns(
    prompt_text: str,
    expected_answer: int,
) -> None:
    """The example reward parser should understand every committed dataset prompt."""
    assert _parse_expected_answer(prompt_text) == expected_answer


def test_answer_extractor_prefers_final_line_then_falls_back_to_last_number() -> None:
    """Final-line answers should win, with a deterministic whole-response fallback."""
    assert _extract_predicted_answer("Work...\nFinal answer: 42") == 42
    assert _extract_predicted_answer("We saw 3 and then 7 before stopping") == 7


def test_reasoning_reward_gives_partial_credit_for_single_tag_presence() -> None:
    """Malformed tag structure should still get partial structure reward."""
    open_only = reasoning_reward_fn(make_rollout(REASONING_PROMPTS[0], "<reason> working"))
    close_only = reasoning_reward_fn(make_rollout(REASONING_PROMPTS[0], "working </reason>"))

    assert open_only.reward == pytest.approx(0.20)
    assert close_only.reward == pytest.approx(0.20)
    assert open_only.metadata["structure_score"] == pytest.approx(0.20)
    assert close_only.metadata["structure_score"] == pytest.approx(0.20)


def test_reasoning_reward_is_dense_and_structure_dominant() -> None:
    """Structured outputs should score above unstructured ones even when wrong."""
    prompt_text = REASONING_PROMPTS[0]
    structured_correct = reasoning_reward_fn(
        make_rollout(
            prompt_text,
            "<reason>Add 15 and 27 carefully to get 42 in the final line.</reason>\n42",
        )
    )
    structured_wrong = reasoning_reward_fn(
        make_rollout(
            prompt_text,
            "<reason>Add 15 and 27 carefully to get 41 in the final line.</reason>\n41",
        )
    )
    correct_without_tags = reasoning_reward_fn(make_rollout(prompt_text, "42"))

    assert structured_correct.reward == pytest.approx(1.0)
    assert structured_wrong.reward > 0.0
    assert correct_without_tags.reward > 0.0
    assert structured_correct.reward > structured_wrong.reward
    assert structured_correct.reward > correct_without_tags.reward
    assert structured_wrong.reward > correct_without_tags.reward


def test_reasoning_reward_metadata_explains_structure_and_correctness() -> None:
    """Reward metadata should expose the parsed answers and score breakdown."""
    reward = reasoning_reward_fn(
        make_rollout(
            "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I divide 24 by 3, what do I get?",
            "<reason>24 divided by 3 is 8, so the answer is 8.</reason>\n8",
        )
    )

    assert reward.metadata["expected_answer"] == 8
    assert reward.metadata["predicted_answer"] == 8
    assert reward.metadata["has_open_tag"] is True
    assert reward.metadata["has_close_tag"] is True
    assert reward.metadata["has_reason_content"] is True
    assert reward.metadata["answer_parseable"] is True
    assert reward.metadata["is_correct"] is True
    assert reward.metadata["structure_score"] > reward.metadata["correctness_score"]
