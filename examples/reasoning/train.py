"""DeepSeek-R1 style reasoning training example."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flashrl.framework import FlashRL
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
)


REASONING_PROMPTS = [
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 15 + 27?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I have 3 apples and get 5 more, how many do I have?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 8 × 7?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 100 - 37?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I divide 24 by 3, what do I get?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 12 + 15 + 8?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I have 20 items and give away 7, how many remain?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 9 × 6?",
]

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
REASON_BLOCK_PATTERN = re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL)


def reasoning_rollout_fn(
    prompts: list[Prompt],
    actor,
) -> list[RolloutOutput]:
    """Generate one reasoning rollout per prompt with the actor model."""
    samples = actor.generate_batch([prompt.text for prompt in prompts])
    rollouts: list[RolloutOutput] = []
    for prompt, sample in zip(prompts, samples, strict=True):
        rollouts.append(
            RolloutOutput(
                text=sample.text,
                log_prob=sample.log_prob,
                prompt_token_ids=sample.prompt_token_ids,
                response_token_ids=sample.response_token_ids,
                response_token_logprobs=sample.response_token_logprobs,
                metadata=dict(sample.metadata),
                conversation=Conversation(
                    messages=[
                        Message(role="user", content=prompt.text),
                        Message(role="assistant", content=sample.text),
                    ]
                ),
            )
        )
    return rollouts


def _normalize_number(value: float) -> int | float:
    """Return integral values as ints so metadata stays readable."""
    if float(value).is_integer():
        return int(value)
    return float(value)


def _parse_number(text: str) -> int | float:
    """Parse one numeric literal into an int or float."""
    return _normalize_number(float(text))


def _extract_question_text(prompt_text: str) -> str:
    """Extract the arithmetic question from the full user prompt."""
    _, separator, question = prompt_text.rpartition("Question:")
    if separator:
        return question.strip()
    return prompt_text.strip()


def _parse_expected_answer(prompt_text: str) -> int | float | None:
    """Parse the expected arithmetic answer from the committed prompt patterns."""
    question = _extract_question_text(prompt_text)
    matchers = [
        (
            re.compile(r"What is (\d+) \+ (\d+) \+ (\d+)\?$"),
            lambda match: int(match.group(1)) + int(match.group(2)) + int(match.group(3)),
        ),
        (
            re.compile(r"What is (\d+) \+ (\d+)\?$"),
            lambda match: int(match.group(1)) + int(match.group(2)),
        ),
        (
            re.compile(r"What is (\d+) (?:×|x|\*) (\d+)\?$"),
            lambda match: int(match.group(1)) * int(match.group(2)),
        ),
        (
            re.compile(r"What is (\d+) - (\d+)\?$"),
            lambda match: int(match.group(1)) - int(match.group(2)),
        ),
        (
            re.compile(r"If I divide (\d+) by (\d+), what do I get\?$"),
            lambda match: _normalize_number(int(match.group(1)) / int(match.group(2))),
        ),
        (
            re.compile(r"If I have (\d+) apples and get (\d+) more, how many do I have\?$"),
            lambda match: int(match.group(1)) + int(match.group(2)),
        ),
        (
            re.compile(r"If I have (\d+) items and give away (\d+), how many remain\?$"),
            lambda match: int(match.group(1)) - int(match.group(2)),
        ),
    ]

    for pattern, resolver in matchers:
        match = pattern.fullmatch(question)
        if match is not None:
            return resolver(match)
    return None


def _extract_predicted_answer(response_text: str) -> int | float | None:
    """Extract the model's predicted final numeric answer deterministically."""
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    candidates: list[str] = []
    if lines:
        candidates.append(lines[-1])

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(lines or [response_text]))
        if sentence.strip()
    ]
    if sentences:
        candidates.append(sentences[-1])

    for candidate in candidates:
        numbers = NUMBER_PATTERN.findall(candidate)
        if numbers:
            return _parse_number(numbers[-1])

    numbers = NUMBER_PATTERN.findall(response_text)
    if numbers:
        return _parse_number(numbers[-1])
    return None


def _prompt_text_from_rollout(rollout: RolloutOutput) -> str:
    """Recover the original user prompt text from the rollout conversation."""
    last_user_message = rollout.conversation.last_user_message()
    if last_user_message is not None:
        return last_user_message.content
    for message in rollout.conversation.messages:
        if message.role == "user":
            return message.content
    return ""


def _numbers_equal(left: int | float | None, right: int | float | None) -> bool:
    """Compare numeric answers with a small float tolerance."""
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) < 1e-9


def reasoning_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Reward reasoning structure heavily and arithmetic correctness second."""
    text = rollout.text
    prompt_text = _prompt_text_from_rollout(rollout)
    expected_answer = _parse_expected_answer(prompt_text)
    predicted_answer = _extract_predicted_answer(text)

    has_open_tag = "<reason>" in text.lower()
    has_close_tag = "</reason>" in text.lower()
    reason_match = REASON_BLOCK_PATTERN.search(text)
    reason_content = reason_match.group(1).strip() if reason_match is not None else ""
    has_reason_content = bool(reason_content)

    structure_score = 0.0
    if has_open_tag:
        structure_score += 0.20
    if has_close_tag:
        structure_score += 0.20
    if has_reason_content:
        structure_score += 0.20
    if has_reason_content and len(reason_content.split()) >= 8:
        structure_score += 0.10

    correctness_score = 0.0
    if predicted_answer is not None:
        correctness_score += 0.05
    if _numbers_equal(predicted_answer, expected_answer):
        correctness_score += 0.25

    reward = min(structure_score + correctness_score, 1.0)
    is_correct = _numbers_equal(predicted_answer, expected_answer)

    return RewardOutput(
        reward=reward,
        metadata={
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "has_open_tag": has_open_tag,
            "has_close_tag": has_close_tag,
            "has_reason_content": has_reason_content,
            "reason_content_length": len(reason_content),
            "structure_score": structure_score,
            "correctness_score": correctness_score,
            "answer_parseable": predicted_answer is not None,
            "is_correct": is_correct,
        },
    )


def build_dataset() -> list[Prompt]:
    """Build the fixed reasoning dataset used by this example."""
    return [Prompt(text=prompt) for prompt in REASONING_PROMPTS]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the example CLI parser."""
    parser = argparse.ArgumentParser(description="Run the FlashRL reasoning example.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the example YAML config file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the reasoning example from YAML."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        flashrl = FlashRL.from_yaml(args.config)
        flashrl.train()
        flashrl.save_checkpoint("/tmp/flashrl_reasoning_checkpoint.pt")
    except Exception as exc:
        print(f"\nFlashRL reasoning example failed: {exc}", file=sys.stderr)
        print(
            "\nNote: This example downloads 'Qwen/Qwen2.5-0.5B-Instruct'.",
            file=sys.stderr,
        )
        print(
            "If you're offline or have network issues, use a local model in the YAML config.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
