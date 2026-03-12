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


def reasoning_rollout_fn(
    prompts: list[Prompt],
    actor,
    group_size: int,
) -> list[list[RolloutOutput]]:
    """Generate grouped reasoning rollouts with the actor model."""
    grouped_samples = actor.generate_grouped([prompt.text for prompt in prompts], group_size)
    grouped_rollouts: list[list[RolloutOutput]] = []
    for prompt, samples in zip(prompts, grouped_samples, strict=True):
        prompt_rollouts: list[RolloutOutput] = []
        for sample in samples:
            prompt_rollouts.append(
                RolloutOutput(
                    text=sample.text,
                    log_prob=sample.log_prob,
                    prompt_token_ids=sample.prompt_token_ids,
                    response_token_ids=sample.response_token_ids,
                    response_token_logprobs=sample.response_token_logprobs,
                    conversation=Conversation(
                        messages=[
                            Message(role="user", content=prompt.text),
                            Message(role="assistant", content=sample.text),
                        ]
                    ),
                )
            )
        grouped_rollouts.append(prompt_rollouts)
    return grouped_rollouts


def reasoning_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Reward correct reasoning-tag structure and detailed reasoning text."""
    text = rollout.text
    reward = 0.0

    has_open_tag = "<reason>" in text
    has_close_tag = "</reason>" in text
    has_reason = has_open_tag and has_close_tag

    if has_reason:
        match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if match:
            reason_content = match.group(1).strip()
            reward = min(len(reason_content) / 50.0, 10.0)
        else:
            reward = 0.5

    return RewardOutput(
        reward=reward,
        metadata={
            "has_reason": has_reason,
            "has_open_tag": has_open_tag,
            "has_close_tag": has_close_tag,
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
