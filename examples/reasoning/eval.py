"""Held-out evaluation for the strict reasoning example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flashrl.framework import FlashRL

from examples.reasoning.train import (
    _prepare_example_environment,
    build_eval_dataset,
    reasoning_reward_fn,
    reasoning_rollout_fn,
)


def evaluate_model(
    flashrl: FlashRL,
    *,
    limit: int | None = None,
    batch_size: int = 8,
) -> dict[str, float | int]:
    """Run held-out evaluation against the current serving backend."""
    dataset = build_eval_dataset(limit=limit)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if flashrl._serving_backend is None:  # pragma: no cover - defensive in live usage
        raise RuntimeError("FlashRL serving backend is not initialized.")

    flashrl._serving_backend.set_generation_defaults(
        max_new_tokens=flashrl.rollout_config.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        do_sample=False,
    )

    total_reward = 0.0
    accuracy_passes = 0
    format_passes = 0
    truncations = 0
    sample_count = 0

    for start in range(0, len(dataset), batch_size):
        prompts = dataset[start:start + batch_size]
        rollouts = reasoning_rollout_fn(prompts, flashrl._serving_backend)
        rewards = [reasoning_reward_fn(rollout) for rollout in rollouts]
        for reward in rewards:
            sample_count += 1
            total_reward += float(reward.reward)
            accuracy_passes += int(bool(reward.metadata.get("accuracy_pass", False)))
            format_passes += int(bool(reward.metadata.get("format_pass", False)))
            truncations += int(bool(reward.metadata.get("truncated", False)))

    if sample_count == 0:
        return {
            "sample_count": 0,
            "reward_mean": 0.0,
            "exact_match": 0.0,
            "format_pass_rate": 0.0,
            "truncation_rate": 0.0,
        }

    return {
        "sample_count": sample_count,
        "reward_mean": total_reward / sample_count,
        "exact_match": accuracy_passes / sample_count,
        "format_pass_rate": format_passes / sample_count,
        "truncation_rate": truncations / sample_count,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for held-out evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the FlashRL reasoning example.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_vllm.yaml")),
        help="Path to the example YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to load before evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of held-out samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of held-out prompts to evaluate per generate_batch call.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run held-out evaluation and print compact JSON metrics."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    _prepare_example_environment(args.config)

    flashrl: FlashRL | None = None
    try:
        flashrl = FlashRL.from_yaml(args.config)
        if args.checkpoint:
            flashrl.load_checkpoint(args.checkpoint)
        metrics = evaluate_model(
            flashrl,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, ensure_ascii=True, sort_keys=True))
    except Exception as exc:
        print(f"FlashRL reasoning evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
