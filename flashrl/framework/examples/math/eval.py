"""Held-out evaluation for the strict math example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

import train as reasoning_math_example

from flashrl.framework import FlashRL
from flashrl.framework.data_models import Prompt


def evaluate_model(
    flashrl: FlashRL,
    *,
    dataset: list[Prompt],
    batch_size: int,
    rollout_fn=None,
    training_mode: str = "reasoning",
) -> dict[str, float | int]:
    """Run held-out evaluation against the current serving backend."""
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
    effective_rollout_fn = rollout_fn or reasoning_math_example.reasoning_rollout_fn

    for start in range(0, len(dataset), batch_size):
        prompts = dataset[start:start + batch_size]
        rollouts = effective_rollout_fn(prompts, flashrl._serving_backend)
        rewards = [
            reasoning_math_example.math_reward_fn(rollout, training_mode=training_mode)
            for rollout in rollouts
        ]
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
    parser = argparse.ArgumentParser(
        description="Evaluate the FlashRL math example."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_vllm.yaml")),
        help="Path to the FlashRL runtime/training profile.",
    )
    parser.add_argument(
        "--dataset",
        choices=reasoning_math_example.SUPPORTED_MATH_DATASETS,
        default=reasoning_math_example.DEFAULT_MATH_DATASET,
        help="Math dataset to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to load before evaluation.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional number of held-out questions to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=reasoning_math_example.DEFAULT_REASONING_EVAL_BATCH_SIZE,
        help="Optional number of prompts to evaluate per generate_batch call.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["math", "reasoning"],
        default="math",
        help="Training mode for evaluation: 'math' for pure math, 'reasoning' for reasoning + math",
    )
    parser.add_argument(
        "--rollout-mode",
        choices=reasoning_math_example.SUPPORTED_MATH_ROLLOUT_MODES,
        default=reasoning_math_example.DEFAULT_MATH_ROLLOUT_MODE,
        help="Rollout implementation: 'blackbox' uses the example rollout hook, "
        "'whitebox' uses FlashRL's traced agent building blocks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run held-out evaluation and print compact JSON metrics."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    reasoning_math_example.prepare_reasoning_environment(args.config)

    flashrl: FlashRL | None = None
    try:
        dataset = reasoning_math_example.build_math_eval_dataset(
            dataset=args.dataset,
            limit=args.eval_limit,
            training_mode=args.training_mode,  # NEW
        )
        batch_size = args.batch_size
        checkpoint = args.checkpoint
        if checkpoint is None:
            default_checkpoint = Path(reasoning_math_example.DEFAULT_REASONING_CHECKPOINT_PATH)
            if default_checkpoint.exists():
                checkpoint = str(default_checkpoint)
        system_prompt = reasoning_math_example.build_math_system_prompt(args.training_mode)
        rollout_impl = reasoning_math_example.build_math_rollout(
            rollout_mode=args.rollout_mode,
            training_mode=args.training_mode,
            system_prompt=system_prompt,
        )

        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=rollout_impl,
            reward_fn=reasoning_math_example.math_reward_fn,
        )
        if checkpoint:
            flashrl.load_checkpoint(checkpoint)
        metrics = evaluate_model(
            flashrl,
            dataset=dataset,
            batch_size=batch_size,
            rollout_fn=rollout_impl,
            training_mode=args.training_mode,
        )
        print(json.dumps(metrics, ensure_ascii=True, sort_keys=True))
    except Exception as exc:
        print(f"FlashRL math evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
