"""Evaluate the reference agent harness example on its held-out tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from flashrl.examples.agent_harness.config import CodingHarnessConfig
from flashrl.examples.agent_harness.harness import (
    build_coding_agent,
    build_coding_eval_dataset,
    build_coding_reward_fn,
    evaluate_rollouts,
)
from flashrl.examples.agent_harness.train import DEFAULT_CONFIG_PATH, load_example_config, prepare_environment
from flashrl.framework import FlashRL

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


def evaluate_model(
    flashrl: FlashRL,
    rollout_agent,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    dataset = build_coding_eval_dataset(limit=limit)
    if flashrl._serving_backend is None:
        raise RuntimeError("FlashRL serving backend is not initialized.")
    rollouts = rollout_agent.run_batch(dataset, flashrl._serving_backend)
    reward_fn = build_coding_reward_fn()
    rewards = [reward_fn(rollout) for rollout in rollouts]
    return {
        "dataset_size": len(dataset),
        **evaluate_rollouts(rollouts, rewards),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the reference agent harness example.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the FlashRL config YAML.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Optional override for the eval task count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_environment(args.config)
    harness_config, dataset_config = load_example_config(args.config)
    eval_limit = args.eval_limit
    if eval_limit is None:
        raw_eval_limit = dataset_config.get("eval_limit")
        eval_limit = int(raw_eval_limit) if isinstance(raw_eval_limit, int) else None
    flashrl: FlashRL | None = None
    try:
        rollout_agent = build_coding_agent(harness_config)
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=rollout_agent,
            reward_fn=build_coding_reward_fn(),
        )
        payload = evaluate_model(
            flashrl,
            rollout_agent,
            limit=eval_limit,
        )
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
