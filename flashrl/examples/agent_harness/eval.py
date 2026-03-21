"""Evaluate the reference agent harness example on its held-out tasks."""

from __future__ import annotations

import argparse
import json

from flashrl.examples.agent_harness.common import (
    DEFAULT_CONFIG_PATH,
    load_example_config,
    prepare_environment,
    resolve_dataset_limit,
)
from flashrl.examples.agent_harness.dataset import reward_fn
from flashrl.examples.agent_harness.evaluation import evaluate_model
from flashrl.examples.agent_harness.harness import build_agent_harness
from flashrl.framework import FlashRL


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
    eval_limit = resolve_dataset_limit(args.eval_limit, dataset_config, "eval_limit")
    flashrl: FlashRL | None = None
    try:
        rollout_agent = build_agent_harness(harness_config)
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=rollout_agent,
            reward_fn=reward_fn,
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
