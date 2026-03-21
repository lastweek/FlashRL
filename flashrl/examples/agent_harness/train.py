"""Train the reference agent harness example."""

from __future__ import annotations

import argparse

from flashrl.examples.agent_harness.common import (
    DEFAULT_CONFIG_PATH,
    load_example_config,
    prepare_environment,
    resolve_dataset_limit,
)
from flashrl.examples.agent_harness.dataset import build_train_dataset, reward_fn
from flashrl.examples.agent_harness.harness import build_agent_harness
from flashrl.framework import FlashRL


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reference agent harness example.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the FlashRL config YAML.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional override for the training task count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_environment(args.config)
    harness_config, dataset_config = load_example_config(args.config)
    dataset = build_train_dataset(
        limit=resolve_dataset_limit(args.train_limit, dataset_config, "train_limit")
    )
    flashrl: FlashRL | None = None
    try:
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=build_agent_harness(harness_config),
            reward_fn=reward_fn,
        )
        flashrl.train(dataset)
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
