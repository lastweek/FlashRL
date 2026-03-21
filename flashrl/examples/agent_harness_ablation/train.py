"""Run a local-first ablation study over agent harness variants."""

from __future__ import annotations

import argparse
import json

from flashrl.examples.agent_harness.common import prepare_environment
from flashrl.examples.agent_harness_ablation.study import (
    DEFAULT_CONFIG_PATH,
    run_study,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the agent harness ablation study.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the study config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_environment(args.config)
    manifest_path = run_study(args.config)
    print(json.dumps({"manifest_path": str(manifest_path.resolve())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
