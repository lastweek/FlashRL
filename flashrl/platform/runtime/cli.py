"""Direct pod-command dispatcher for FlashRL platform workloads."""

from __future__ import annotations

import argparse

from flashrl.platform.runtime.controller import run_controller_pod
from flashrl.platform.runtime.learner import run_learner_pod
from flashrl.platform.runtime.reward import run_reward_pod
from flashrl.platform.runtime.rollout import run_rollout_pod
from flashrl.platform.runtime.serving import run_serving_pod


def build_component_argument_parser() -> argparse.ArgumentParser:
    """Build the direct pod-runtime CLI parser."""
    parser = argparse.ArgumentParser(prog="flashrl", description="FlashRL platform pod runtime")
    subparsers = parser.add_subparsers(dest="component_command", required=True)
    for component in ("controller", "rollout", "reward", "learner", "serving"):
        subparser = subparsers.add_parser(component, help=f"Run the {component} platform pod")
        subparser.add_argument("--host", default="0.0.0.0")
        subparser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the pod command, then dispatch to one explicit role module."""
    parser = build_component_argument_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "controller": run_controller_pod,
        "rollout": run_rollout_pod,
        "reward": run_reward_pod,
        "learner": run_learner_pod,
        "serving": run_serving_pod,
    }
    return int(dispatch[args.component_command](host=args.host, port=args.port))
