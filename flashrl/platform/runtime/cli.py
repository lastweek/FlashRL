"""Direct pod-command dispatcher for FlashRL platform workloads."""

from __future__ import annotations

import argparse

from flashrl.platform.runtime.platform_shim_controller import PlatformShimController
from flashrl.platform.runtime.platform_shim_learner import PlatformShimLearner
from flashrl.platform.runtime.platform_shim_reward import PlatformShimReward
from flashrl.platform.runtime.platform_shim_rollout import PlatformShimRollout
from flashrl.platform.runtime.platform_shim_serving import PlatformShimServing


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
        "controller": PlatformShimController,
        "rollout": PlatformShimRollout,
        "reward": PlatformShimReward,
        "learner": PlatformShimLearner,
        "serving": PlatformShimServing,
    }
    return int(dispatch[args.component_command]().run(host=args.host, port=args.port))
