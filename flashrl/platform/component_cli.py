"""CLI entrypoints for platform component pods."""

from __future__ import annotations

import argparse

import uvicorn

from flashrl.platform.controller import create_controller_app
from flashrl.platform.runtime import (
    create_learner_component_app,
    create_reward_component_app,
    create_rollout_component_app,
    create_serving_component_app,
    load_job_config,
)


def build_component_argument_parser() -> argparse.ArgumentParser:
    """Build the component runtime CLI parser."""
    parser = argparse.ArgumentParser(prog="flashrl", description="FlashRL component runtime")
    subparsers = parser.add_subparsers(dest="component_command", required=True)
    run = subparsers.add_parser("run", help="Run one FlashRL platform component")
    run.add_argument(
        "component",
        choices=["controller", "rollout", "reward", "learner", "serving-vllm"],
    )
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one platform component entrypoint."""
    parser = build_component_argument_parser()
    args = parser.parse_args(argv)
    if args.component_command != "run":
        parser.error("unsupported component command")

    if args.component == "controller":
        uvicorn.run(create_controller_app(), host=args.host, port=args.port, log_level="info")
        return 0

    job = load_job_config()
    if args.component == "rollout":
        app = create_rollout_component_app(job)
    elif args.component == "reward":
        app = create_reward_component_app(job)
    elif args.component == "learner":
        app = create_learner_component_app(job)
    elif args.component == "serving-vllm":
        app = create_serving_component_app(job)
    else:  # pragma: no cover - argparse choices guard this.
        parser.error(f"unknown component {args.component!r}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
