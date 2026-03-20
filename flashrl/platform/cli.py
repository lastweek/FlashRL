"""CLI for FlashRL platform jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from flashrl.platform.crd import FlashRLJob
from flashrl.platform.operator import FlashRLOperator, render_child_resources


def _load_job(path: str | Path) -> FlashRLJob:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return FlashRLJob.model_validate(payload)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the platform CLI parser."""
    parser = argparse.ArgumentParser(prog="flashrl", description="FlashRL platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    platform_parser = subparsers.add_parser("platform", help="Manage FlashRL platform jobs")
    platform_subparsers = platform_parser.add_subparsers(dest="platform_command", required=True)

    submit = platform_subparsers.add_parser("submit", help="Submit one FlashRLJob")
    submit.add_argument("--file", required=True, help="Path to a FlashRLJob YAML file.")
    submit.add_argument("--namespace", default="default")
    submit.add_argument(
        "--render-only",
        action="store_true",
        help="Validate the job and print rendered child resources without contacting Kubernetes.",
    )

    status = platform_subparsers.add_parser("status", help="Show one FlashRLJob status")
    status.add_argument("name", help="FlashRLJob name.")
    status.add_argument("--namespace", default="default")

    describe = platform_subparsers.add_parser("describe", help="Describe one FlashRLJob")
    describe.add_argument("name", help="FlashRLJob name.")
    describe.add_argument("--namespace", default="default")

    cancel = platform_subparsers.add_parser("cancel", help="Delete one FlashRLJob")
    cancel.add_argument("name", help="FlashRLJob name.")
    cancel.add_argument("--namespace", default="default")

    logs = platform_subparsers.add_parser("logs", help="Show child pod labels for one job")
    logs.add_argument("name", help="FlashRLJob name.")
    logs.add_argument("--namespace", default="default")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the platform CLI."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command != "platform":
        parser.error("unsupported command")

    if args.platform_command == "submit":
        job = _load_job(args.file)
        if args.render_only:
            print(json.dumps(render_child_resources(job), indent=2))
            return 0
        operator = FlashRLOperator()
        operator.ensure_crd()
        operator.submit_job(job, namespace=args.namespace)
        print(f"Submitted FlashRLJob {job.name} to namespace={args.namespace}.")
        return 0

    if args.platform_command == "logs":
        print(
            f"Select logs for job={args.name} in namespace={args.namespace} with label flashrl.dev/job={args.name}."
        )
        return 0

    operator = FlashRLOperator()
    if args.platform_command in {"status", "describe"}:
        payload = operator.get_job(args.name, namespace=args.namespace)
        print(json.dumps(payload, indent=2))
        return 0
    if args.platform_command == "cancel":
        operator.delete_job(args.name, namespace=args.namespace)
        print(f"Deleted FlashRLJob {args.name} from namespace={args.namespace}.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
