"""CLI for FlashRL platform jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from flashrl.platform.config import PlatformConfig, build_flashrl_job, load_flashrl_config
from flashrl.platform.crd import FlashRLJob
from flashrl.platform.operator import (
    FlashRLOperator,
    render_child_resources,
    render_operator_resources,
)


def _load_job_file(path: str | Path) -> FlashRLJob:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return FlashRLJob.model_validate(payload)


def _load_job_from_config(*, config: str, profile: str | None) -> FlashRLJob:
    resolved = load_flashrl_config(config, profile=profile)
    if resolved.platform is None:
        raise ValueError("`flashrl platform render` requires a top-level `platform:` section.")
    return build_flashrl_job(
        run_config=resolved.framework,
        platform_config=PlatformConfig.model_validate(resolved.platform),
    )


def _dump_yaml(payload: dict[str, object]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the platform CLI parser."""
    parser = argparse.ArgumentParser(prog="flashrl", description="FlashRL platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    platform_parser = subparsers.add_parser("platform", help="Manage FlashRL platform jobs")
    platform_subparsers = platform_parser.add_subparsers(dest="platform_command", required=True)

    render = platform_subparsers.add_parser("render", help="Render one FlashRLJob from one config file")
    render.add_argument("--config", required=True, help="Path to one FlashRL config.yaml file.")
    render.add_argument("--profile", default=None, help="Optional config profile to apply.")
    render.add_argument("--output", default=None, help="Optional path to write rendered YAML.")
    render.add_argument(
        "--children",
        action="store_true",
        help="Render child Kubernetes resources instead of the FlashRLJob itself.",
    )

    submit = platform_subparsers.add_parser("submit", help="Submit one FlashRLJob")
    submit.add_argument("--file", help="Path to a FlashRLJob YAML file.")
    submit.add_argument("--config", help="Path to one FlashRL config.yaml file.")
    submit.add_argument("--profile", default=None, help="Optional config profile to apply.")
    submit.add_argument("--namespace", default=None)
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

    operator = platform_subparsers.add_parser("operator", help="Run the FlashRL platform operator")
    operator.add_argument("--namespace", default=None, help="Namespace to watch. Defaults to all namespaces.")
    operator.add_argument("--resync-seconds", type=int, default=60)

    render_operator = platform_subparsers.add_parser(
        "render-operator",
        help="Print the operator Deployment and RBAC manifests.",
    )
    render_operator.add_argument("--namespace", default="flashrl-system")
    render_operator.add_argument("--image", default="ghcr.io/flashrl/flashrl-operator:latest")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the platform CLI."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command != "platform":
        parser.error("unsupported command")

    if args.platform_command == "render":
        job = _load_job_from_config(
            config=str(args.config),
            profile=getattr(args, "profile", None),
        )
        if args.children:
            payload = json.dumps(render_child_resources(job), indent=2)
        else:
            payload = _dump_yaml(job.model_dump(mode="json", by_alias=True))
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.platform_command == "submit":
        if getattr(args, "file", None):
            job = _load_job_file(args.file)
        elif getattr(args, "config", None):
            job = _load_job_from_config(
                config=str(args.config),
                profile=getattr(args, "profile", None),
            )
        else:
            raise ValueError("Pass either --file or --config.")
        if args.namespace is not None:
            job.metadata["namespace"] = args.namespace
        if args.render_only:
            print(json.dumps(render_child_resources(job), indent=2))
            return 0
        operator = FlashRLOperator()
        operator.ensure_crd()
        namespace = str(job.metadata.get("namespace") or "default")
        operator.submit_job(job, namespace=namespace)
        print(f"Submitted FlashRLJob {job.name} to namespace={namespace}.")
        return 0

    if args.platform_command == "logs":
        print(
            f"Select logs for job={args.name} in namespace={args.namespace} with label flashrl.dev/job={args.name}."
        )
        return 0

    if args.platform_command == "render-operator":
        print(json.dumps(render_operator_resources(namespace=args.namespace, image=args.image), indent=2))
        return 0

    if args.platform_command == "operator":
        operator = FlashRLOperator()
        operator.ensure_crd()
        operator.run_forever(namespace=args.namespace, resync_seconds=args.resync_seconds)
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
