"""CLI for FlashRL platform jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import yaml

from flashrl.platform.config import PlatformConfig, build_flashrl_job, load_flashrl_config
from flashrl.platform.k8s.job import FlashRLJob
from flashrl.platform.k8s.job_resources import render_job_resources
from flashrl.platform.k8s.operator import FlashRLOperator


def _load_job_file(path: str | Path) -> FlashRLJob:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return FlashRLJob.model_validate(payload)


def _load_job_from_config(*, config: str) -> FlashRLJob:
    resolved = load_flashrl_config(config)
    if resolved.platform is None:
        raise ValueError("`flashrl platform render` requires a top-level `platform:` section.")
    return build_flashrl_job(
        run_config=resolved.framework,
        platform_config=PlatformConfig.model_validate(resolved.platform),
    )


def _dump_yaml(payload: dict[str, object]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def _recent_job_events(payload: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    status = payload.get("status") or {}
    events = status.get("events") or []
    if not isinstance(events, list):
        return []
    return [dict(item) for item in events[-limit:] if isinstance(item, dict)]


def _print_job_logs_summary(payload: dict[str, Any]) -> None:
    status = payload.get("status") or {}
    progress = status.get("progress") or {}
    print(f"job: {payload.get('metadata', {}).get('name', '<unknown>')}")
    print(f"phase: {status.get('phase', '<unknown>')}")
    print(f"log_root: {status.get('logRoot') or '<unknown>'}")
    print(f"controller_run_dir: {status.get('activeControllerRunDir') or '<unknown>'}")
    print(f"controller_run_id: {status.get('activeControllerRunId') or '<unknown>'}")
    print(
        "progress:"
        f" epoch={progress.get('currentEpoch', 0)}"
        f" step={progress.get('currentStep', 0)}"
        f" lastCompletedStep={progress.get('lastCompletedStep', 0)}"
    )
    last_error = status.get("lastError")
    if last_error:
        print(f"last_error: {last_error}")
    events = _recent_job_events(payload)
    if not events:
        print("recent_events: <none>")
        return
    print("recent_events:")
    for event in events:
        component = event.get("component") or "job"
        print(
            f"  {event.get('timestamp', '<unknown>')} "
            f"[{component}] {event.get('event', 'event')} "
            f"{event.get('message', '')}"
        )


def _stream_status_events(
    *,
    operator: FlashRLOperator,
    name: str,
    namespace: str,
    poll_seconds: float,
    stop_event: threading.Event,
) -> None:
    seen: set[tuple[str, str, str]] = set()
    while not stop_event.is_set():
        try:
            payload = operator.get_job(name, namespace=namespace)
        except Exception:
            time.sleep(poll_seconds)
            continue
        for event in _recent_job_events(payload, limit=100):
            key = (
                str(event.get("timestamp", "")),
                str(event.get("event", "")),
                str(event.get("message", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            component = event.get("component") or "job"
            print(
                f"{event.get('timestamp', '<unknown>')} "
                f"[{component}] {event.get('event', 'event')} "
                f"{event.get('message', '')}",
                flush=True,
            )
        time.sleep(poll_seconds)


def _follow_component_logs(
    *,
    name: str,
    namespace: str,
    component: str,
    operator_namespace: str,
) -> int:
    if component == "operator":
        command = [
            "kubectl",
            "logs",
            "-f",
            "-n",
            operator_namespace,
            "-l",
            "app.kubernetes.io/name=flashrl-operator",
            "--tail=200",
            "--prefix=true",
        ]
    else:
        selector = f"flashrl.dev/job={name}"
        if component != "all":
            selector += f",app.kubernetes.io/component={component}"
        command = [
            "kubectl",
            "logs",
            "-f",
            "-n",
            namespace,
            "-l",
            selector,
            "--tail=200",
            "--prefix=true",
        ]
    return subprocess.call(command)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the platform CLI parser."""
    parser = argparse.ArgumentParser(prog="flashrl", description="FlashRL platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    platform_parser = subparsers.add_parser("platform", help="Manage FlashRL platform jobs")
    platform_subparsers = platform_parser.add_subparsers(dest="platform_command", required=True)

    render = platform_subparsers.add_parser("render", help="Render one FlashRLJob from one config file")
    render.add_argument("--config", required=True, help="Path to one FlashRL config.yaml file.")
    render.add_argument("--output", default=None, help="Optional path to write rendered YAML.")
    render.add_argument(
        "--children",
        action="store_true",
        help="Render the Kubernetes job resources instead of the FlashRLJob itself.",
    )

    submit = platform_subparsers.add_parser("submit", help="Submit one FlashRLJob")
    submit.add_argument("--file", help="Path to a FlashRLJob YAML file.")
    submit.add_argument("--config", help="Path to one FlashRL config.yaml file.")
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

    logs = platform_subparsers.add_parser("logs", help="Show platform job logs and progress")
    logs.add_argument("name", help="FlashRLJob name.")
    logs.add_argument("--namespace", default="default")
    logs.add_argument(
        "--follow",
        action="store_true",
        help="Stream recent job events and selected component logs.",
    )
    logs.add_argument(
        "--component",
        choices=("controller", "learner", "serving", "rollout", "reward", "operator", "all"),
        default="controller",
        help="Which component logs to follow when --follow is set.",
    )
    logs.add_argument("--poll-seconds", type=float, default=5.0)
    logs.add_argument("--operator-namespace", default="flashrl-system")

    operator = platform_subparsers.add_parser("operator", help="Run the FlashRL platform operator")
    operator.add_argument("--namespace", default=None, help="Namespace to watch. Defaults to all namespaces.")
    operator.add_argument("--resync-seconds", type=int, default=60)

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
        )
        if args.children:
            payload = json.dumps(render_job_resources(job), indent=2)
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
            )
        else:
            raise ValueError("Pass either --file or --config.")
        if args.namespace is not None:
            job.metadata["namespace"] = args.namespace
        if args.render_only:
            print(json.dumps(render_job_resources(job), indent=2))
            return 0
        operator = FlashRLOperator()
        operator.ensure_crd()
        namespace = str(job.metadata.get("namespace") or "default")
        operator.apply_job(job, namespace=namespace)
        print(f"Submitted FlashRLJob {job.name} to namespace={namespace}.")
        return 0

    if args.platform_command == "logs":
        operator = FlashRLOperator()
        payload = operator.get_job(args.name, namespace=args.namespace)
        _print_job_logs_summary(payload)
        if not args.follow:
            return 0
        stop_event = threading.Event()
        watcher = threading.Thread(
            target=_stream_status_events,
            kwargs={
                "operator": operator,
                "name": args.name,
                "namespace": args.namespace,
                "poll_seconds": float(args.poll_seconds),
                "stop_event": stop_event,
            },
            daemon=True,
        )
        watcher.start()
        try:
            return int(
                _follow_component_logs(
                    name=args.name,
                    namespace=args.namespace,
                    component=args.component,
                    operator_namespace=args.operator_namespace,
                )
            )
        finally:
            stop_event.set()
            watcher.join(timeout=1.0)
        return 0

    if args.platform_command == "operator":
        operator = FlashRLOperator()
        operator.ensure_crd()
        operator.watch(namespace=args.namespace, resync_seconds=args.resync_seconds)
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
