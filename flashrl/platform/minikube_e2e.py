"""Local minikube end-to-end runner for FlashRL platform jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from flashrl.platform.config import PlatformConfig, load_flashrl_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "flashrl/examples/math/config.yaml"
DEFAULT_PROFILE = "minikube"
DEFAULT_OPERATOR_NAMESPACE = "flashrl-system"
DEFAULT_OPERATOR_IMAGE = "flashrl-operator:minikube"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_POLL_SECONDS = 5.0
RESOURCE_KIND = "flashrljob"

IMAGE_BUILDS: tuple[tuple[str, str], ...] = (
    ("flashrl-operator:minikube", "docker/operator.Dockerfile"),
    ("flashrl-runtime:minikube", "docker/runtime.Dockerfile"),
    ("flashrl-serving-vllm:minikube", "docker/serving-vllm.Dockerfile"),
    ("flashrl-training-fsdp:minikube", "docker/training-fsdp.Dockerfile"),
)


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    capture_output: bool = False,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=check,
        text=True,
        input=input_text,
        capture_output=capture_output,
    )


def _run_capture(command: list[str], *, cwd: Path = REPO_ROOT) -> str:
    return _run(command, cwd=cwd, capture_output=True).stdout


def _json_output(command: list[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    payload = _run_capture(command, cwd=cwd)
    return json.loads(payload or "{}")


def _write_yaml_documents(path: Path, documents: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump_all(documents, sort_keys=False),
        encoding="utf-8",
    )


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [item for item in yaml.safe_load_all(handle) if item is not None]


def _ensure_minikube_cluster(profile: str) -> None:
    try:
        _run(["minikube", "-p", profile, "status"], capture_output=True)
    except subprocess.CalledProcessError:
        _run(["minikube", "start", "-p", profile])
    current_context = _run_capture(["kubectl", "config", "current-context"]).strip()
    if current_context != profile:
        _run(["minikube", "-p", profile, "update-context"])
    _run(["kubectl", "cluster-info"], capture_output=True)


def _build_images(profile: str) -> None:
    for tag, dockerfile in IMAGE_BUILDS:
        _run(
            [
                "minikube",
                "-p",
                profile,
                "image",
                "build",
                "-t",
                tag,
                "-f",
                dockerfile,
                ".",
            ]
        )


def _kubectl_apply(payloads: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        _write_yaml_documents(temp_path, payloads)
        _run(["kubectl", "apply", "-f", str(temp_path)])
    finally:
        temp_path.unlink(missing_ok=True)


def _kubectl_apply_file(path: Path) -> None:
    _run(["kubectl", "apply", "-f", str(path)])


def _ensure_namespace(namespace: str) -> None:
    try:
        _run(["kubectl", "get", "namespace", namespace], capture_output=True)
    except subprocess.CalledProcessError:
        _run(["kubectl", "create", "namespace", namespace])


def _wait_for_rollout(namespace: str, workload: str, name: str, *, timeout_seconds: int) -> None:
    _run(
        [
            "kubectl",
            "rollout",
            "status",
            f"{workload}/{name}",
            "-n",
            namespace,
            f"--timeout={timeout_seconds}s",
        ]
    )


def _wait_for_pods_ready(namespace: str, job_name: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    expected_components = {"controller", "learner", "serving", "rollout", "reward"}
    while time.monotonic() < deadline:
        pods = _json_output(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"flashrl.dev/job={job_name}",
                "-o",
                "json",
            ]
        ).get("items", [])
        seen_components: set[str] = set()
        ready_components: set[str] = set()
        for pod in pods:
            labels = pod.get("metadata", {}).get("labels", {})
            component = labels.get("app.kubernetes.io/component")
            if component is None:
                continue
            seen_components.add(component)
            conditions = pod.get("status", {}).get("conditions", [])
            if any(item.get("type") == "Ready" and item.get("status") == "True" for item in conditions):
                ready_components.add(component)
        if expected_components.issubset(seen_components) and expected_components.issubset(ready_components):
            return
        time.sleep(DEFAULT_POLL_SECONDS)
    raise RuntimeError(f"Timed out waiting for all FlashRL component pods to become Ready in namespace={namespace}.")


def _get_job_status(namespace: str, job_name: str) -> dict[str, Any]:
    return _json_output(
        [
            "kubectl",
            "get",
            RESOURCE_KIND,
            job_name,
            "-n",
            namespace,
            "-o",
            "json",
        ]
    )


def _wait_for_job_progress(namespace: str, job_name: str, *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = _get_job_status(namespace, job_name)
        status = payload.get("status", {})
        phase = status.get("phase")
        if phase == "Failed":
            raise RuntimeError(f"FlashRLJob {job_name} entered Failed phase.")
        progress = status.get("progress", {})
        weight_version = status.get("weightVersion", {})
        active_version = weight_version.get("active") or {}
        last_completed_step = int(progress.get("lastCompletedStep", 0) or 0)
        active_version_id = int(active_version.get("version_id", 0) or 0)
        if last_completed_step >= 1 and active_version_id >= 1:
            return payload
        time.sleep(DEFAULT_POLL_SECONDS)
    raise RuntimeError(f"Timed out waiting for FlashRLJob {job_name} to complete at least one step.")


def _artifact_listing(namespace: str, pod_name: str, mount_path: str) -> list[str]:
    output = _run_capture(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            pod_name,
            "--",
            "sh",
            "-lc",
            f"find {shlex.quote(mount_path)} -type f | sort",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _controller_pod_name(namespace: str, job_name: str) -> str:
    payload = _json_output(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"flashrl.dev/job={job_name},app.kubernetes.io/component=controller",
            "-o",
            "json",
        ]
    )
    items = payload.get("items", [])
    if not items:
        raise RuntimeError("Controller pod was not found.")
    return str(items[0]["metadata"]["name"])


def _collect_component_logs(namespace: str, selector: str) -> str:
    try:
        return _run_capture(["kubectl", "logs", "-n", namespace, "-l", selector, "--prefix=true", "--tail=500"])
    except subprocess.CalledProcessError as exc:
        return exc.stdout or exc.stderr or ""


def _write_diagnostics(
    *,
    artifact_dir: Path,
    job_name: str,
    namespace: str,
    operator_namespace: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "platform-status.json": [
            sys.executable,
            "-m",
            "flashrl",
            "platform",
            "status",
            job_name,
            "--namespace",
            namespace,
        ],
        "kubectl-get-all.txt": [
            "kubectl",
            "get",
            RESOURCE_KIND,
            "pods",
            "deployments",
            "statefulsets",
            "services",
            "pvc",
            "-n",
            namespace,
            "-o",
            "wide",
        ],
        "flashrljob-describe.txt": [
            "kubectl",
            "describe",
            RESOURCE_KIND,
            job_name,
            "-n",
            namespace,
        ],
        "pods-describe.txt": [
            "kubectl",
            "describe",
            "pods",
            "-n",
            namespace,
            "-l",
            f"flashrl.dev/job={job_name}",
        ],
        "operator-logs.txt": [
            "kubectl",
            "logs",
            "-n",
            operator_namespace,
            "-l",
            "app.kubernetes.io/name=flashrl-operator",
            "--tail=500",
        ],
    }
    for filename, command in commands.items():
        try:
            output = _run_capture(command)
        except subprocess.CalledProcessError as exc:
            output = exc.stdout or exc.stderr or ""
        (artifact_dir / filename).write_text(output, encoding="utf-8")

    for component in ("controller", "learner", "serving", "rollout", "reward"):
        selector = f"flashrl.dev/job={job_name},app.kubernetes.io/component={component}"
        (artifact_dir / f"{component}.log").write_text(
            _collect_component_logs(namespace, selector),
            encoding="utf-8",
        )


def _delete_job(namespace: str, job_name: str) -> None:
    _run(
        [
            "kubectl",
            "delete",
            RESOURCE_KIND,
            job_name,
            "-n",
            namespace,
            "--ignore-not-found=true",
            "--wait=true",
        ],
        check=False,
    )


def run_minikube_math_e2e(
    *,
    config: str | Path = DEFAULT_CONFIG,
    profile: str = DEFAULT_PROFILE,
    minikube_profile: str = "minikube",
    operator_namespace: str = DEFAULT_OPERATOR_NAMESPACE,
    operator_image: str = DEFAULT_OPERATOR_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    keep_resources: bool = False,
    skip_build: bool = False,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the math example end to end on a local minikube cluster."""
    config_path = Path(config)
    resolved = load_flashrl_config(str(config_path), profile=profile)
    if resolved.platform is None:
        raise ValueError("Minikube E2E requires `platform:` in the selected config profile.")
    platform = PlatformConfig.model_validate(resolved.platform)
    namespace = platform.job.namespace
    job_name = platform.job.name
    artifact_root = Path(artifact_dir) if artifact_dir is not None else REPO_ROOT / "logs" / "platform-e2e" / job_name

    _ensure_minikube_cluster(minikube_profile)
    if not skip_build:
        _build_images(minikube_profile)

    _ensure_namespace(operator_namespace)
    _ensure_namespace(namespace)
    _kubectl_apply_file(REPO_ROOT / "flashrl/platform/k8s/namespace.yaml")
    _kubectl_apply_file(REPO_ROOT / "flashrl/platform/k8s/crd.yaml")
    rbac_docs = _load_yaml_documents(REPO_ROOT / "flashrl/platform/k8s/operator-rbac.yaml")
    for payload in rbac_docs:
        if payload.get("kind") == "ServiceAccount":
            payload["metadata"]["namespace"] = operator_namespace
        if payload.get("kind") == "ClusterRoleBinding":
            payload["subjects"][0]["namespace"] = operator_namespace
    _kubectl_apply(rbac_docs)
    operator_docs = _load_yaml_documents(REPO_ROOT / "flashrl/platform/k8s/operator.yaml")
    for payload in operator_docs:
        if payload.get("kind") == "Deployment":
            payload["metadata"]["namespace"] = operator_namespace
            payload["spec"]["template"]["spec"]["containers"][0]["image"] = operator_image
    _kubectl_apply(operator_docs)
    _wait_for_rollout(operator_namespace, "deployment", "flashrl-operator", timeout_seconds=timeout_seconds)

    _delete_job(namespace, job_name)
    _run(
        [
            sys.executable,
            "-m",
            "flashrl",
            "platform",
            "submit",
            "--config",
            str(config_path),
            "--profile",
            profile,
        ]
    )

    try:
        _wait_for_pods_ready(namespace, job_name, timeout_seconds=timeout_seconds)
        job_payload = _wait_for_job_progress(namespace, job_name, timeout_seconds=timeout_seconds)
        controller_pod = _controller_pod_name(namespace, job_name)
        files = _artifact_listing(namespace, controller_pod, platform.sharedStorage.mountPath)
        if not any("/checkpoints/" in path for path in files):
            raise RuntimeError("No checkpoint artifacts were found in the shared storage mount.")
        if not any("/weights/" in path for path in files):
            raise RuntimeError("No published weight artifacts were found in the shared storage mount.")
        return {
            "job": job_payload,
            "artifacts": files,
            "namespace": namespace,
            "job_name": job_name,
        }
    except Exception:
        _write_diagnostics(
            artifact_dir=artifact_root,
            job_name=job_name,
            namespace=namespace,
            operator_namespace=operator_namespace,
        )
        raise
    finally:
        if not keep_resources:
            _delete_job(namespace, job_name)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the minikube E2E CLI parser."""
    parser = argparse.ArgumentParser(description="Run the FlashRL minikube platform E2E.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--minikube-profile", default="minikube")
    parser.add_argument("--operator-namespace", default=DEFAULT_OPERATOR_NAMESPACE)
    parser.add_argument("--operator-image", default=DEFAULT_OPERATOR_IMAGE)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-resources", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the local minikube E2E runner."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_minikube_math_e2e(
        config=args.config,
        profile=args.profile,
        minikube_profile=args.minikube_profile,
        operator_namespace=args.operator_namespace,
        operator_image=args.operator_image,
        timeout_seconds=args.timeout_seconds,
        keep_resources=args.keep_resources,
        skip_build=args.skip_build,
        artifact_dir=args.artifact_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
