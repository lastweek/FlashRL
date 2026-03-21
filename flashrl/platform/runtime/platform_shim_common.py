"""Literal mounted pod contract shared by every PlatformShim."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from flashrl.platform.k8s.job import FlashRLJob


def load_mounted_job(path: str | Path | None = None) -> FlashRLJob:
    """Load the FlashRLJob spec mounted into one workload pod."""
    resolved_path = Path(path or os.environ.get("FLASHRL_JOB_CONFIG_PATH", ""))
    if not resolved_path.exists():
        raise FileNotFoundError(f"FlashRL job config was not found at {resolved_path}.")
    return FlashRLJob.model_validate(json.loads(resolved_path.read_text(encoding="utf-8")))


def service_url_for(workload: str) -> str:
    """Resolve one sibling workload URL from env or in-cluster naming rules."""
    env_name = f"FLASHRL_{workload.upper()}_URL"
    value = os.environ.get(env_name)
    if value:
        return value.rstrip("/")
    job_name = os.environ.get("FLASHRL_JOB_NAME", "flashrl")
    namespace = os.environ.get("FLASHRL_NAMESPACE", "default")
    if workload == "learner":
        return f"http://{job_name}-learner-0.{job_name}-learner.{namespace}.svc.cluster.local"
    return f"http://{job_name}-{workload}.{namespace}.svc.cluster.local"


def storage_path_from_uri(uri: str, *, purpose: str) -> Path:
    """Resolve a platform storage URI into one container filesystem path."""
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        raw_path = parsed.path if parsed.scheme == "file" else uri
        return Path(raw_path).expanduser()
    raise ValueError(
        f"Only plain paths and file:// URIs are supported for platform {purpose} storage today; got {uri!r}."
    )


def job_uid_for(job: FlashRLJob) -> str:
    """Resolve the stable per-job UID used in pod env and log paths."""
    env_value = os.environ.get("FLASHRL_JOB_UID")
    if env_value:
        return env_value
    metadata_uid = job.metadata.get("uid")
    if metadata_uid:
        return str(metadata_uid)
    return f"{job.name}-pending"


def pod_name_for(component: str) -> str:
    """Resolve the current pod name or a readable fallback during tests."""
    env_value = os.environ.get("FLASHRL_POD_NAME")
    if env_value:
        return env_value
    return f"{os.environ.get('FLASHRL_JOB_NAME', 'flashrl')}-{component}"


def resolve_job_log_root(job: FlashRLJob) -> Path:
    """Resolve the canonical job log root for one platform run."""
    env_value = os.environ.get("FLASHRL_JOB_LOG_ROOT")
    if env_value:
        return Path(env_value)
    log_dir = Path(job.spec.framework.logging.log_dir).expanduser()
    job_suffix = Path(job.name) / job_uid_for(job)
    if log_dir.is_absolute():
        return log_dir / job_suffix
    if job.spec.sharedStorage.enabled:
        mount_path = Path(job.spec.sharedStorage.mountPath)
        return mount_path / log_dir / job_suffix
    fallback_root = Path("/tmp/flashrl-platform-logs")
    return fallback_root / log_dir / job_suffix


def resolve_component_log_dir(job: FlashRLJob, component: str) -> Path:
    """Resolve the canonical per-pod log directory for one component pod."""
    env_value = os.environ.get("FLASHRL_COMPONENT_LOG_DIR")
    if env_value:
        return Path(env_value) / pod_name_for(component)
    return resolve_job_log_root(job) / "_pods" / component / pod_name_for(component)


def component_log_metadata(job: FlashRLJob, component: str) -> dict[str, str]:
    """Build status metadata describing where one platform pod writes logs."""
    return {
        "jobLogRoot": str(resolve_job_log_root(job)),
        "componentLogDir": str(resolve_component_log_dir(job, component)),
        "podName": pod_name_for(component),
        "jobUid": job_uid_for(job),
    }
