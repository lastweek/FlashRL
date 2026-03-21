"""Mounted pod contract shared by every FlashRL platform workload."""

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
