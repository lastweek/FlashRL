"""Shared helpers for controller and component pod runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from flashrl.framework import runtime_support
from flashrl.framework.config import RolloutConfig
from flashrl.framework.data_models import Prompt
from flashrl.platform.k8s.job import DatasetSpec, FlashRLJob


def load_job_config(path: str | Path | None = None) -> FlashRLJob:
    """Load the mounted FlashRLJob spec for one pod."""
    resolved_path = Path(path or os.environ.get("FLASHRL_JOB_CONFIG_PATH", ""))
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"FlashRL job config was not found at {resolved_path}."
        )
    return FlashRLJob.model_validate(json.loads(resolved_path.read_text(encoding="utf-8")))


def load_dataset(job: FlashRLJob) -> list[Prompt]:
    """Resolve the controller-owned dataset source for one platform run."""
    dataset_spec = job.spec.dataset
    if dataset_spec.type == "hook":
        if job.spec.userCode.dataset is None:
            raise ValueError("dataset.type='hook' requires userCode.dataset.")
        dataset = runtime_support.instantiate_hook(job.spec.userCode.dataset)
        return runtime_support.normalize_dataset(dataset)
    return _load_dataset_from_uri(dataset_spec)


def build_rollout_config(job: FlashRLJob) -> RolloutConfig:
    """Build rollout-generation config from the GRPO section."""
    return runtime_support.build_rollout_config(job.spec.framework.grpo)


def service_url(component: str) -> str:
    """Resolve one in-cluster service URL from env or job naming conventions."""
    env_name = f"FLASHRL_{component.upper()}_URL"
    value = os.environ.get(env_name)
    if value:
        return value.rstrip("/")
    job_name = os.environ.get("FLASHRL_JOB_NAME", "flashrl")
    namespace = os.environ.get("FLASHRL_NAMESPACE", "default")
    if component == "learner":
        return f"http://{job_name}-learner-0.{job_name}-learner.{namespace}.svc.cluster.local"
    return f"http://{job_name}-{component}.{namespace}.svc.cluster.local"


def shared_path(uri: str, *, purpose: str) -> Path:
    """Resolve a platform path/URI into one container filesystem path."""
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        raw_path = parsed.path if parsed.scheme == "file" else uri
        return Path(raw_path).expanduser()
    raise ValueError(
        f"Only plain paths and file:// URIs are supported for platform {purpose} storage today; got {uri!r}."
    )


def _load_dataset_from_uri(dataset_spec: DatasetSpec) -> list[Prompt]:
    path = shared_path(str(dataset_spec.uri), purpose="dataset")
    if dataset_spec.format == "jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif dataset_spec.format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
    else:
        raise NotImplementedError(
            f"Platform dataset.type='uri' currently supports only json/jsonl paths; got format={dataset_spec.format!r}."
        )

    normalized: list[Prompt] = []
    for item in items:
        if isinstance(item, str):
            normalized.append(Prompt(text=item))
            continue
        if isinstance(item, dict):
            normalized.append(Prompt.model_validate(item))
            continue
        normalized.append(Prompt(text=str(item)))
    return normalized
