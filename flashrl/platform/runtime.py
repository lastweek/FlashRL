"""Shared platform runtime helpers for controller and component pods."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from flashrl.framework import runtime_support
from flashrl.framework.config import RolloutConfig, ServingConfig
from flashrl.framework.data_models import Prompt
from flashrl.framework.distributed import (
    HttpServingClient,
    LocalLearnerClient,
    LocalRewardClient,
    LocalRolloutClient,
    LocalServingClient,
    create_learner_app,
    create_reward_app,
    create_rollout_app,
    create_serving_app,
)
from flashrl.framework.distributed.remote_serving_backend import HttpServingBackend
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.base import build_rollout_generator
from flashrl.framework.serving import create_serving_backend
from flashrl.framework.training import create_training_backend
from flashrl.platform.crd import DatasetSpec, FlashRLJob


def load_job_config(path: str | Path | None = None) -> FlashRLJob:
    """Load the mounted FlashRLJob spec for one component pod."""
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


def create_rollout_component_app(job: FlashRLJob):
    """Build the rollout FastAPI app for one platform job."""
    rollout_impl = runtime_support.instantiate_hook(job.spec.userCode.rollout)
    serving_backend = HttpServingBackend(
        config=job.spec.framework.serving.model_copy(deep=True),
        client=HttpServingClient(_service_url("serving")),
    )
    rollout_generator = build_rollout_generator(
        rollout_fn=rollout_impl,
        serving_backend=serving_backend,
        config=_build_rollout_config(job),
    )
    return create_rollout_app(LocalRolloutClient(rollout_generator))


def create_reward_component_app(job: FlashRLJob):
    """Build the reward FastAPI app for one platform job."""
    reward_impl = runtime_support.instantiate_hook(job.spec.userCode.reward)
    reward = UserDefinedReward(reward_fn=reward_impl, config=job.spec.framework.grpo)
    return create_reward_app(LocalRewardClient(reward))


def create_learner_component_app(job: FlashRLJob):
    """Build the learner FastAPI app for one platform job."""
    actor_backend = create_training_backend(
        job.spec.framework.actor.model_copy(deep=True),
        role="actor",
        learning_rate=job.spec.framework.trainer.learning_rate,
    )
    reference_backend = (
        create_training_backend(
            job.spec.framework.reference.model_copy(deep=True),
            role="reference",
        )
        if job.spec.framework.reference is not None
        else None
    )
    publish_dir = _shared_path(job.spec.storage.weights.uriPrefix, purpose="weights")
    publish_dir.mkdir(parents=True, exist_ok=True)
    client = LocalLearnerClient(
        actor_backend,
        reference_backend,
        grpo_config=job.spec.framework.grpo,
        publish_dir=publish_dir,
        synchronize_serving=False,
    )
    return create_learner_app(client)


def create_serving_component_app(job: FlashRLJob):
    """Build the serving FastAPI app for one platform job."""
    serving_backend = create_serving_backend(
        job.spec.framework.serving.model_copy(deep=True),
        log_dir=_shared_path(job.spec.storage.weights.uriPrefix, purpose="serving-cache"),
    )
    return create_serving_app(LocalServingClient(serving_backend))


def _build_rollout_config(job: FlashRLJob) -> RolloutConfig:
    return runtime_support.build_rollout_config(job.spec.framework.grpo)


def _service_url(component: str) -> str:
    env_name = f"FLASHRL_{component.upper()}_URL"
    value = os.environ.get(env_name)
    if value:
        return value.rstrip("/")
    job_name = os.environ.get("FLASHRL_JOB_NAME", "flashrl")
    namespace = os.environ.get("FLASHRL_NAMESPACE", "default")
    if component == "learner":
        return f"http://{job_name}-learner-0.{job_name}-learner.{namespace}.svc.cluster.local"
    return f"http://{job_name}-{component}.{namespace}.svc.cluster.local"


def _shared_path(uri: str, *, purpose: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        raw_path = parsed.path if parsed.scheme == "file" else uri
        return Path(raw_path).expanduser()
    raise ValueError(
        f"Only plain paths and file:// URIs are supported for platform {purpose} storage today; got {uri!r}."
    )


def _load_dataset_from_uri(dataset_spec: DatasetSpec) -> list[Prompt]:
    path = _shared_path(str(dataset_spec.uri), purpose="dataset")
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
