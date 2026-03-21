"""Explicit controller pod runtime for FlashRL platform mode."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from fastapi import FastAPI
import uvicorn

from flashrl.framework.admin import utc_now_iso
from flashrl.framework import runtime_support
from flashrl.framework.distributed.models import ComponentStatus
from flashrl.framework.distributed.server_common import install_common_routes
from flashrl.framework.distributed import (
    LearnerClient,
    RewardClient,
    RolloutClient,
    ServingClient,
)
from flashrl.framework.data_models import Prompt
from flashrl.framework.trainer.grpo.trainer import GRPOTrainer
from flashrl.platform.k8s.job import DatasetSpec, FlashRLJob, GROUP, PLURAL, VERSION
from flashrl.platform.runtime.pod import (
    load_mounted_job,
    service_url_for,
    storage_path_from_uri,
)


class FlashRLJobStatusWriter:
    """Best-effort CRD status writer used by the controller pod."""

    def __init__(self, *, job_name: str, namespace: str) -> None:
        self._job_name = job_name
        self._namespace = namespace
        self._api = None

    def get_job(self) -> FlashRLJob | None:
        api = self._load_api()
        if api is None:
            return None
        try:
            payload = api.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=self._namespace,
                plural=PLURAL,
                name=self._job_name,
            )
        except Exception:
            return None
        return FlashRLJob.model_validate(payload)

    def patch_status(self, patch: dict[str, Any]) -> None:
        api = self._load_api()
        if api is None:
            return
        api.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=self._namespace,
            plural=PLURAL,
            name=self._job_name,
            body={"status": patch},
        )

    def _load_api(self):
        if self._api is not None:
            return self._api
        try:
            from kubernetes import client, config
        except ImportError:
            self._api = None
            return None
        try:
            config.load_incluster_config()
        except Exception:
            try:
                config.load_kube_config()
            except Exception:
                self._api = None
                return None
        self._api = client.CustomObjectsApi()
        return self._api


def load_controller_dataset(job: FlashRLJob) -> list[Prompt]:
    """Resolve the controller-owned dataset source for one platform run."""
    dataset_spec = job.spec.dataset
    if dataset_spec.type == "hook":
        if job.spec.userCode.dataset is None:
            raise ValueError("dataset.type='hook' requires userCode.dataset.")
        dataset = runtime_support.instantiate_hook(job.spec.userCode.dataset)
        return runtime_support.normalize_dataset(dataset)
    return _load_dataset_from_uri(dataset_spec)


class _ControllerServiceState:
    """Track the long-lived controller pod state while training runs in the background."""

    def __init__(self, *, job_path: str | Path | None = None) -> None:
        self._job_path = job_path
        self._lock = Lock()
        self._phase = "Starting"
        self._healthy = True
        self._last_error: str | None = None
        self._started = False
        self._completed = False
        self._thread: Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._started = True
            self._phase = "Running"
            self._thread = Thread(target=self._run, daemon=True, name="flashrl-controller")
            self._thread.start()

    def status(self) -> ComponentStatus:
        with self._lock:
            return ComponentStatus(
                name="controller",
                phase=self._phase,
                healthy=self._healthy,
                ready_replica_count=1 if self._healthy and self._started else 0,
                desired_replica_count=1,
                last_error=self._last_error,
                metadata={
                    "started": self._started,
                    "completed": self._completed,
                },
            )

    def _run(self) -> None:
        try:
            run_controller(self._job_path)
        except Exception as exc:
            with self._lock:
                self._phase = "Failed"
                self._healthy = False
                self._last_error = str(exc)
            return
        with self._lock:
            self._phase = "Succeeded"
            self._healthy = True
            self._completed = True


def create_controller_app(job_path: str | Path | None = None) -> FastAPI:
    """Create the long-lived controller pod app with background training execution.

    This module is the only platform-specific long-running runtime. It loads
    the mounted job, talks to the distributed services through remote clients, and patches
    controller-owned status back into the FlashRLJob CR.
    """
    state = _ControllerServiceState(job_path=job_path)
    app = FastAPI(title="FlashRL Controller")
    install_common_routes(
        app,
        status_getter=state.status,
        kind="ControllerService",
        name="controller",
    )

    @app.on_event("startup")
    def startup() -> None:
        state.start()

    return app


def run_controller_pod(*, host: str = "0.0.0.0", port: int = 8000) -> int:
    """Run the controller pod HTTP server."""
    uvicorn.run(create_controller_app(), host=host, port=port, log_level="info")
    return 0


def run_controller(job_path: str | Path | None = None) -> None:
    """Run the GRPO controller loop for one platform job."""
    static_job = load_mounted_job(job_path)
    namespace = str(
        os.environ.get("FLASHRL_NAMESPACE")
        or static_job.metadata.get("namespace")
        or "default"
    )
    os.environ.setdefault("FLASHRL_JOB_NAME", static_job.name)
    os.environ.setdefault("FLASHRL_NAMESPACE", namespace)
    status_writer = FlashRLJobStatusWriter(job_name=static_job.name, namespace=namespace)
    live_job = status_writer.get_job()
    job = static_job
    if live_job is not None:
        job.status = live_job.status.model_copy(deep=True)
        for metadata_key in ("uid", "resourceVersion", "generation", "deletionTimestamp", "finalizers"):
            if metadata_key in live_job.metadata:
                job.metadata[metadata_key] = live_job.metadata[metadata_key]

    rollout = RolloutClient(service_url_for("rollout"))
    reward = RewardClient(service_url_for("reward"))
    learner = LearnerClient(service_url_for("learner"))
    serving = ServingClient(service_url_for("serving"))

    trainer = GRPOTrainer(
        config=job.spec.framework.trainer,
        grpo_config=job.spec.framework.grpo,
        actor_backend=None,
        reference_backend=None,
        serving_backend=None,
        reward_fn=None,
        rollout_generator=None,
        rollout=rollout,
        reward=reward,
        learner=learner,
        serving=serving,
        reference_configured=job.spec.framework.reference is not None,
        on_step_complete=lambda info: _on_step_complete(
            info=info,
            job=job,
            trainer=trainer,
            serving=serving,
            status_writer=status_writer,
        ),
    )

    status_writer.patch_status(
        {
            "startedAt": job.status.startedAt or utc_now_iso(),
            "progress": job.status.progress.model_dump(mode="json"),
        }
    )

    resume_target = _resume_target(job)
    if resume_target is not None:
        trainer.load_checkpoint(resume_target)
        status_writer.patch_status(
            {
                "progress": {
                    "currentEpoch": trainer.current_epoch,
                    "currentStep": trainer.total_steps,
                    "lastCompletedStep": trainer.total_steps,
                }
            }
        )

    dataset = load_controller_dataset(job)

    try:
        trainer.train(dataset)
        if job.spec.checkpointing.save_on_run_end:
            checkpoint_path = _checkpoint_path(job, step=trainer.total_steps, final=True)
            trainer.save_checkpoint(str(checkpoint_path))
            status_writer.patch_status(
                {
                    "checkpoint": {
                        "latestUri": str(checkpoint_path),
                        "lastSavedAt": utc_now_iso(),
                    }
                }
            )
        status_writer.patch_status({"finishedAt": utc_now_iso()})
    except Exception:
        status_writer.patch_status({"finishedAt": utc_now_iso()})
        raise


def _resume_target(job: FlashRLJob) -> str | None:
    resume_from = job.spec.checkpointing.resume_from
    if resume_from is None:
        return None
    if resume_from == "latest":
        return job.status.checkpoint.latestUri
    return str(resume_from)


def _checkpoint_path(job: FlashRLJob, *, step: int, final: bool) -> Path:
    if final and job.spec.checkpointing.final_path is not None:
        return Path(job.spec.checkpointing.final_path)
    base_dir = storage_path_from_uri(job.spec.storage.checkpoints.uriPrefix, purpose="checkpoints")
    base_dir.mkdir(parents=True, exist_ok=True)
    if final:
        return base_dir / "final.pt"
    return base_dir / f"step-{int(step):08d}.pt"


def _on_step_complete(
    *,
    info: dict[str, Any],
    job: FlashRLJob,
    trainer: GRPOTrainer,
    serving: ServingClient,
    status_writer: FlashRLJobStatusWriter,
) -> None:
    step = int(info["step"])
    epoch = int(info["epoch"])
    patch: dict[str, Any] = {
        "progress": {
            "currentEpoch": epoch,
            "currentStep": step,
            "lastCompletedStep": step,
        }
    }
    try:
        serving_status = serving.status().status
    except Exception:
        serving_status = None
    if serving_status is not None and serving_status.active_weight_version is not None:
        active_payload = serving_status.active_weight_version.model_dump(mode="json")
        patch["weightVersion"] = {
            "desired": active_payload,
            "active": active_payload,
        }

    save_every_steps = job.spec.checkpointing.save_every_steps
    if save_every_steps is not None and step > 0 and step % int(save_every_steps) == 0:
        checkpoint_path = _checkpoint_path(job, step=step, final=False)
        trainer.save_checkpoint(str(checkpoint_path))
        patch["checkpoint"] = {
            "latestUri": str(checkpoint_path),
            "lastSavedAt": utc_now_iso(),
        }

    status_writer.patch_status(patch)


def _load_dataset_from_uri(dataset_spec: DatasetSpec) -> list[Prompt]:
    path = storage_path_from_uri(str(dataset_spec.uri), purpose="dataset")
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
