"""Platform shim for the controller pod."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock, Thread
import time
from typing import Any

from fastapi import FastAPI

from flashrl.framework.admin import utc_now_iso
from flashrl.framework import runtime_support
from flashrl.framework.checkpointing import CheckpointManager, RestoredCheckpoint
from flashrl.framework.distributed.http_common import install_common_routes
from flashrl.framework.distributed.models import ComponentStatus
from flashrl.framework.distributed import (
    LearnerClient,
    RewardClient,
    RolloutClient,
    ServingClient,
)
from flashrl.framework.data_models import Prompt
from flashrl.framework.memory import capture_memory_snapshot
from flashrl.framework.metrics import build_metrics_sink
from flashrl.framework.train_runtime import (
    build_train_run_state,
    finish_run_observers,
    open_run_logger as open_shared_run_logger,
    start_run_metrics,
)
from flashrl.framework.trainer.grpo.trainer import GRPOTrainer
from flashrl.platform.k8s.job import (
    DatasetSpec,
    FlashRLJob,
    GROUP,
    PLURAL,
    VERSION,
    append_job_event,
)
from flashrl.platform.runtime.platform_pod_logging import PlatformPodLogger
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import (
    component_log_metadata,
    job_uid_for,
    load_mounted_job,
    resolve_job_log_root,
    service_url_for,
    storage_path_from_uri,
)


class FlashRLJobStatusWriter:
    """Best-effort CRD status writer used by the controller pod."""

    def __init__(
        self,
        *,
        job_name: str,
        namespace: str,
        log_root: str | Path | None = None,
        event_limit: int = 100,
    ) -> None:
        self._job_name = job_name
        self._namespace = namespace
        self._log_root = Path(log_root) if log_root is not None else None
        self._event_limit = int(event_limit)
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
        self.write_status_snapshot()

    def append_event(
        self,
        *,
        event: str,
        message: str,
        component: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        job = self.get_job()
        if job is None:
            return
        events = append_job_event(
            job.status.events,
            timestamp=utc_now_iso(),
            event=event,
            message=message,
            component=component,
            metadata=metadata,
            limit=self._event_limit,
        )
        self.patch_status({"events": events})

    def write_status_snapshot(self) -> None:
        if self._log_root is None:
            return
        job = self.get_job()
        if job is None:
            return
        snapshot_dir = self._log_root / "_status"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / "status-snapshots.jsonl"
        payload = {
            "timestamp": utc_now_iso(),
            "status": job.status.model_dump(mode="json"),
        }
        with snapshot_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

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


def _status_writer_append_event(
    status_writer: Any,
    *,
    event: str,
    message: str,
    component: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    append = getattr(status_writer, "append_event", None)
    if callable(append):
        append(
            event=event,
            message=message,
            component=component,
            metadata=metadata,
        )


def _attach_run_logger(trainer: Any, run_logger: Any | None) -> None:
    attach = getattr(trainer, "attach_run_logger", None)
    if callable(attach):
        attach(run_logger)
        return
    try:
        setattr(trainer, "run_logger", run_logger)
    except Exception:
        pass


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

    def __init__(self, *, shim: "PlatformShimController") -> None:
        self._shim = shim
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
                    **self._shim.status_metadata(),
                },
            )

    def _run(self) -> None:
        try:
            self._shim.run_training_loop()
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


class PlatformShimController(PlatformShim):
    """Load the mounted job, then run controller training behind a small HTTP surface."""

    def create_app(self) -> FastAPI:
        job = load_mounted_job(self._job_path)
        self._pod_logger = PlatformPodLogger(job=job, component="controller")
        self._pod_logger.configure_python_logging()
        self._pod_logger.emit("shim_startup", message="Bootstrapping controller pod.")
        state = _ControllerServiceState(shim=self)
        app = FastAPI(title="FlashRL Controller")
        install_common_routes(
            app,
            status_getter=state.status,
            kind="ControllerService",
            name="controller",
            event_logger=self._pod_logger.emit,
        )

        @app.on_event("startup")
        def startup() -> None:
            state.start()

        return app

    def run_training_loop(self) -> None:
        run_controller(self._job_path, pod_logger=self._pod_logger)

    def status_metadata(self) -> dict[str, Any]:
        job = load_mounted_job(self._job_path)
        return {
            **component_log_metadata(job, "controller"),
            "memory": capture_memory_snapshot(None),
        }


def run_controller(
    job_path: str | Path | None = None,
    *,
    pod_logger: PlatformPodLogger | None = None,
) -> None:
    """Run the GRPO controller loop for one platform job."""
    static_job = load_mounted_job(job_path)
    namespace = str(
        os.environ.get("FLASHRL_NAMESPACE")
        or static_job.metadata.get("namespace")
        or "default"
    )
    os.environ.setdefault("FLASHRL_JOB_NAME", static_job.name)
    os.environ.setdefault("FLASHRL_NAMESPACE", namespace)
    resolved_log_root = resolve_job_log_root(static_job)
    if pod_logger is not None:
        pod_logger.emit(
            "controller_log_root_resolved",
            message=f"Resolved platform job log root at {resolved_log_root}.",
            metadata={"jobLogRoot": str(resolved_log_root)},
        )
    try:
        status_writer = FlashRLJobStatusWriter(
            job_name=static_job.name,
            namespace=namespace,
            log_root=resolved_log_root,
            event_limit=int(static_job.spec.observability.jobEventHistoryLimit),
        )
    except TypeError:
        status_writer = FlashRLJobStatusWriter(job_name=static_job.name, namespace=namespace)
    live_job = status_writer.get_job()
    job = static_job
    if live_job is not None:
        job.status = live_job.status.model_copy(deep=True)
        for metadata_key in ("uid", "resourceVersion", "generation", "deletionTimestamp", "finalizers"):
            if metadata_key in live_job.metadata:
                job.metadata[metadata_key] = live_job.metadata[metadata_key]
    os.environ.setdefault("FLASHRL_JOB_UID", job_uid_for(job))
    os.environ.setdefault("FLASHRL_JOB_LOG_ROOT", str(resolved_log_root))
    os.environ.setdefault("FLASHRL_COMPONENT_LOG_DIR", str(resolved_log_root / "_pods" / "controller"))

    rollout = RolloutClient(service_url_for("rollout"))
    reward = RewardClient(service_url_for("reward"))
    learner = LearnerClient(service_url_for("learner"))
    serving = ServingClient(service_url_for("serving"))

    metrics_config = job.spec.framework.metrics.model_copy(deep=True)
    if not job.spec.observability.metricsEnabled:
        metrics_config.enabled = False
    metrics_sink = build_metrics_sink(
        metrics_config,
        model_name=job.spec.framework.actor.model_name,
        runtime="platform_controller",
    )

    logging_config = job.spec.framework.logging.model_copy(deep=True)
    logging_config.log_dir = resolved_log_root
    checkpoint_manager = CheckpointManager(job.spec.checkpointing.model_copy(deep=True))
    lifecycle_totals: dict[str, float] = {}
    restored_run_lifecycle_totals: dict[str, float] = {}
    managed_resume: RestoredCheckpoint | None = None
    managed_resume_load_seconds = 0.0
    bootstrap_console_lines = [
        "FlashRL platform controller",
        f"  controller namespace={namespace} job={job.name}",
        f"  services rollout={service_url_for('rollout')}",
        f"  services reward={service_url_for('reward')}",
        f"  services learner={service_url_for('learner')}",
        f"  services serving={service_url_for('serving')}",
    ]
    bootstrap_events: list[dict[str, Any]] = []

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
            checkpoint_manager=checkpoint_manager,
            lifecycle_totals=lifecycle_totals,
            serving=serving,
            status_writer=status_writer,
            pod_logger=pod_logger,
        ),
        metrics_sink=metrics_sink,
    )
    lifecycle_totals["startup_total_seconds"] = 0.0
    _status_writer_append_event(
        status_writer,
        event="controller_starting",
        message="Controller pod is starting the platform training loop.",
        component="controller",
        metadata={"jobLogRoot": str(resolved_log_root)},
    )

    resume_target = _resume_target(job)
    if resume_target is not None:
        started_at = time.perf_counter()
        checkpoint_metadata = None
        load_checkpoint_with_metadata = getattr(trainer, "load_checkpoint_with_metadata", None)
        if callable(load_checkpoint_with_metadata):
            controller_state, checkpoint_metadata = load_checkpoint_with_metadata(resume_target)
            managed_resume = checkpoint_manager.build_restored_checkpoint(
                checkpoint_path=Path(resume_target),
                checkpoint_metadata=checkpoint_metadata,
            )
        else:
            trainer.load_checkpoint(resume_target)
            controller_state = {
                "epoch": getattr(trainer, "current_epoch", 0),
                "total_steps": getattr(trainer, "total_steps", 0),
            }
        managed_resume_load_seconds = time.perf_counter() - started_at
        restored_run_lifecycle_totals = (
            dict(managed_resume.lifecycle_totals) if managed_resume is not None else {}
        )
        lifecycle_totals["checkpoint_load_seconds"] = managed_resume_load_seconds
        _status_writer_append_event(
            status_writer,
            event="checkpoint_loaded",
            message=f"Resumed controller state from {resume_target}.",
            component="controller",
            metadata={"checkpointPath": resume_target, "controllerState": controller_state},
        )

    dataset_started_at = time.perf_counter()
    _status_writer_append_event(
        status_writer,
        event="dataset_load_started",
        message="Loading controller dataset.",
        component="controller",
    )
    dataset = load_controller_dataset(job)
    dataset_load_seconds = time.perf_counter() - dataset_started_at
    lifecycle_totals["dataset_load_seconds"] = dataset_load_seconds
    _status_writer_append_event(
        status_writer,
        event="dataset_load_finished",
        message=f"Loaded controller dataset with {len(dataset)} prompts.",
        component="controller",
        metadata={"datasetSize": len(dataset), "durationSeconds": dataset_load_seconds},
    )

    run_state = build_train_run_state(
        dataset,
        trainer_config=job.spec.framework.trainer,
        grpo_config=job.spec.framework.grpo,
    )
    lifecycle_totals = _merge_float_mappings(
        restored_run_lifecycle_totals,
        lifecycle_totals,
    )
    run_logger = open_shared_run_logger(
        logging_config=logging_config,
        model_name=job.spec.framework.actor.model_name,
        actor_config=job.spec.framework.actor,
        reference_config=job.spec.framework.reference,
        serving_config=job.spec.framework.serving,
        trainer_config=job.spec.framework.trainer,
        grpo_config=job.spec.framework.grpo,
        run_state=run_state,
        actor_device="remote",
        reference_device=("remote" if job.spec.framework.reference is not None else None),
        serving_device="remote",
        admin_base_url=None,
        bootstrap_console_lines=bootstrap_console_lines,
        bootstrap_events=bootstrap_events,
        managed_resume=managed_resume,
        managed_resume_load_seconds=managed_resume_load_seconds,
        resumed_epoch=trainer.current_epoch + 1,
        resumed_step=trainer.total_steps,
    )
    _attach_run_logger(trainer, run_logger)
    start_run_metrics(metrics_sink, run_logger=run_logger)
    status_writer.patch_status(
        {
            "startedAt": job.status.startedAt or utc_now_iso(),
            "progress": {
                "currentEpoch": trainer.current_epoch + (1 if trainer.total_steps > 0 else 0),
                "currentStep": trainer.total_steps,
                "lastCompletedStep": trainer.total_steps,
            },
            "logRoot": str(resolved_log_root),
            "activeControllerRunDir": str(run_logger.run_dir),
            "activeControllerRunId": run_logger.run_id,
        }
    )
    _status_writer_append_event(
        status_writer,
        event="controller_run_started",
        message=f"Controller run {run_logger.run_id} started.",
        component="controller",
        metadata={"runDir": str(run_logger.run_dir), "runId": run_logger.run_id},
    )
    if pod_logger is not None:
        pod_logger.emit(
            "controller_run_started",
            message=f"Controller run {run_logger.run_id} started.",
            metadata={"runDir": str(run_logger.run_dir), "jobLogRoot": str(resolved_log_root)},
        )

    training_loop_started_at = time.perf_counter()
    final_status = "completed"
    try:
        trainer.train(dataset)
        lifecycle_totals["training_loop_seconds"] = time.perf_counter() - training_loop_started_at
        _save_final_checkpoint_if_enabled(
            job=job,
            trainer=trainer,
            checkpoint_manager=checkpoint_manager,
            lifecycle_totals=lifecycle_totals,
            status_writer=status_writer,
        )
        _status_writer_append_event(
            status_writer,
            event="controller_run_finished",
            message=f"Controller run {run_logger.run_id} completed successfully.",
            component="controller",
        )
        status_writer.patch_status({"finishedAt": utc_now_iso()})
    except Exception as exc:
        final_status = "failed"
        lifecycle_totals["training_loop_seconds"] = time.perf_counter() - training_loop_started_at
        if not bool(getattr(exc, "_flashrl_logged", False)):
            context: dict[str, Any] = {
                "stage": "platform_controller",
                "step": trainer.total_steps,
                "epoch": trainer.current_epoch + 1,
            }
            learner_stage = getattr(exc, "stage_name", None)
            if learner_stage is not None:
                context["learner_stage"] = str(learner_stage)
            memory_snapshot = getattr(exc, "memory_snapshot", None)
            if isinstance(memory_snapshot, dict):
                context["memory"] = memory_snapshot
            reason_tags = getattr(exc, "reason_tags", None)
            if isinstance(reason_tags, list) and reason_tags:
                context["memory_reason_tags"] = [str(tag) for tag in reason_tags]
            run_logger.log_exception(exc, context=context)
        if pod_logger is not None:
            pod_logger.emit_exception(exc, stage="run_controller")
        _status_writer_append_event(
            status_writer,
            event="controller_run_failed",
            message=f"{type(exc).__name__}: {exc}",
            component="controller",
        )
        status_writer.patch_status({"finishedAt": utc_now_iso(), "lastError": str(exc)})
        raise
    finally:
        _attach_run_logger(trainer, None)
        finish_run_observers(
            run_logger=run_logger,
            metrics_sink=metrics_sink,
            status=final_status,
            total_steps=trainer.total_steps,
            lifecycle_totals=lifecycle_totals,
        )


def _resume_target(job: FlashRLJob) -> str | None:
    resume_from = job.spec.checkpointing.resume_from
    if resume_from is None:
        return None
    if resume_from == "latest":
        return job.status.checkpoint.latestUri
    return str(resume_from)


def _merge_float_mappings(*mappings: dict[str, Any]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            try:
                merged[key] = merged.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
    return merged


def _build_checkpoint_metadata(
    *,
    trainer: GRPOTrainer,
    run_dir: Path,
    run_id: str,
    run_index: int,
    run_logger_state: dict[str, Any],
    lifecycle_totals: dict[str, float],
) -> dict[str, Any]:
    return {
        "run": {
            "run_id": run_id,
            "run_index": int(run_index),
            "run_dir": str(run_dir),
        },
        "run_logger_state": dict(run_logger_state),
        "lifecycle_totals": dict(lifecycle_totals),
        "controller_state": {
            "epoch": trainer.current_epoch,
            "total_steps": trainer.total_steps,
        },
    }


def _save_trainer_checkpoint(
    trainer: GRPOTrainer,
    path: str,
    *,
    checkpoint_metadata: dict[str, Any],
) -> None:
    try:
        trainer.save_checkpoint(path, checkpoint_metadata=checkpoint_metadata)
    except TypeError:
        trainer.save_checkpoint(path)


def _save_final_checkpoint_if_enabled(
    *,
    job: FlashRLJob,
    trainer: GRPOTrainer,
    checkpoint_manager: CheckpointManager,
    lifecycle_totals: dict[str, float],
    status_writer: FlashRLJobStatusWriter,
) -> None:
    if not job.spec.checkpointing.save_on_run_end or trainer.run_logger is None:
        return
    checkpoint_path = (
        Path(job.spec.checkpointing.final_path)
        if job.spec.checkpointing.final_path is not None
        else checkpoint_manager.final_checkpoint_path(run_dir=trainer.run_logger.run_dir)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    _save_trainer_checkpoint(
        trainer,
        str(checkpoint_path),
        checkpoint_metadata=_build_checkpoint_metadata(
            trainer=trainer,
            run_dir=trainer.run_logger.run_dir,
            run_id=trainer.run_logger.run_id,
            run_index=trainer.run_logger.run_index,
            run_logger_state=trainer.run_logger.export_state(),
            lifecycle_totals=lifecycle_totals,
        ),
    )
    duration_seconds = time.perf_counter() - started_at
    lifecycle_totals["checkpoint_save_seconds"] = (
        lifecycle_totals.get("checkpoint_save_seconds", 0.0) + duration_seconds
    )
    trainer.run_logger.log_checkpoint(
        "save",
        str(checkpoint_path),
        epoch=trainer.current_epoch + 1,
        step=trainer.total_steps,
        duration_seconds=duration_seconds,
        trigger="final",
    )
    checkpoint_manager.write_latest_manifest(
        run_dir=trainer.run_logger.run_dir,
        checkpoint_path=checkpoint_path,
        epoch=trainer.current_epoch + 1,
        step=trainer.total_steps,
        trigger="final",
        run_id=trainer.run_logger.run_id,
        run_index=trainer.run_logger.run_index,
    )
    status_writer.patch_status(
        {
            "checkpoint": {
                "latestUri": str(checkpoint_path),
                "lastSavedAt": utc_now_iso(),
            }
        }
    )
    _status_writer_append_event(
        status_writer,
        event="checkpoint_saved",
        message=f"Saved final checkpoint to {checkpoint_path}.",
        component="controller",
        metadata={"trigger": "final", "durationSeconds": duration_seconds},
    )


def _on_step_complete(
    *,
    info: dict[str, Any],
    job: FlashRLJob,
    trainer: GRPOTrainer,
    checkpoint_manager: CheckpointManager,
    lifecycle_totals: dict[str, float],
    serving: ServingClient,
    status_writer: FlashRLJobStatusWriter,
    pod_logger: PlatformPodLogger | None,
) -> None:
    step = int(info["step"])
    epoch = int(info["epoch"])
    patch: dict[str, Any] = {
        "progress": {
            "currentEpoch": epoch,
            "currentStep": step,
            "lastCompletedStep": step,
        },
        "activeControllerRunDir": (
            str(trainer.run_logger.run_dir) if trainer.run_logger is not None else None
        ),
        "activeControllerRunId": (
            trainer.run_logger.run_id if trainer.run_logger is not None else None
        ),
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
    if (
        save_every_steps is not None
        and step > 0
        and step % int(save_every_steps) == 0
        and trainer.run_logger is not None
    ):
        checkpoint_path = checkpoint_manager.interval_checkpoint_path(
            run_dir=trainer.run_logger.run_dir,
            step=step,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        _save_trainer_checkpoint(
            trainer,
            str(checkpoint_path),
            checkpoint_metadata=_build_checkpoint_metadata(
                trainer=trainer,
                run_dir=trainer.run_logger.run_dir,
                run_id=trainer.run_logger.run_id,
                run_index=trainer.run_logger.run_index,
                run_logger_state=trainer.run_logger.export_state(),
                lifecycle_totals=lifecycle_totals,
            ),
        )
        duration_seconds = time.perf_counter() - started_at
        lifecycle_totals["checkpoint_save_seconds"] = (
            lifecycle_totals.get("checkpoint_save_seconds", 0.0) + duration_seconds
        )
        trainer.run_logger.log_checkpoint(
            "save",
            str(checkpoint_path),
            epoch=trainer.current_epoch + 1,
            step=trainer.total_steps,
            duration_seconds=duration_seconds,
            trigger="interval",
        )
        checkpoint_manager.write_latest_manifest(
            run_dir=trainer.run_logger.run_dir,
            checkpoint_path=checkpoint_path,
            epoch=trainer.current_epoch + 1,
            step=trainer.total_steps,
            trigger="interval",
            run_id=trainer.run_logger.run_id,
            run_index=trainer.run_logger.run_index,
        )
        patch["checkpoint"] = {
            "latestUri": str(checkpoint_path),
            "lastSavedAt": utc_now_iso(),
        }

    status_writer.patch_status(patch)
    _status_writer_append_event(
        status_writer,
        event="step_completed",
        message=f"Completed controller step {step} in epoch {epoch}.",
        component="controller",
        metadata={"step": step, "epoch": epoch},
    )
    if pod_logger is not None:
        pod_logger.emit(
            "step_completed",
            message=f"Completed controller step {step} in epoch {epoch}.",
            metadata={"step": step, "epoch": epoch},
        )


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
