"""Shared run-scoped logger and metrics lifecycle for FlashRL training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from flashrl.framework.checkpointing import RestoredCheckpoint
from flashrl.framework.config import (
    GrpoConfig,
    LoggingConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)
from flashrl.framework.data_models import Prompt
from flashrl.framework.metrics import MetricsSink
from flashrl.framework.run_logger import RunLogger


@dataclass
class TrainRunState:
    """Precomputed metadata shared across one training run."""

    dataset: list[Prompt]
    dataset_size: int
    prompts_per_step: int
    steps_per_epoch: int
    total_planned_steps: int
    run_logger: RunLogger | None = None


def build_train_run_state(
    dataset: list[Prompt],
    *,
    trainer_config: TrainerConfig,
    grpo_config: GrpoConfig,
) -> TrainRunState:
    """Compute the shared per-run dataset/step metadata once."""
    prompts_per_step = trainer_config.batch_size // grpo_config.group_size
    steps_per_epoch = math.ceil(len(dataset) / prompts_per_step) if dataset else 0
    return TrainRunState(
        dataset=dataset,
        dataset_size=len(dataset),
        prompts_per_step=prompts_per_step,
        steps_per_epoch=steps_per_epoch,
        total_planned_steps=steps_per_epoch * trainer_config.max_epochs,
    )


def open_run_logger(
    *,
    logging_config: LoggingConfig,
    model_name: str,
    actor_config: TrainingConfig,
    reference_config: TrainingConfig | None,
    serving_config: ServingConfig,
    trainer_config: TrainerConfig,
    grpo_config: GrpoConfig,
    run_state: TrainRunState,
    actor_device: str,
    reference_device: str | None,
    serving_device: str,
    admin_base_url: str | None,
    bootstrap_console_lines: list[str],
    bootstrap_events: list[dict[str, Any]],
    managed_resume: RestoredCheckpoint | None,
    managed_resume_load_seconds: float,
    resumed_epoch: int = 0,
    resumed_step: int = 0,
    serving_log_dir_setter: Any | None = None,
) -> RunLogger:
    """Open or resume the shared per-run logger for local or platform mode."""
    run_open_kwargs = {
        "dataset_size": run_state.dataset_size,
        "batch_size": trainer_config.batch_size,
        "max_epochs": trainer_config.max_epochs,
        "total_batches": run_state.total_planned_steps,
        "device": actor_config.device or "auto",
        "dtype": actor_config.dtype,
        "cpu_threads": actor_config.num_threads,
        "runtime_shape": "single-device-per-backend",
        "group_size": grpo_config.group_size,
        "clip_ratio": grpo_config.clip_ratio,
        "prompts_per_step": run_state.prompts_per_step,
        "steps_per_epoch": run_state.steps_per_epoch,
        "total_planned_steps": run_state.total_planned_steps,
        "actor_backend": actor_config.backend,
        "actor_device": actor_device,
        "actor_dp_size": actor_config.dp_size,
        "reference_configured": reference_config is not None,
        "reference_backend": (reference_config.backend if reference_config is not None else None),
        "reference_device": reference_device,
        "reference_dp_size": (reference_config.dp_size if reference_config is not None else None),
        "serving_backend": serving_config.backend,
        "serving_device": serving_device,
        "serving_num_replicas": serving_config.num_replicas,
        "admin_base_url": admin_base_url,
        "max_new_tokens": grpo_config.max_new_tokens,
        "include_startup_divider": bool(bootstrap_console_lines),
    }
    if managed_resume is not None:
        run_logger = RunLogger.open_existing_run(
            logging_config,
            model_name=model_name,
            run_id=managed_resume.run_id,
            run_index=managed_resume.run_index,
            run_dir=managed_resume.run_dir,
            restored_state=managed_resume.run_logger_state,
        )
    else:
        run_logger = RunLogger(logging_config, model_name=model_name)
    if callable(serving_log_dir_setter):
        serving_log_dir_setter(run_logger.run_dir)
    run_logger.replay_startup_lines(bootstrap_console_lines)
    if managed_resume is not None:
        run_logger.resume_run(
            checkpoint_path=str(managed_resume.checkpoint_path),
            **run_open_kwargs,
        )
    else:
        run_logger.start_run(**run_open_kwargs)
    for event in bootstrap_events:
        run_logger.log_model_load(
            event["component"],
            event["status"],
            event["metadata"],
        )
    if managed_resume is not None:
        run_logger.log_checkpoint(
            "load",
            str(managed_resume.checkpoint_path),
            epoch=int(resumed_epoch),
            step=int(resumed_step),
            duration_seconds=managed_resume_load_seconds,
            trigger="resume",
        )
    run_state.run_logger = run_logger
    return run_logger


def start_run_metrics(metrics_sink: MetricsSink | None, *, run_logger: RunLogger) -> None:
    """Start the shared metrics sink for one training run."""
    if metrics_sink is None:
        return
    metrics_sink.start_run(run_dir=run_logger.run_dir, run_id=run_logger.run_id)


def finish_run_observers(
    *,
    run_logger: RunLogger | None,
    metrics_sink: MetricsSink | None,
    status: str,
    total_steps: int,
    lifecycle_totals: dict[str, float],
) -> None:
    """Finalize the shared run logger and metrics sink."""
    if metrics_sink is not None:
        metrics_sink.finish_run()
    if run_logger is not None:
        run_logger.finish_run(
            status=status,
            total_steps=total_steps,
            lifecycle_totals=lifecycle_totals,
        )
        run_logger.close()
