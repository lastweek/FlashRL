"""Run-scoped logging and terminal UX for local FlashRL training."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any
from uuid import uuid4

from flashrl.framework.config import LoggingConfig

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when rich is unavailable.
    Console = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    RICH_AVAILABLE = False


LEVEL_PRIORITIES = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

HOT_PATH_TOTAL_KEYS = [
    "rollout_seconds",
    "reward_seconds",
    "advantage_seconds",
    "tokenize_full_seconds",
    "tokenize_prompt_seconds",
    "actor_forward_seconds",
    "reference_forward_seconds",
    "loss_assembly_seconds",
    "backward_seconds",
    "optimizer_step_seconds",
    "weight_sync_seconds",
    "logging_seconds",
]

LIFECYCLE_TOTAL_KEYS = [
    "startup_total_seconds",
    "training_loop_seconds",
    "checkpoint_save_seconds",
    "checkpoint_load_seconds",
]

STARTUP_COMPONENT_KEYS = [
    "startup_training_backend_seconds",
    "startup_serving_backend_seconds",
    "startup_reference_model_seconds",
]

PHASE_GROUP_SPECS = [
    ("rollout", ("rollout_seconds",)),
    ("reward", ("reward_seconds",)),
    (
        "calculate_loss",
        (
            "advantage_seconds",
            "tokenize_full_seconds",
            "tokenize_prompt_seconds",
            "actor_forward_seconds",
            "reference_forward_seconds",
            "loss_assembly_seconds",
        ),
    ),
    (
        "train",
        (
            "backward_seconds",
            "optimizer_step_seconds",
            "weight_sync_seconds",
        ),
    ),
]

LOSS_PATH_DETAIL_SPECS = [
    ("advantage", "advantage_seconds"),
    ("tokenize_full", "tokenize_full_seconds"),
    ("tokenize_prompt", "tokenize_prompt_seconds"),
    ("actor_forward", "actor_forward_seconds"),
    ("reference_forward", "reference_forward_seconds"),
    ("loss_assembly", "loss_assembly_seconds"),
]

TRAIN_PATH_DETAIL_SPECS = [
    ("backward", "backward_seconds"),
    ("optimizer", "optimizer_step_seconds"),
    ("sync", "weight_sync_seconds"),
]

STEP_PHASE_FIELD_ORDERS = {
    "rollout": [
        "prompt_tokens_mean",
        "prompt_tokens_max",
        "response_tokens_mean",
        "response_tokens_max",
    ],
    "reward": [
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
    ],
    "advantage": [
        "advantage_mean",
        "advantage_std",
        "advantage_min",
        "advantage_max",
    ],
    "tokenize_full": [
        "full_tokens_mean",
        "full_tokens_max",
    ],
    "tokenize_prompt": [
        "prompt_tokens_mean",
        "prompt_tokens_max",
    ],
    "actor_forward": [
        "full_tokens_total",
    ],
    "reference_forward": [
        "full_tokens_total",
    ],
    "loss_assembly": [
        "loss",
        "policy_loss",
        "kl_divergence",
        "response_tokens_total",
    ],
    "backward": [
        "loss",
    ],
    "optimizer": [
        "learning_rate",
    ],
    "sync": [
        "step_duration_seconds",
        "tokens_per_second",
    ],
}

STEP_DONE_FIELD_ORDER = [
    "loss",
    "policy_loss",
    "kl_divergence",
    "reward_mean",
    "response_tokens_total",
    "tokens_per_second",
    "step_duration_seconds",
]


class RunLogger:
    """Run-scoped logger with console UX and persistent event files."""

    def __init__(
        self,
        config: LoggingConfig,
        model_name: str,
    ) -> None:
        """Initialize a logger for a single training run."""
        self.config = config
        self.model_name = model_name
        self.run_id = self._build_run_id(model_name)
        self._level_threshold = LEVEL_PRIORITIES.get(config.level.upper(), LEVEL_PRIORITIES["INFO"])
        self._started_at = time.perf_counter()
        self._total_epochs = 0
        self._last_stage = "initializing"
        self._finished = False
        self._step_durations: list[float] = []
        self._hot_path_totals: defaultdict[str, float] = defaultdict(float)
        self._lifecycle_totals: defaultdict[str, float] = defaultdict(float)
        self._slowest_step_seconds = 0.0
        self._slowest_step_phase = "n/a"

        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._events_handle = (
            open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
            if config.file
            else None
        )
        self._transcript_handle = (
            open(self.run_dir / "console.log", "a", encoding="utf-8")
            if config.file
            else None
        )

        if config.console and RICH_AVAILABLE:
            self._console: Console | None = Console()
        else:
            self._console = None

        if self._transcript_handle is not None and RICH_AVAILABLE:
            self._transcript_console: Console | None = Console(
                file=self._transcript_handle,
                color_system=None,
                force_terminal=False,
                width=160,
            )
        else:
            self._transcript_console = None

        self._progress = None
        self._progress_task_id: int | None = None

    def close(self) -> None:
        """Close open file handles."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

        if self._events_handle is not None:
            self._events_handle.close()
            self._events_handle = None

        if self._transcript_handle is not None:
            self._transcript_handle.close()
            self._transcript_handle = None

        self._transcript_console = None

    def start_run(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        max_epochs: int,
        total_batches: int,
        device: str,
        dtype: str,
        cpu_threads: int,
        runtime_shape: str,
        reference_enabled: bool,
        reference_device: str,
    ) -> None:
        """Start a training run and initialize progress tracking."""
        self._total_epochs = max_epochs
        self._started_at = time.perf_counter()
        self._start_progress(total_batches)

        payload = {
            "model_name": self.model_name,
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "total_batches": total_batches,
            "device": device,
            "dtype": dtype,
            "cpu_threads": cpu_threads,
            "runtime_shape": runtime_shape,
            "reference_enabled": reference_enabled,
            "reference_device": reference_device,
            "run_dir": str(self.run_dir),
        }
        self._emit("run_started", payload)

    def log_model_load(
        self,
        component: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log model/backend loading progress."""
        payload = {
            "component": component,
            "status": status,
        }
        if details:
            payload.update(details)
        self._emit("model_load", payload)

    def log_epoch_start(
        self,
        epoch: int,
        total_epochs: int,
        batches_in_epoch: int,
    ) -> None:
        """Log the start of an epoch."""
        self._update_progress(epoch=epoch, stage="starting")
        self._emit(
            "epoch_started",
            {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batches_in_epoch": batches_in_epoch,
            },
        )

    def update_stage(
        self,
        *,
        epoch: int,
        stage: str,
        batch_index: int | None = None,
    ) -> None:
        """Update the progress bar stage without emitting a separate event."""
        self._update_progress(epoch=epoch, stage=stage, batch_index=batch_index)

    def log_step_phase(self, payload: dict[str, Any]) -> None:
        """Log a phase-completion event for a training step."""
        if not self.should_log_step(int(payload["step"])):
            return
        self._emit("step_phase", payload)

    def log_step_done(self, metrics: dict[str, Any]) -> None:
        """Log the completed step summary and advance progress state."""
        self._update_progress(
            epoch=int(metrics["epoch"]),
            stage="complete",
            batch_index=int(metrics["batch_index"]),
            advance=1,
        )

        self._record_step_metrics(metrics)
        if not self.should_log_step(int(metrics["step"])):
            return

        self._emit("step_done", metrics)

    def log_batch_metrics(self, metrics: dict[str, Any]) -> None:
        """Backward-compatible alias for logging a completed step."""
        self.log_step_done(metrics)

    def record_logging_overhead(self, seconds: float) -> None:
        """Record logging time that happened outside the step event itself."""
        self._hot_path_totals["logging_seconds"] += float(seconds)

    def log_sample_preview(
        self,
        *,
        step: int,
        prompt: str,
        response: str,
        reward: float,
    ) -> None:
        """Log a truncated prompt/response sample preview."""
        payload = {
            "step": step,
            "prompt": self._truncate(prompt),
            "response": self._truncate(response),
            "reward": reward,
        }
        self._emit("sample_preview", payload)

    def log_epoch_summary(
        self,
        epoch: int,
        metrics: dict[str, Any],
        duration_seconds: float,
    ) -> None:
        """Log epoch-level summary metrics."""
        payload = {
            "epoch": epoch,
            "duration_seconds": duration_seconds,
            **metrics,
        }
        self._emit("epoch_summary", payload)

    def log_checkpoint(
        self,
        action: str,
        path: str,
        *,
        epoch: int | None = None,
        step: int | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Log checkpoint save/load activity."""
        payload = {
            "action": action,
            "path": path,
        }
        if epoch is not None:
            payload["epoch"] = epoch
        if step is not None:
            payload["step"] = step
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds
        self._emit("checkpoint", payload)

    def log_exception(
        self,
        exc: BaseException,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log an exception with traceback and extra context."""
        payload = {
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(exc.__class__, exc, exc.__traceback__)
            ).rstrip(),
        }
        if context:
            payload["context"] = context
        self._emit("exception", payload, level="ERROR")

    def finish_run(
        self,
        *,
        status: str,
        total_steps: int,
        duration_seconds: float | None = None,
        lifecycle_totals: dict[str, float] | None = None,
    ) -> None:
        """Log the final run status and stop the progress bar."""
        if self._progress is not None:
            self._update_progress(epoch=self._total_epochs, stage=status)
            self._progress.stop()
            self._progress = None

        if self._finished:
            return

        elapsed = duration_seconds if duration_seconds is not None else (
            time.perf_counter() - self._started_at
        )
        if lifecycle_totals:
            for key, value in lifecycle_totals.items():
                self._lifecycle_totals[key] += float(value)

        lifecycle_payload = {
            key: float(self._lifecycle_totals.get(key, 0.0))
            for key in LIFECYCLE_TOTAL_KEYS
            if float(self._lifecycle_totals.get(key, 0.0)) > 0
        }
        startup_components = {
            key: float(self._lifecycle_totals.get(key, 0.0))
            for key in STARTUP_COMPONENT_KEYS
            if float(self._lifecycle_totals.get(key, 0.0)) > 0
        }
        hot_path_payload = {
            key: float(self._hot_path_totals.get(key, 0.0))
            for key in HOT_PATH_TOTAL_KEYS
            if float(self._hot_path_totals.get(key, 0.0)) > 0
        }
        phase_group_totals = self._build_phase_group_totals(hot_path_payload)
        payload = {
            "status": status,
            "total_steps": total_steps,
            "duration_seconds": elapsed,
            "run_dir": str(self.run_dir),
            "lifecycle_totals": lifecycle_payload,
            "lifecycle_percentages": self._percentages(lifecycle_payload),
            "startup_components": startup_components,
            "hot_path_totals": hot_path_payload,
            "hot_path_percentages": self._percentages(hot_path_payload),
            "phase_group_totals": phase_group_totals,
            "phase_group_percentages": self._percentages(phase_group_totals),
            "avg_step_seconds": (
                sum(self._step_durations) / len(self._step_durations)
                if self._step_durations
                else 0.0
            ),
            "slowest_step_seconds": self._slowest_step_seconds,
            "slowest_phase": self._slowest_step_phase,
            "no_training_steps_completed": len(self._step_durations) == 0,
        }
        self._emit("run_finished", payload)
        self._finished = True

    def should_log_step(self, step: int) -> bool:
        """Return whether step metrics should be emitted."""
        interval = self.config.log_every_steps
        if interval <= 0:
            return step == 1
        return step == 1 or step % interval == 0

    def should_log_sample(self, step: int) -> bool:
        """Return whether a sample preview should be emitted."""
        interval = self.config.sample_every_steps
        if interval <= 0:
            return step == 1
        return step == 1 or step % interval == 0

    def _record_step_metrics(self, metrics: dict[str, Any]) -> None:
        self._step_durations.append(float(metrics["step_duration_seconds"]))
        if float(metrics["step_duration_seconds"]) >= self._slowest_step_seconds:
            self._slowest_step_seconds = float(metrics["step_duration_seconds"])
            self._slowest_step_phase = str(metrics["dominant_phase"])

        timings = metrics.get("timings", {})
        for key in HOT_PATH_TOTAL_KEYS:
            if key == "logging_seconds":
                continue
            self._hot_path_totals[key] += float(timings.get(key, 0.0))

    def _build_run_dir(self) -> Path:
        base_dir = Path(self.config.log_dir).expanduser()
        return base_dir / self.run_id

    def _build_run_id(self, model_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-").lower()
        slug = slug[:48] or "model"
        return f"{timestamp}-{slug}-{uuid4().hex[:8]}"

    def _start_progress(self, total_batches: int) -> None:
        if (
            not self.config.console
            or not self.config.rich_progress
            or not RICH_AVAILABLE
            or total_batches <= 0
        ):
            return

        assert self._console is not None
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.fields[label]}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("epoch {task.fields[epoch]}"),
            TextColumn("stage {task.fields[stage]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._progress_task_id = self._progress.add_task(
            "training",
            label="training",
            total=total_batches,
            epoch=f"0/{self._total_epochs}",
            stage=self._last_stage,
        )

    def _update_progress(
        self,
        *,
        epoch: int,
        stage: str,
        batch_index: int | None = None,
        advance: int = 0,
    ) -> None:
        self._last_stage = stage
        if self._progress is None or self._progress_task_id is None:
            return

        self._progress.update(
            self._progress_task_id,
            advance=advance,
            epoch=f"{epoch}/{self._total_epochs}",
            stage=stage,
            label=f"training batch {batch_index}" if batch_index is not None else "training",
        )

    def _emit(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        level: str = "INFO",
    ) -> None:
        level = level.upper()
        if LEVEL_PRIORITIES.get(level, LEVEL_PRIORITIES["INFO"]) < self._level_threshold:
            return

        serialized_payload = self._serialize(payload)
        event_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "event": event,
            "level": level,
            "payload": serialized_payload,
        }

        if self._events_handle is not None:
            self._events_handle.write(json.dumps(event_record, ensure_ascii=True) + "\n")
            self._events_handle.flush()

        lines = self._render_lines(event, serialized_payload)
        self._write_lines(lines, level=level)

    def _render_lines(self, event: str, payload: dict[str, Any]) -> list[str]:
        if event == "run_started":
            reference_state = "enabled" if payload["reference_enabled"] else "disabled"
            return [
                f"FlashRL training run {self.run_id}",
                "  "
                f"model={payload['model_name']} device={payload['device']} "
                f"dtype={payload['dtype']} cpu_threads={payload['cpu_threads']}",
                "  "
                f"dataset={payload['dataset_size']} batch_size={payload['batch_size']} "
                f"epochs={payload['max_epochs']} total_batches={payload['total_batches']}",
                "  "
                f"runtime={payload['runtime_shape']} reference={reference_state} "
                f"reference_device={payload['reference_device']}",
                f"  logs={payload['run_dir']}",
            ]

        if event == "model_load":
            status = "loading" if payload["status"] == "started" else "loaded"
            details = []
            if "device" in payload:
                details.append(f"device={payload['device']}")
            if "cpu_threads" in payload:
                details.append(f"cpu_threads={payload['cpu_threads']}")
            if "duration_seconds" in payload:
                details.append(f"latency={self._format_duration(payload['duration_seconds'])}")
            suffix = f" ({', '.join(details)})" if details else ""
            return [f"{status} {payload['component']}{suffix}"]

        if event == "epoch_started":
            return [
                f"epoch {payload['epoch']}/{payload['total_epochs']} "
                f"(batches={payload['batches_in_epoch']})"
            ]

        if event == "step_phase":
            return [self._render_step_phase_line(payload)]

        if event in {"step_done", "step_metrics"}:
            return [self._render_step_done_line(payload)]

        if event == "sample_preview":
            return [
                f"sample step {payload['step']} reward={payload['reward']:.4f}",
                f"  prompt: {payload['prompt']}",
                f"  response: {payload['response']}",
            ]

        if event == "epoch_summary":
            lines = [
                f"epoch {payload['epoch']} summary "
                f"loss={payload['loss']:.4f} reward={payload['reward_mean']:.4f} "
                f"kl={payload['kl_divergence']:.4f} tok/s={payload['tokens_per_second']:.1f} "
                f"duration={self._format_duration(payload['duration_seconds'])}"
            ]
            if payload.get("no_training_steps_completed"):
                lines.append("  no training steps completed in this epoch")
                return lines

            lines.append(
                "  perf "
                f"avg_step={self._format_duration(payload['avg_step_seconds'])} "
                f"slowest_step={self._format_duration(payload['slowest_step_seconds'])} "
                f"slowest_phase={payload['slowest_phase']}"
            )
            lines.append(
                self._format_named_breakdown(
                    "  phases ",
                    payload["phase_group_totals"],
                    percentages=payload["phase_group_percentages"],
                    order=[name for name, _ in PHASE_GROUP_SPECS],
                )
            )
            return lines

        if event == "checkpoint":
            location = f" epoch={payload['epoch']}" if "epoch" in payload else ""
            if "step" in payload:
                location += f" step={payload['step']}"
            duration = ""
            if "duration_seconds" in payload:
                duration = f" latency={self._format_duration(payload['duration_seconds'])}"
            return [f"checkpoint {payload['action']}: {payload['path']}{location}{duration}"]

        if event == "exception":
            lines = [f"error {payload['error_type']}: {payload['message']}"]
            context = payload.get("context")
            if context:
                lines.append(f"  context: {json.dumps(context, ensure_ascii=True, sort_keys=True)}")
            return lines

        if event == "run_finished":
            lines = [
                f"run {payload['status']} total_steps={payload['total_steps']} "
                f"duration={self._format_duration(payload['duration_seconds'])}",
                f"  logs={payload['run_dir']}",
            ]
            if payload["no_training_steps_completed"]:
                lines.append("  no training steps completed")
            else:
                lines.append(
                    "  perf "
                    f"avg_step={self._format_duration(payload['avg_step_seconds'])} "
                    f"slowest_step={self._format_duration(payload['slowest_step_seconds'])} "
                    f"slowest_phase={payload['slowest_phase']}"
                )

            if payload["lifecycle_totals"]:
                lines.append(
                    self._format_ranked_breakdown(
                        "  lifecycle ",
                        payload["lifecycle_totals"],
                        payload["lifecycle_percentages"],
                    )
                )
            if payload["startup_components"]:
                lines.append(
                    self._format_ranked_breakdown(
                        "  startup_components ",
                        payload["startup_components"],
                    )
                )
            if not payload["no_training_steps_completed"] and payload["hot_path_totals"]:
                lines.append(
                    self._format_named_breakdown(
                        "  phases ",
                        payload["phase_group_totals"],
                        percentages=payload["phase_group_percentages"],
                        order=[name for name, _ in PHASE_GROUP_SPECS],
                    )
                )
            return lines

        return [f"{event}: {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"]

    def _write_lines(self, lines: list[str], *, level: str) -> None:
        if self.config.console:
            self._write_console_lines(lines, level=level)

        if self._transcript_console is not None:
            for line in lines:
                self._transcript_console.print(line)
            self._transcript_handle.flush()
        elif self._transcript_handle is not None:
            for line in lines:
                self._transcript_handle.write(line + "\n")
            self._transcript_handle.flush()

    def _write_console_lines(self, lines: list[str], *, level: str) -> None:
        style = "red" if level == "ERROR" else "cyan" if lines and lines[0].startswith("FlashRL") else None
        if self._console is not None:
            for index, line in enumerate(lines):
                line_style = style if index == 0 else None
                self._console.print(line, style=line_style)
        else:
            for line in lines:
                print(line)

    def _render_step_phase_line(self, payload: dict[str, Any]) -> str:
        return self._format_key_value_line(
            self._step_context_parts(payload)
            + [
                f"phase={payload['phase']}",
                f"stage={payload['stage']}",
                f"latency={self._format_duration(float(payload['latency_seconds']))}",
            ]
            + self._ordered_phase_fields(payload),
        )

    def _render_step_done_line(self, payload: dict[str, Any]) -> str:
        return self._format_key_value_line(
            self._step_context_parts(payload)
            + [
                "phase=step_done",
                "stage=complete",
            ]
            + self._ordered_fields(payload, STEP_DONE_FIELD_ORDER),
        )

    def _step_context_parts(self, payload: dict[str, Any]) -> list[str]:
        return [
            f"step={int(payload['step'])}",
            f"epoch={int(payload['epoch'])}/{int(payload['total_epochs'])}",
            f"batch={int(payload['batch_index'])}/{int(payload['batches_in_epoch'])}",
            f"batch_size={int(payload['batch_size'])}",
        ]

    def _ordered_phase_fields(self, payload: dict[str, Any]) -> list[str]:
        order = STEP_PHASE_FIELD_ORDERS.get(str(payload["phase"]), [])
        return self._ordered_fields(payload, order)

    def _ordered_fields(
        self,
        payload: dict[str, Any],
        order: list[str],
    ) -> list[str]:
        return [
            f"{field}={self._format_field_value(field, payload[field])}"
            for field in order
            if field in payload
        ]

    def _format_key_value_line(self, parts: list[str]) -> str:
        return " ".join(part for part in parts if part)

    def _format_field_value(self, field: str, value: Any) -> str:
        if field.endswith("_seconds") or field == "latency":
            return self._format_duration(float(value))
        if field == "learning_rate":
            return f"{float(value):.2e}"
        if field == "tokens_per_second":
            return f"{float(value):.1f}"
        if field.endswith("_mean"):
            return f"{float(value):.1f}" if "tokens" in field else f"{float(value):.4f}"
        if field.endswith("_std") or field.endswith("_min") or field.endswith("_max"):
            return f"{float(value):.1f}" if "tokens" in field else f"{float(value):.4f}"
        if field.endswith("_total") or field == "batch_size":
            return str(int(round(float(value))))
        if field in {"loss", "policy_loss", "kl_divergence", "reward_mean"}:
            return f"{float(value):.4f}"
        return str(value)

    def _percentages(self, totals: dict[str, float]) -> dict[str, float]:
        denominator = sum(totals.values())
        if denominator <= 0:
            return {key: 0.0 for key in totals}
        return {
            key: (value / denominator) * 100.0
            for key, value in totals.items()
        }

    def _format_ranked_breakdown(
        self,
        prefix: str,
        totals: dict[str, Any],
        percentages: dict[str, Any] | None = None,
    ) -> str:
        ranked = sorted(
            (
                (name, float(value), float(percentages.get(name, 0.0)) if percentages else None)
                for name, value in totals.items()
                if float(value) > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            return prefix + "no timing data"

        parts = []
        for name, value, percent in ranked:
            rendered = f"{name}={self._format_duration(value)}"
            if percent is not None:
                rendered += f" ({percent:.1f}%)"
            parts.append(rendered)
        return prefix + ", ".join(parts)

    def _format_named_breakdown(
        self,
        prefix: str,
        totals: dict[str, Any],
        *,
        order: list[str],
        percentages: dict[str, Any] | None = None,
    ) -> str:
        parts = []
        for name in order:
            value = float(totals.get(name, 0.0))
            if value <= 0:
                continue
            rendered = f"{name}={self._format_duration(value)}"
            if percentages is not None and name in percentages:
                rendered += f" ({float(percentages[name]):.1f}%)"
            parts.append(rendered)
        if not parts:
            return prefix + "no timing data"
        return prefix + " ".join(parts)

    def _format_phase_entries(
        self,
        prefix: str,
        entries: list[dict[str, Any]],
    ) -> str:
        if not entries:
            return prefix + "no timing data"
        return self._format_named_breakdown(
            prefix,
            self._phase_entry_totals(entries),
            order=[str(entry["name"]) for entry in entries],
        )

    def _phase_entries_by_name(
        self,
        phase_breakdown: list[dict[str, Any]],
        specs: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        entries_by_name = {
            str(entry["name"]): {
                "name": str(entry["name"]),
                "seconds": float(entry["seconds"]),
            }
            for entry in phase_breakdown
        }
        return [
            entries_by_name[name]
            for name, _ in specs
            if name in entries_by_name
        ]

    def _phase_entry_totals(self, entries: list[dict[str, Any]]) -> dict[str, float]:
        return {
            str(entry["name"]): float(entry["seconds"])
            for entry in entries
            if float(entry["seconds"]) > 0
        }

    def _build_phase_group_totals(self, hot_path_totals: dict[str, float]) -> dict[str, float]:
        return {
            name: sum(float(hot_path_totals.get(key, 0.0)) for key in keys)
            for name, keys in PHASE_GROUP_SPECS
            if sum(float(hot_path_totals.get(key, 0.0)) for key in keys) > 0
        }

    def _build_named_breakdown(
        self,
        totals: dict[str, Any],
        specs: list[tuple[str, str]],
    ) -> dict[str, float]:
        return {
            name: float(totals.get(key, 0.0))
            for name, key in specs
            if float(totals.get(key, 0.0)) > 0
        }

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._serialize(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _truncate(self, text: str, limit: int = 220) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _format_duration(self, seconds: float) -> str:
        value = float(seconds)
        if value < 1:
            return f"{value * 1000:.1f}ms"
        return f"{value:.3f}s"
