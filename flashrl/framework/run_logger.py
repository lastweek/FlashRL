"""Run-scoped logging for FlashRL training."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from flashrl.framework.config import LoggingConfig


STAGE_DISPLAY_ORDER = [
    "rollout",
    "reward",
    "advantage",
    "prepare_inputs",
    "actor_forward",
    "reference_forward",
    "loss_assembly",
    "backward",
    "optimizer",
    "sync",
]


def _sanitize_model_name(model_name: str) -> str:
    """Convert a model name into a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()
    return slug or "model"


class RunLogger:
    """Run-scoped append-only logger with structured event output."""

    def __init__(self, config: LoggingConfig, model_name: str) -> None:
        """Initialize the run logger."""
        self.config = config
        self.model_name = model_name
        self.run_id = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{_sanitize_model_name(model_name)}-"
            f"{uuid4().hex[:8]}"
        )
        self.run_dir = Path(config.log_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.console_path = self.run_dir / "console.log"
        self.rollouts_path = self.run_dir / "rollouts.jsonl"
        self.events_path.touch(exist_ok=True)
        self.console_path.touch(exist_ok=True)
        self.rollouts_path.touch(exist_ok=True)

        self._total_step_count = 0
        self._total_step_seconds = 0.0
        self._slowest_step_seconds = 0.0
        self._dominant_stage = "n/a"
        self._stage_totals: dict[str, float] = {}
        self._total_batches = 0
        self._current_step_header: int | None = None

    def start_run(
        self,
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
        group_size: int | None = None,
        clip_ratio: float | None = None,
    ) -> None:
        """Log the start of a training run."""
        self._total_batches = total_batches
        self._current_step_header = None

        payload = {
            "run_id": self.run_id,
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
            "group_size": group_size,
            "clip_ratio": clip_ratio,
            "run_dir": str(self.run_dir),
        }
        self._emit_event("run_started", payload)

        if self.config.console_mode == "compact":
            self._emit_console_lines(
                self._format_compact_run_header(
                    device=device,
                    dtype=dtype,
                    cpu_threads=cpu_threads,
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    total_batches=total_batches,
                    runtime_shape=runtime_shape,
                    reference_enabled=reference_enabled,
                    reference_device=reference_device,
                    group_size=group_size,
                    clip_ratio=clip_ratio,
                )
            )
            return

        reference_state = "enabled" if reference_enabled else "disabled"
        self._emit_console(f"FlashRL training run {self.run_id}")
        self._emit_console(
            f"  model={self.model_name} device={device} dtype={dtype} cpu_threads={cpu_threads}"
        )
        self._emit_console(
            f"  dataset={dataset_size} batch_size={batch_size} epochs={max_epochs} total_batches={total_batches}"
        )
        self._emit_console(
            f"  runtime={runtime_shape} reference={reference_state} reference_device={reference_device}"
        )
        if group_size is not None:
            line = f"  grpo=group_size:{group_size}"
            if clip_ratio is not None:
                line += f" clip_ratio={clip_ratio:.4f}"
            self._emit_console(line)
        self._emit_console(f"  logs={self.run_dir}")

    def log_model_load(
        self,
        component: str,
        status: str,
        metadata: dict[str, Any],
    ) -> None:
        """Log startup model-loading events."""
        payload = {
            "component": component,
            "status": status,
            **metadata,
        }
        self._emit_event("model_load", payload)

        if status != "completed":
            return
        if self.config.console_mode == "compact":
            self._emit_console(self._format_compact_load_line(component, metadata))
            return

        line = f"loaded {component}"
        if metadata.get("device") is not None:
            line += f" device={metadata['device']}"
        if metadata.get("cpu_threads") is not None:
            line += f" cpu_threads={metadata['cpu_threads']}"
        if metadata.get("duration_seconds") is not None:
            line += f" latency={self._format_duration(metadata['duration_seconds'])}"
        self._emit_console(line)

    def log_epoch_start(self, epoch: int, total_epochs: int, num_batches: int) -> None:
        """Log the start of an epoch."""
        payload = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "num_batches": num_batches,
        }
        self._emit_event("epoch_start", payload)
        self._current_step_header = None
        if self.config.console_mode == "compact":
            self._emit_console(f"epoch {epoch}/{total_epochs}  batches={num_batches}")
            return
        self._emit_console(f"epoch {epoch}/{total_epochs} (batches={num_batches})")

    def log_step_stage(self, payload: dict[str, Any]) -> None:
        """Log one completed training stage."""
        if not self._should_log_step(int(payload["step"])):
            return

        self._emit_event("step_stage", payload)

        if self.config.console_mode == "compact":
            if self._current_step_header != int(payload["step"]):
                self._emit_console(self._format_compact_step_header(payload))
                self._current_step_header = int(payload["step"])
            self._emit_console(self._format_compact_stage_row(payload))
            return

        line = self._format_verbose_step_prefix(payload)
        line += f" latency={self._format_duration(float(payload['latency_seconds']))}"
        for key in self._verbose_stage_keys(str(payload["stage"])):
            if key in payload:
                line += f" {key}={self._format_verbose_value(key, payload[key])}"
        self._emit_console(line)

    def log_step_done(self, payload: dict[str, Any]) -> None:
        """Log the completion of a training step."""
        stage_timings = {
            key: float(value) for key, value in payload.get("stage_timings", {}).items()
        }
        step_duration_seconds = float(payload.get("step_duration_seconds", 0.0))

        self._total_step_count += 1
        self._total_step_seconds += step_duration_seconds
        self._accumulate_totals(self._stage_totals, stage_timings)
        if step_duration_seconds >= self._slowest_step_seconds:
            self._slowest_step_seconds = step_duration_seconds
            self._dominant_stage = str(payload.get("dominant_stage", self._dominant_stage))

        if not self._should_log_step(int(payload["step"])):
            return

        self._emit_event("step_done", payload)

        if self.config.console_mode == "compact":
            if self._current_step_header != int(payload["step"]):
                self._emit_console(self._format_compact_step_header(payload))
                self._current_step_header = int(payload["step"])
            self._emit_console(self._format_compact_done_row(payload))
            return

        line = self._format_verbose_step_prefix({**payload, "stage": "complete"})
        line += (
            f" loss={self._format_verbose_value('loss', payload['loss'])}"
            f" policy_loss={self._format_verbose_value('policy_loss', payload['policy_loss'])}"
            f" kl_divergence={self._format_verbose_value('kl_divergence', payload['kl_divergence'])}"
            f" reward_mean={self._format_verbose_value('reward_mean', payload['reward_mean'])}"
            f" response_tokens_total={self._format_verbose_value('response_tokens_total', payload['response_tokens_total'])}"
            f" tokens_per_second={self._format_verbose_value('tokens_per_second', payload['tokens_per_second'])}"
            f" step_duration_seconds={self._format_duration(step_duration_seconds)}"
        )
        self._emit_console(line)

    def log_sample_preview(
        self,
        step: int,
        prompt: str,
        response: str,
        reward: float,
    ) -> None:
        """Log one sampled prompt/response pair as a structured event only."""
        if not self._should_sample_step(step):
            return

        prompt_preview = self._truncate(prompt)
        response_preview = self._truncate(response)
        payload = {
            "step": step,
            "reward": reward,
            "prompt_preview": prompt_preview,
            "response_preview": response_preview,
        }
        self._emit_event("sample_preview", payload)

    def log_rollout_batch(
        self,
        *,
        step: int,
        epoch: int,
        batch_index: int,
        batches_in_epoch: int,
        prompts: list[Any],
        rollouts: list[Any],
        rewards: list[Any],
        prompt_indices: list[int],
        candidate_indices: list[int],
        group_size: int,
        prompt_count: int,
    ) -> None:
        """Persist one full-fidelity rollout record per sample."""
        sample_count = len(prompts)
        for sample_index, (prompt, rollout, reward, prompt_index, candidate_index) in enumerate(
            zip(prompts, rollouts, rewards, prompt_indices, candidate_indices, strict=True),
            start=1,
        ):
            record = {
                "run_id": self.run_id,
                "step": step,
                "epoch": epoch,
                "batch_index": batch_index,
                "batches_in_epoch": batches_in_epoch,
                "sample_index": sample_index,
                "prompt_index": prompt_index,
                "candidate_index": candidate_index,
                "group_size": group_size,
                "prompt_count": prompt_count,
                "sample_count": sample_count,
                "prompt": {
                    "text": prompt.text,
                    "metadata": self._serialize_for_json(prompt.metadata),
                },
                "rollout": {
                    "response_text": rollout.text,
                    "log_prob": rollout.log_prob,
                    "prompt_token_count": len(getattr(rollout, "prompt_token_ids", [])),
                    "response_token_count": len(getattr(rollout, "response_token_ids", [])),
                    "metadata": self._serialize_for_json(rollout.metadata),
                },
                "conversation": self._serialize_for_json(rollout.conversation),
                "reward": {
                    "value": reward.reward,
                    "metadata": self._serialize_for_json(reward.metadata),
                },
            }
            self._emit_rollout_record(record)

    def log_epoch_summary(self, payload: dict[str, Any]) -> None:
        """Log a summarized view of one epoch."""
        stage_totals = {
            key: float(value) for key, value in payload.get("stage_totals", {}).items()
        }
        payload = {
            **payload,
            "stage_totals": stage_totals,
            "stage_percentages": self._percentages(stage_totals),
        }
        self._emit_event("epoch_summary", payload)
        self._current_step_header = None

        if self.config.console_mode == "compact":
            self._emit_console_lines(self._format_compact_epoch_summary(payload))
            return

        self._emit_console(
            f"epoch {payload['epoch']} summary"
            f" loss={self._format_verbose_value('loss', payload['loss'])}"
            f" reward={self._format_verbose_value('reward', payload['reward'])}"
            f" kl={self._format_verbose_value('kl_divergence', payload['kl_divergence'])}"
            f" tok/s={self._format_verbose_value('tokens_per_second', payload['tokens_per_second'])}"
            f" duration={self._format_duration(float(payload['duration_seconds']))}"
        )
        if stage_totals:
            self._emit_console(
                "  stages "
                + self._format_verbose_stage_totals(
                    payload["stage_totals"],
                    payload["stage_percentages"],
                )
            )

    def log_exception(self, exc: Exception, context: dict[str, Any]) -> None:
        """Log an exception during training."""
        payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "context": context,
        }
        self._emit_event("exception", payload)

        line = f"error type={type(exc).__name__} message={self._truncate(str(exc), limit=200)}"
        stage = context.get("stage")
        if stage is not None:
            line += f" stage={stage}"
        self._emit_console(line)

    def log_checkpoint(
        self,
        action: str,
        path: str,
        epoch: int,
        step: int,
        duration_seconds: float,
    ) -> None:
        """Log checkpoint save/load operations."""
        payload = {
            "action": action,
            "path": path,
            "epoch": epoch,
            "step": step,
            "duration_seconds": duration_seconds,
        }
        self._emit_event("checkpoint", payload)

        if self.config.console_mode == "compact":
            self._emit_console(
                f"checkpoint {action:<4} epoch={epoch} step={step}  {self._format_duration(duration_seconds)}  {path}"
            )
            return

        self._emit_console(
            f"checkpoint action={action} epoch={epoch} step={step}"
            f" path={path} latency={self._format_duration(duration_seconds)}"
        )

    def finish_run(
        self,
        status: str,
        total_steps: int,
        lifecycle_totals: dict[str, float] | None = None,
    ) -> None:
        """Log the end of a training run."""
        lifecycle_totals = {
            key: float(value)
            for key, value in (lifecycle_totals or {}).items()
        }
        stage_totals = dict(self._stage_totals)
        no_training_steps_completed = self._total_step_count == 0

        top_level_duration = (
            lifecycle_totals.get("startup_total_seconds", 0.0)
            + lifecycle_totals.get("training_loop_seconds", 0.0)
            + lifecycle_totals.get("checkpoint_save_seconds", 0.0)
            + lifecycle_totals.get("checkpoint_load_seconds", 0.0)
        )

        payload = {
            "status": status,
            "total_steps": total_steps,
            "run_dir": str(self.run_dir),
            "lifecycle_totals": lifecycle_totals,
            "stage_totals": {} if no_training_steps_completed else stage_totals,
            "stage_percentages": {} if no_training_steps_completed else self._percentages(stage_totals),
            "avg_step_seconds": (
                0.0 if no_training_steps_completed else self._total_step_seconds / self._total_step_count
            ),
            "slowest_step_seconds": 0.0 if no_training_steps_completed else self._slowest_step_seconds,
            "dominant_stage": "n/a" if no_training_steps_completed else self._dominant_stage,
            "no_training_steps_completed": no_training_steps_completed,
        }
        self._emit_event("run_finished", payload)
        self._current_step_header = None

        if self.config.console_mode == "compact":
            self._emit_console_lines(
                self._format_compact_run_summary(payload, lifecycle_totals, top_level_duration)
            )
            return

        self._emit_console(
            f"run {status} total_steps={total_steps} duration={self._format_duration(top_level_duration)}"
        )
        self._emit_console(f"  logs={self.run_dir}")
        if no_training_steps_completed:
            self._emit_console("  no training steps completed")
        else:
            self._emit_console(
                f"  perf avg_step={self._format_duration(payload['avg_step_seconds'])}"
                f" slowest_step={self._format_duration(payload['slowest_step_seconds'])}"
                f" dominant_stage={payload['dominant_stage']}"
            )
            self._emit_console(
                "  stages "
                + self._format_verbose_stage_totals(
                    payload["stage_totals"],
                    payload["stage_percentages"],
                )
            )
        if lifecycle_totals:
            self._emit_console(
                "  lifecycle "
                + " ".join(
                    f"{key}={self._format_duration(value)}"
                    for key, value in lifecycle_totals.items()
                )
            )

    def close(self) -> None:
        """Close the logger."""

    def _format_compact_run_header(
        self,
        *,
        device: str,
        dtype: str,
        cpu_threads: int,
        dataset_size: int,
        batch_size: int,
        max_epochs: int,
        total_batches: int,
        runtime_shape: str,
        reference_enabled: bool,
        reference_device: str,
        group_size: int | None,
        clip_ratio: float | None,
    ) -> list[str]:
        reference_state = "enabled" if reference_enabled else "disabled"
        lines = [
            "FlashRL training run",
            f"  run      {self.run_id}",
            f"  model    {self.model_name}  device={device} dtype={dtype} cpu={cpu_threads}",
            f"  data     dataset={dataset_size} batch={batch_size} epochs={max_epochs} total_batches={total_batches}",
            f"  runtime  {runtime_shape}  reference={reference_state} ref_device={reference_device}",
        ]
        if group_size is not None:
            grpo_line = f"  grpo     group={group_size}"
            if clip_ratio is not None:
                grpo_line += f"  clip={self._format_fixed(float(clip_ratio), 4)}"
            lines.append(grpo_line)
        lines.append(f"  logs     {self.run_dir}")
        return lines

    def _format_compact_load_line(self, component: str, metadata: dict[str, Any]) -> str:
        device = metadata.get("device", "unknown")
        cpu_threads = metadata.get("cpu_threads", "?")
        duration = self._format_duration(float(metadata.get("duration_seconds", 0.0)))
        return f"load {component:<16} device={device}  cpu={cpu_threads}  {duration}"

    def _format_compact_step_header(self, payload: dict[str, Any]) -> str:
        total_batches = self._total_batches if self._total_batches else "?"
        return (
            f"step {payload['step']}/{total_batches}  "
            f"epoch {payload['epoch']}/{payload['total_epochs']}  "
            f"batch {payload['batch_index']}/{payload['batches_in_epoch']}  "
            f"prompts={payload.get('prompt_count', '?')}  "
            f"group={payload.get('group_size', '?')}  "
            f"samples={payload['batch_size']}"
        )

    def _format_compact_stage_row(self, payload: dict[str, Any]) -> str:
        stage = str(payload["stage"])
        latency = self._format_duration(float(payload["latency_seconds"]))
        details = self._format_compact_stage_details(stage, payload)
        row = f"  {stage:<15} {latency:>8}"
        if details:
            row += f"  {details}"
        return row

    def _format_compact_stage_details(self, stage: str, payload: dict[str, Any]) -> str:
        if stage == "rollout":
            return (
                f"prompt_tok {self._format_compact_ratio(payload['prompt_tokens_mean'], payload['prompt_tokens_max'])}  "
                f"response_tok {self._format_compact_ratio(payload['response_tokens_mean'], payload['response_tokens_max'])}"
            )
        if stage == "reward":
            return (
                f"mean {self._format_fixed(float(payload['reward_mean']), 4)}  "
                f"std {self._format_fixed(float(payload['reward_std']), 4)}  "
                f"min {self._format_compact_scalar(payload['reward_min'])}  "
                f"max {self._format_compact_scalar(payload['reward_max'])}"
            )
        if stage == "advantage":
            return (
                f"mean {self._format_fixed(float(payload['advantage_mean']), 4)}  "
                f"std {self._format_fixed(float(payload['advantage_std']), 4)}"
            )
        if stage == "prepare_inputs":
            return (
                f"full_tok {self._format_compact_ratio(payload['full_tokens_mean'], payload['full_tokens_max'])}  "
                f"resp_tok {self._format_compact_scalar(payload['response_tokens_total'])}"
            )
        if stage in {"actor_forward", "reference_forward"}:
            return f"full_tok_total {self._format_compact_scalar(payload['full_tokens_total'])}"
        if stage == "loss_assembly":
            return (
                f"loss {self._format_fixed(float(payload['loss']), 4)}  "
                f"policy {self._format_fixed(float(payload['policy_loss']), 4)}  "
                f"kl {self._format_fixed(float(payload['kl_divergence']), 4)}  "
                f"resp_tok {self._format_compact_scalar(payload['response_tokens_total'])}"
            )
        if stage == "backward":
            return f"loss {self._format_fixed(float(payload['loss']), 4)}"
        if stage == "optimizer":
            return f"lr {self._format_fixed(float(payload['learning_rate']), 4)}"
        if stage == "sync":
            return (
                f"partial {self._format_precise_duration(float(payload['step_duration_seconds']))}  "
                f"tok/s {self._format_fixed(float(payload['tokens_per_second']), 2)}"
            )
        return ""

    def _format_compact_done_row(self, payload: dict[str, Any]) -> str:
        return (
            f"  {'done':<15} {self._format_duration(float(payload['step_duration_seconds'])):>8}  "
            f"loss {self._format_fixed(float(payload['loss']), 4)}  "
            f"reward {self._format_fixed(float(payload['reward_mean']), 4)}  "
            f"tok/s {self._format_fixed(float(payload['tokens_per_second']), 2)}  "
            f"dominant {payload['dominant_stage']}"
        )

    def _format_compact_epoch_summary(self, payload: dict[str, Any]) -> list[str]:
        lines = [
            (
                f"epoch {payload['epoch']}/{payload['total_epochs']} summary  "
                f"loss {self._format_fixed(float(payload['loss']), 4)}  "
                f"reward {self._format_fixed(float(payload['reward']), 4)}  "
                f"kl {self._format_fixed(float(payload['kl_divergence']), 4)}  "
                f"tok/s {self._format_fixed(float(payload['tokens_per_second']), 2)}  "
                f"duration {self._format_duration(float(payload['duration_seconds']))}"
            )
        ]
        if payload["stage_totals"]:
            lines.append(
                "  stages  "
                + self._format_compact_stage_totals(
                    payload["stage_totals"],
                    payload["stage_percentages"],
                )
            )
        return lines

    def _format_compact_run_summary(
        self,
        payload: dict[str, Any],
        lifecycle_totals: dict[str, float],
        top_level_duration: float,
    ) -> list[str]:
        lines = [f"run {payload['status']}  total_steps={payload['total_steps']}  duration={self._format_duration(top_level_duration)}"]
        if payload["no_training_steps_completed"]:
            lines.append("  perf      no training steps completed")
        else:
            lines.append(
                f"  perf      avg_step={self._format_duration(float(payload['avg_step_seconds']))}  "
                f"slowest={self._format_duration(float(payload['slowest_step_seconds']))}  "
                f"dominant={payload['dominant_stage']}"
            )
        if lifecycle_totals:
            lines.append("  lifecycle " + self._format_compact_lifecycle(lifecycle_totals))
        if payload["stage_totals"]:
            lines.append(
                "  stages    "
                + self._format_compact_stage_totals(
                    payload["stage_totals"],
                    payload["stage_percentages"],
                )
            )
        lines.append(f"  logs      {self.run_dir}")
        return lines

    def _format_compact_lifecycle(self, lifecycle_totals: dict[str, float]) -> str:
        label_map = {
            "startup_training_backend_seconds": "train_init",
            "startup_serving_backend_seconds": "serve_init",
            "startup_reference_model_seconds": "ref_init",
            "startup_total_seconds": "startup",
            "training_loop_seconds": "train_loop",
            "checkpoint_save_seconds": "ckpt_save",
            "checkpoint_load_seconds": "ckpt_load",
        }
        ordered_keys = [key for key in label_map if key in lifecycle_totals] + [
            key for key in lifecycle_totals if key not in label_map
        ]
        return " | ".join(
            f"{label_map.get(key, key)} {self._format_duration(lifecycle_totals[key])}"
            for key in ordered_keys
        )

    def _format_compact_stage_totals(
        self,
        totals: dict[str, float],
        percentages: dict[str, float],
    ) -> str:
        keys = [
            key
            for key in STAGE_DISPLAY_ORDER
            if key in totals and totals[key] > 0.0
        ] + [key for key in totals if key not in STAGE_DISPLAY_ORDER and totals[key] > 0.0]
        return " | ".join(
            f"{key} {self._format_duration(totals[key])} ({percentages.get(key, 0.0):.1f}%)"
            for key in keys
        )

    def _format_verbose_step_prefix(self, payload: dict[str, Any]) -> str:
        return (
            f"step={payload['step']}"
            f" epoch={payload['epoch']}/{payload['total_epochs']}"
            f" batch={payload['batch_index']}/{payload['batches_in_epoch']}"
            f" batch_size={payload['batch_size']}"
            f" prompt_count={payload.get('prompt_count', '?')}"
            f" group_size={payload.get('group_size', '?')}"
            f" stage={payload['stage']}"
        )

    def _verbose_stage_keys(self, stage: str) -> list[str]:
        ordered_keys = {
            "rollout": [
                "sample_count",
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
                "reward_per_item_mean_seconds",
            ],
            "advantage": [
                "advantage_mean",
                "advantage_std",
                "advantage_min",
                "advantage_max",
            ],
            "prepare_inputs": [
                "full_tokens_mean",
                "full_tokens_max",
                "response_tokens_total",
            ],
            "actor_forward": ["full_tokens_total"],
            "reference_forward": ["full_tokens_total"],
            "loss_assembly": [
                "loss",
                "policy_loss",
                "kl_divergence",
                "response_tokens_total",
            ],
            "backward": ["loss"],
            "optimizer": ["learning_rate"],
            "sync": ["step_duration_seconds", "tokens_per_second"],
        }
        return ordered_keys.get(stage, [])

    def _format_verbose_stage_totals(
        self,
        totals: dict[str, float],
        percentages: dict[str, float],
    ) -> str:
        keys = [key for key in STAGE_DISPLAY_ORDER if key in totals] + [
            key for key in totals if key not in STAGE_DISPLAY_ORDER
        ]
        return " ".join(
            f"{key}={self._format_duration(totals[key])} ({percentages.get(key, 0.0):.1f}%)"
            for key in keys
        )

    def _should_log_step(self, step: int) -> bool:
        return step % max(self.config.log_every_steps, 1) == 0

    def _should_sample_step(self, step: int) -> bool:
        interval = max(self.config.sample_every_steps, 1)
        return step == 1 or step % interval == 0

    def _emit_console(self, line: str) -> None:
        if self.config.console:
            print(line)
        if self.config.file:
            with self.console_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")

    def _emit_console_lines(self, lines: list[str]) -> None:
        for line in lines:
            self._emit_console(line)

    def _emit_event(self, event: str, payload: dict[str, Any]) -> None:
        if not self.config.file:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _emit_rollout_record(self, payload: dict[str, Any]) -> None:
        if not self.config.file:
            return
        with self.rollouts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _percentages(self, totals: dict[str, float]) -> dict[str, float]:
        total = sum(totals.values())
        if total <= 0.0:
            return {key: 0.0 for key in totals}
        return {
            key: (value / total) * 100.0
            for key, value in totals.items()
        }

    def _accumulate_totals(
        self,
        target: dict[str, float],
        update: dict[str, float],
    ) -> None:
        for key, value in update.items():
            target[key] = target.get(key, 0.0) + float(value)

    def _format_verbose_value(self, key: str, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if key.endswith("_max") or key.endswith("_min") or key.endswith("_total"):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_compact_scalar(self, value: Any) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_compact_ratio(self, mean_value: Any, max_value: Any) -> str:
        return f"{self._format_fixed(float(mean_value), 1)}/{self._format_compact_scalar(max_value)}"

    def _format_fixed(self, value: float, decimals: int) -> str:
        return f"{value:.{decimals}f}"

    def _format_duration(self, value: float) -> str:
        if value < 1.0:
            return f"{value * 1000.0:.1f}ms"
        return f"{value:.3f}s"

    def _format_precise_duration(self, value: float) -> str:
        if value < 1.0:
            return f"{value * 1000.0:.1f}ms"
        return f"{value:.4f}s"

    def _truncate(self, text: str, *, limit: int = 240) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def _serialize_for_json(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value
