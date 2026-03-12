"""Run-scoped logging for FlashRL training."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import textwrap
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

RUN_INDEX_WIDTH = 6
PROMPT_WRAP_WIDTH = 88
PROMPT_MAX_LINES = 6
PROMPT_MAX_CHARS = 320
PROMPT_SEPARATOR = "  " + ("=" * 72)

ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "run_header": "\033[1;36m",
    "meta_label": "\033[1;34m",
    "step_header": "\033[1;36m",
    "stage_label": "\033[1;34m",
    "serving_header": "\033[1;35m",
    "serving_candidate": "\033[1;33m",
    "serving_timing": "\033[1;32m",
    "divider": "\033[2;37m",
    "error": "\033[1;31m",
}


def _sanitize_model_name(model_name: str) -> str:
    """Convert a model name into a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()
    return slug or "model"


def _allocate_run_index(log_dir: Path) -> int:
    """Allocate the next global run index under the run root."""
    log_dir.mkdir(parents=True, exist_ok=True)
    counter_path = log_dir / ".run_counter"
    counter_path.touch(exist_ok=True)

    with counter_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        text = handle.read().strip()
        current = int(text) if text else 0
        run_index = current + 1
        handle.seek(0)
        handle.write(f"{run_index}\n")
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return run_index


def _format_run_index(run_index: int) -> str:
    """Format a run index with stable zero padding."""
    return f"{run_index:0{RUN_INDEX_WIDTH}d}"


class RunLogger:
    """Run-scoped append-only logger with structured event output."""

    def __init__(self, config: LoggingConfig, model_name: str) -> None:
        """Initialize the run logger."""
        self.config = config
        self.model_name = model_name
        self.log_root = Path(config.log_dir)
        self.run_index = _allocate_run_index(self.log_root)
        formatted_run_index = _format_run_index(self.run_index)
        self.run_id = (
            f"{formatted_run_index}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{_sanitize_model_name(model_name)}-"
            f"{uuid4().hex[:8]}"
        )
        self.run_dir = self.log_root / self.run_id
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
        self._serving_stream_open = False
        self._current_serving_prompt_key: tuple[int, int] | None = None
        self._step_block_complete = False

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
        prompts_per_step: int | None = None,
        steps_per_epoch: int | None = None,
        total_planned_steps: int | None = None,
    ) -> None:
        """Log the start of a training run."""
        self._total_batches = total_batches
        self._current_step_header = None
        self._current_serving_prompt_key = None
        self._step_block_complete = False

        payload = {
            "run_id": self.run_id,
            "run_index": self.run_index,
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
            "dataset_prompt_count": dataset_size,
            "planned_prompts_per_step": prompts_per_step,
            "planned_samples_per_step": batch_size,
            "completions_per_prompt": group_size,
            "planned_completions_per_step": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "total_planned_steps": total_planned_steps,
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
                    prompts_per_step=prompts_per_step,
                    steps_per_epoch=steps_per_epoch,
                    total_planned_steps=total_planned_steps,
                )
            )
            return

        reference_state = "enabled" if reference_enabled else "disabled"
        self._emit_console(f"FlashRL training run {self.run_id}")
        self._emit_console(
            f"  model={self.model_name} device={device} dtype={dtype} cpu_threads={cpu_threads}"
        )
        self._emit_console(
            f"  dataset_prompts={dataset_size} epochs={max_epochs} total_batches={total_batches}"
        )
        self._emit_console(
            f"  runtime={runtime_shape} reference={reference_state} reference_device={reference_device}"
        )
        if group_size is not None:
            line = f"  grpo=completions_per_prompt:{group_size}"
            if clip_ratio is not None:
                line += f" clip_ratio={clip_ratio:.4f}"
            self._emit_console(line)
        if prompts_per_step is not None and steps_per_epoch is not None and total_planned_steps is not None:
            self._emit_console(
                "  mapping="
                f"dataset_prompts:{dataset_size} "
                f"prompts_per_step:{prompts_per_step} "
                f"completions_per_prompt:{group_size} "
                f"completions_per_step:{batch_size} "
                f"steps_per_epoch:{steps_per_epoch} "
                f"total_steps:{total_planned_steps}"
            )
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
        self._current_serving_prompt_key = None
        self._step_block_complete = False
        if self.config.console_mode == "compact":
            self._emit_console(f"epoch {epoch}/{total_epochs}  batches={num_batches}")
            return
        self._emit_console(f"epoch {epoch}/{total_epochs} (batches={num_batches})")

    def log_step_stage(self, payload: dict[str, Any]) -> None:
        """Log one completed training stage."""
        if not self._should_log_step(int(payload["step"])):
            return

        self._emit_event("step_stage", payload)

        if self._current_step_header != int(payload["step"]) and self._step_block_complete:
            self._emit_console("")
            self._step_block_complete = False

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
            self._step_block_complete = True
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
        self._step_block_complete = True

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
        """Persist one full-fidelity grouped rollout record per unique prompt."""
        sample_count = len(prompts)
        grouped_records: dict[int, dict[str, Any]] = {}
        for prompt, rollout, reward, prompt_index, candidate_index in zip(
            prompts,
            rollouts,
            rewards,
            prompt_indices,
            candidate_indices,
            strict=True,
        ):
            record = grouped_records.setdefault(
                prompt_index,
                {
                    "run_id": self.run_id,
                    "run_index": self.run_index,
                    "step": step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "batches_in_epoch": batches_in_epoch,
                    "prompt_index": prompt_index,
                    "group_size": group_size,
                    "prompt_count": prompt_count,
                    "sample_count": sample_count,
                    "prompt": {
                        "text": prompt.text,
                        "metadata": self._serialize_for_json(prompt.metadata),
                    },
                    "candidates": [],
                },
            )
            record["candidates"].append(
                {
                    "candidate_index": candidate_index,
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
            )

        for prompt_index in sorted(grouped_records):
            record = grouped_records[prompt_index]
            record["candidates"].sort(key=lambda candidate: int(candidate["candidate_index"]))
            self._emit_rollout_record(record)

    def log_serving_debug_start(self, payload: dict[str, Any]) -> None:
        """Log the start of one live serving candidate stream."""
        prompt_key = (int(payload["step"]), int(payload["prompt_index"]))
        candidate_index = int(payload["candidate_index"])
        prompt_text = str(payload.get("prompt_text") or payload.get("prompt_preview", ""))

        if self._current_serving_prompt_key is None:
            pass
        elif self._current_serving_prompt_key != prompt_key:
            self._emit_console(PROMPT_SEPARATOR)
        else:
            self._emit_console("")

        if self._current_serving_prompt_key != prompt_key:
            header = (
                f"serve step={payload['step']} epoch={payload['epoch']}/{payload['total_epochs']} "
                f"batch={payload['batch_index']}/{payload['batches_in_epoch']} "
                f"prompt={int(payload['prompt_index']) + 1}/{payload['prompt_count']} "
                f"completions_per_prompt={payload['group_size']}"
            )
            self._emit_console(header)
            for line in self._format_serving_prompt_lines(prompt_text):
                self._emit_console(line)
            self._current_serving_prompt_key = prompt_key

        self._emit_console(f"  candidate {candidate_index + 1}/{payload['group_size']}")
        if self.config.console:
            sys.stdout.write("    ")
            sys.stdout.flush()
        self._serving_stream_open = True

    def log_serving_debug_chunk(self, payload: dict[str, Any]) -> None:
        """Write one terminal-only streamed text fragment."""
        if not self.config.console:
            return
        sys.stdout.write(str(payload.get("text", "")))
        sys.stdout.flush()

    def log_serving_debug_done(self, payload: dict[str, Any]) -> None:
        """Log the completion of one live serving candidate stream."""
        if self._serving_stream_open and self.config.console:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._serving_stream_open = False

        event_payload = {
            key: payload[key]
            for key in (
                "step",
                "epoch",
                "total_epochs",
                "batch_index",
                "batches_in_epoch",
                "prompt_count",
                "group_size",
                "prompt_index",
                "candidate_index",
                "ttft_seconds",
                "tpot_seconds",
                "generation_seconds",
                "response_token_count",
            )
            if key in payload
        }
        event_payload["response_preview"] = self._truncate(str(payload.get("response_preview", "")))
        self._emit_event("serving_debug", event_payload)

        self._emit_console(
            "  serve_done "
            f"ttft={self._format_duration(float(payload['ttft_seconds']))} "
            f"tpot={self._format_duration(float(payload['tpot_seconds']))} "
            f"tokens={self._format_compact_scalar(payload['response_token_count'])} "
            f"total={self._format_duration(float(payload['generation_seconds']))}"
        )

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
        self._step_block_complete = False

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
        self._current_serving_prompt_key = None
        self._step_block_complete = False

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
        prompts_per_step: int | None,
        steps_per_epoch: int | None,
        total_planned_steps: int | None,
    ) -> list[str]:
        reference_state = "enabled" if reference_enabled else "disabled"
        lines = [
            "FlashRL training run",
            f"  run      #{_format_run_index(self.run_index)}  {self.run_id}",
            f"  model    {self.model_name}  device={device} dtype={dtype} cpu={cpu_threads}",
            f"  data     dataset_prompts={dataset_size} epochs={max_epochs} total_batches={total_batches}",
            f"  runtime  {runtime_shape}  reference={reference_state} ref_device={reference_device}",
        ]
        if group_size is not None:
            grpo_line = f"  grpo     completions_per_prompt={group_size}"
            if clip_ratio is not None:
                grpo_line += f"  clip={self._format_fixed(float(clip_ratio), 4)}"
            lines.append(grpo_line)
        if prompts_per_step is not None and steps_per_epoch is not None and total_planned_steps is not None:
            lines.append(
                "  mapping  "
                f"dataset_prompts={dataset_size}  "
                f"prompts_per_step={prompts_per_step}  "
                f"completions_per_step={batch_size}"
            )
            lines.append(
                "  progress "
                f"steps_per_epoch={steps_per_epoch}  "
                f"total_steps={total_planned_steps}"
            )
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
            f"prompt_window={payload.get('dataset_prompt_start', '?')}-{payload.get('dataset_prompt_end', '?')}/"
            f"{payload.get('dataset_prompt_count', '?')}  "
            f"prompts_this_step={payload.get('prompt_count', '?')}/{payload.get('planned_prompts_per_step', '?')}  "
            f"completions_per_prompt={payload.get('completions_per_prompt', payload.get('group_size', '?'))}  "
            f"completions_this_step={payload.get('completions_this_step', payload.get('samples_this_step', payload['batch_size']))}/"
            f"{payload.get('planned_completions_per_step', payload.get('planned_samples_per_step', '?'))}"
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
            f" prompt_window={payload.get('dataset_prompt_start', '?')}-{payload.get('dataset_prompt_end', '?')}/{payload.get('dataset_prompt_count', '?')}"
            f" completions_this_step={payload.get('completions_this_step', payload.get('samples_this_step', payload['batch_size']))}"
            f" prompts_this_step={payload.get('prompt_count', '?')}"
            f" planned_prompts_per_step={payload.get('planned_prompts_per_step', '?')}"
            f" completions_per_prompt={payload.get('completions_per_prompt', payload.get('group_size', '?'))}"
            f" planned_completions_per_step={payload.get('planned_completions_per_step', payload.get('planned_samples_per_step', '?'))}"
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
        if self._serving_stream_open and self.config.console:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._serving_stream_open = False
        if self.config.console:
            print(self._style_terminal_line(line))
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
            "run_id": self.run_id,
            "run_index": self.run_index,
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

    def _format_serving_prompt_lines(self, prompt_text: str) -> list[str]:
        """Render the original prompt once with adaptive wrapping/truncation."""
        text = prompt_text.replace("\r\n", "\n").strip()
        if not text:
            return []

        truncated = False
        if len(text) > PROMPT_MAX_CHARS:
            text = text[:PROMPT_MAX_CHARS].rstrip()
            truncated = True

        wrapped_lines: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            segments = textwrap.wrap(
                line,
                width=PROMPT_WRAP_WIDTH,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            if not segments:
                segments = [""]
            wrapped_lines.extend(segments)

        if len(wrapped_lines) > PROMPT_MAX_LINES:
            wrapped_lines = wrapped_lines[:PROMPT_MAX_LINES]
            truncated = True

        if truncated and wrapped_lines:
            suffix = " ... [truncated]"
            last_line = wrapped_lines[-1].rstrip()
            available = max(PROMPT_WRAP_WIDTH - len(suffix), 0)
            if len(last_line) > available:
                last_line = last_line[:available].rstrip()
            wrapped_lines[-1] = f"{last_line}{suffix}".strip()

        lines: list[str] = []
        for index, segment in enumerate(wrapped_lines):
            prefix = "  prompt: " if index == 0 else "          "
            lines.append(f"{prefix}{segment}")
        return lines

    def _style_terminal_line(self, line: str) -> str:
        """Colorize terminal metadata while keeping file logs plain."""
        if not self._should_color_terminal():
            return line

        if line == "FlashRL training run":
            return self._color(line, "run_header")

        if line.startswith("serve step="):
            return self._color(line, "serving_header")
        if line.startswith("  candidate "):
            return self._color(line, "serving_candidate")
        if line.startswith("  serve_done "):
            return self._style_key_value_line(line, "serving_timing")
        if line == PROMPT_SEPARATOR:
            return self._color(line, "divider")
        if line.startswith("  prompt: "):
            label, value = line.split(": ", 1)
            return f"{self._color(label + ':', 'meta_label')} {value}"

        compact_label_match = re.match(r"^(  )(run|model|data|runtime|grpo|logs)(\s+)(.*)$", line)
        if compact_label_match:
            indent, label, spacing, value = compact_label_match.groups()
            return f"{indent}{self._color(label, 'meta_label')}{spacing}{value}"

        if line.startswith("load "):
            component, _, remainder = line.partition("  ")
            return f"{self._color(component, 'meta_label')}  {remainder}"

        if line.startswith("step "):
            return self._color(line, "step_header")
        if re.match(r"^epoch \d+/\d+", line):
            return self._color(line, "step_header")
        if line.startswith("run "):
            return self._color(line, "run_header")
        if line.startswith("checkpoint "):
            return self._color(line, "meta_label")
        if line.startswith("error "):
            return self._color(line, "error")
        if line.startswith("  lifecycle ") or line.startswith("  stages"):
            return self._style_key_value_line(line, "meta_label")

        stage_match = re.match(r"^(  )([a-z_]+|done)(\s+)(\S+)(.*)$", line)
        if stage_match:
            indent, label, spacing, duration, suffix = stage_match.groups()
            return (
                f"{indent}{self._color(label, 'stage_label')}"
                f"{spacing}{self._color(duration, 'serving_timing')}"
                f"{suffix}"
            )

        return line

    def _style_key_value_line(self, line: str, style: str) -> str:
        """Colorize the metadata labels in a key=value line."""
        return re.sub(
            r"([A-Za-z_][A-Za-z0-9_/.-]*=)",
            lambda match: self._color(match.group(1), style),
            line,
        )

    def _should_color_terminal(self) -> bool:
        """Enable ANSI colors only for interactive terminals."""
        if not self.config.console:
            return False
        if os.environ.get("TERM", "").lower() == "dumb":
            return False
        stream = sys.stdout
        return bool(hasattr(stream, "isatty") and stream.isatty())

    def _color(self, text: str, style: str) -> str:
        """Apply one ANSI style when terminal coloring is enabled."""
        prefix = ANSI_STYLES.get(style)
        if prefix is None:
            return text
        return f"{prefix}{text}{ANSI_RESET}"

    def _serialize_for_json(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value
