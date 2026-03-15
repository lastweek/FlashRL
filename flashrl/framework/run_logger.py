"""Run-scoped logging for FlashRL training."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap
from typing import Any
from uuid import uuid4

from flashrl.framework import log_paths, rollout_logging
from flashrl.framework.config import LoggingConfig
from flashrl.framework.observability import RuntimeEvent


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
PROMPT_GIST_MAX_CHARS = 96
SECTION_SEPARATOR = "  " + ("=" * 72)
STEP_SEPARATOR = "  " + ("-" * 72)
ROLLOUT_SCHEMA_VERSION = 2

PROMOTED_PROMPT_METADATA_KEYS = (
    "task_id",
    "source",
    "split",
    "language",
    "rating",
    "verifier",
)
PROMOTED_REWARD_METADATA_KEYS = (
    "pass_rate",
    "passed_tests",
    "total_tests",
    "accuracy_pass",
    "format_pass",
    "truncated",
    "execution_seconds",
    "failure_reason",
    "checker_used",
    "execution_status",
    "code_preview",
)
PROMOTED_OUTPUT_METADATA_KEYS = (
    "finish_reason",
    "stop_reason",
    "ttft_seconds",
    "tpot_seconds",
    "generation_seconds",
    "response_token_count",
)

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


def _clone_json_mapping(value: Any) -> dict[str, Any]:
    """Return a shallow dict copy for JSON-like mappings."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def _clone_json_messages(messages: Any) -> list[dict[str, Any]]:
    """Return a list of shallow-copied JSON-like messages."""
    if not isinstance(messages, list):
        return []
    return [dict(message) for message in messages if isinstance(message, dict)]


def _promote_metadata_fields(
    metadata: dict[str, Any],
    promoted_keys: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a metadata mapping into promoted and leftover fields."""
    promoted: dict[str, Any] = {}
    leftovers: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in promoted_keys:
            promoted[key] = value
            continue
        leftovers[key] = value
    return promoted, leftovers


def _factor_shared_messages(
    message_groups: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Extract the longest shared message prefix across candidate transcripts."""
    if not message_groups:
        return [], []

    prefix_length = min(len(messages) for messages in message_groups)
    shared_length = 0
    while shared_length < prefix_length:
        candidate = message_groups[0][shared_length]
        if any(messages[shared_length] != candidate for messages in message_groups[1:]):
            break
        shared_length += 1

    if shared_length == 0:
        return [], [_clone_json_messages(messages) for messages in message_groups]

    return (
        _clone_json_messages(message_groups[0][:shared_length]),
        [_clone_json_messages(messages[shared_length:]) for messages in message_groups],
    )


def _derive_log_prob_stats(
    *,
    log_prob: float,
    response_token_count: int,
    prompt_token_count: int,
) -> dict[str, float | None]:
    """Compute normalized rollout confidence and length ratios."""
    avg_log_prob_per_token: float | None = None
    avg_token_prob: float | None = None
    if response_token_count > 0:
        avg_log_prob_per_token = float(log_prob / response_token_count)
        avg_token_prob = float(math.exp(avg_log_prob_per_token))

    output_to_prompt_token_ratio: float | None = None
    if prompt_token_count > 0:
        output_to_prompt_token_ratio = float(response_token_count / prompt_token_count)

    return {
        "avg_log_prob_per_token": avg_log_prob_per_token,
        "avg_token_prob": avg_token_prob,
        "output_to_prompt_token_ratio": output_to_prompt_token_ratio,
    }


def _candidate_is_solved(reward: dict[str, Any]) -> bool:
    """Infer whether one candidate solved the task from promoted reward fields."""
    accuracy_pass = reward.get("accuracy_pass")
    if accuracy_pass is not None:
        return bool(accuracy_pass)

    passed_tests = reward.get("passed_tests")
    total_tests = reward.get("total_tests")
    if isinstance(passed_tests, int) and isinstance(total_tests, int) and total_tests > 0:
        return passed_tests == total_tests

    pass_rate = reward.get("pass_rate")
    if isinstance(pass_rate, (int, float)):
        return float(pass_rate) >= 1.0

    return False


def _derive_rollout_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute prompt-group summary stats from serialized candidates."""
    if not candidates:
        return {
            "reward_mean": 0.0,
            "reward_max": 0.0,
            "reward_min": 0.0,
            "reward_span": 0.0,
            "best_candidate_index": None,
            "fastest_candidate_index": None,
            "longest_candidate_index": None,
            "format_pass_count": 0,
            "accuracy_pass_count": 0,
            "truncated_count": 0,
            "solved_count": 0,
            "avg_pass_rate": None,
            "avg_generation_seconds": None,
            "avg_log_prob_per_token": None,
        }

    reward_values = [float(candidate["reward"]["value"]) for candidate in candidates]
    generation_pairs = [
        (float(candidate["output"]["generation_seconds"]), int(candidate["candidate_index"]))
        for candidate in candidates
        if isinstance(candidate["output"].get("generation_seconds"), (int, float))
    ]
    longest_pairs = [
        (int(candidate["output"]["response_token_count"]), int(candidate["candidate_index"]))
        for candidate in candidates
    ]
    avg_pass_rates = [
        float(candidate["reward"]["pass_rate"])
        for candidate in candidates
        if isinstance(candidate["reward"].get("pass_rate"), (int, float))
    ]
    avg_generation_seconds = [
        float(candidate["output"]["generation_seconds"])
        for candidate in candidates
        if isinstance(candidate["output"].get("generation_seconds"), (int, float))
    ]
    avg_log_probs = [
        float(candidate["output"]["avg_log_prob_per_token"])
        for candidate in candidates
        if isinstance(candidate["output"].get("avg_log_prob_per_token"), (int, float))
    ]

    best_candidate = max(
        (
            (float(candidate["reward"]["value"]), -int(candidate["candidate_index"]))
            for candidate in candidates
        ),
        default=(0.0, 0),
    )

    return {
        "reward_mean": float(sum(reward_values) / len(reward_values)),
        "reward_max": max(reward_values),
        "reward_min": min(reward_values),
        "reward_span": max(reward_values) - min(reward_values),
        "best_candidate_index": int(-best_candidate[1]),
        "fastest_candidate_index": (
            min(generation_pairs)[1] if generation_pairs else None
        ),
        "longest_candidate_index": (
            max(longest_pairs)[1] if longest_pairs else None
        ),
        "format_pass_count": sum(bool(candidate["reward"].get("format_pass")) for candidate in candidates),
        "accuracy_pass_count": sum(
            bool(candidate["reward"].get("accuracy_pass")) for candidate in candidates
        ),
        "truncated_count": sum(bool(candidate["reward"].get("truncated")) for candidate in candidates),
        "solved_count": sum(_candidate_is_solved(candidate["reward"]) for candidate in candidates),
        "avg_pass_rate": (
            float(sum(avg_pass_rates) / len(avg_pass_rates)) if avg_pass_rates else None
        ),
        "avg_generation_seconds": (
            float(sum(avg_generation_seconds) / len(avg_generation_seconds))
            if avg_generation_seconds
            else None
        ),
        "avg_log_prob_per_token": (
            float(sum(avg_log_probs) / len(avg_log_probs)) if avg_log_probs else None
        ),
    }


class RunLogger:
    """Run-scoped append-only logger with structured event output."""

    def __init__(self, config: LoggingConfig, model_name: str) -> None:
        """Initialize the run logger."""
        self.config = config
        self.model_name = model_name
        self.log_root = Path(config.log_dir)
        self.run_index = log_paths.allocate_run_index(self.log_root)
        formatted_run_index = log_paths.format_run_index(self.run_index)
        self.run_id = (
            f"{formatted_run_index}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{log_paths.sanitize_model_name(model_name)}-"
            f"{uuid4().hex[:8]}"
        )
        self.run_dir = self.log_root / self.run_id
        self._initialize_run_files()
        self._reset_runtime_state()
        self._reset_aggregates()

    @classmethod
    def open_existing_run(
        cls,
        config: LoggingConfig,
        model_name: str,
        *,
        run_id: str,
        run_index: int,
        run_dir: str | Path,
        restored_state: dict[str, Any] | None = None,
    ) -> "RunLogger":
        """Reopen an existing run directory and restore aggregate logger state."""
        logger = cls.__new__(cls)
        logger.config = config
        logger.model_name = model_name
        logger.log_root = Path(config.log_dir)
        logger.run_id = str(run_id)
        logger.run_index = int(run_index)
        logger.run_dir = Path(run_dir)
        if not logger.run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {logger.run_dir}")
        logger._initialize_run_files()
        logger._reset_runtime_state()
        logger._reset_aggregates()
        logger.restore_state(restored_state)
        return logger

    def _initialize_run_files(self) -> None:
        """Ensure the run directory and append-only artifact files exist."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.console_path = self.run_dir / "console.log"
        self.rollouts_path = self.run_dir / "rollouts.jsonl"
        self.events_path.touch(exist_ok=True)
        self.console_path.touch(exist_ok=True)
        self.rollouts_path.touch(exist_ok=True)

    def _reset_runtime_state(self) -> None:
        """Reset transient step/terminal state before a run starts or resumes."""
        self._total_batches = 0
        self._current_step_header: int | None = None
        self._current_serving_prompt_key: tuple[int, int] | None = None
        self._step_block_complete = False
        self._terminal_serving_status_open = False
        self._terminal_serving_status_length = 0

    def _reset_aggregates(self) -> None:
        """Reset cumulative per-run aggregates."""
        self._total_step_count = 0
        self._total_step_seconds = 0.0
        self._slowest_step_seconds = 0.0
        self._dominant_stage = "n/a"
        self._stage_totals: dict[str, float] = {}

    def export_state(self) -> dict[str, Any]:
        """Serialize the aggregate logger state needed for append-resume."""
        return {
            "schema_version": 1,
            "total_step_count": self._total_step_count,
            "total_step_seconds": self._total_step_seconds,
            "slowest_step_seconds": self._slowest_step_seconds,
            "dominant_stage": self._dominant_stage,
            "stage_totals": dict(self._stage_totals),
        }

    def restore_state(self, restored_state: dict[str, Any] | None) -> None:
        """Restore aggregate logger state from a previous checkpoint."""
        if not isinstance(restored_state, dict):
            return
        self._total_step_count = int(restored_state.get("total_step_count", 0))
        self._total_step_seconds = float(restored_state.get("total_step_seconds", 0.0))
        self._slowest_step_seconds = float(restored_state.get("slowest_step_seconds", 0.0))
        dominant_stage = restored_state.get("dominant_stage", "n/a")
        self._dominant_stage = str(dominant_stage) if dominant_stage else "n/a"
        stage_totals = restored_state.get("stage_totals", {})
        if isinstance(stage_totals, dict):
            self._stage_totals = {
                str(key): float(value)
                for key, value in stage_totals.items()
                if isinstance(value, (int, float))
            }

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
        training_backend: str | None = None,
        training_device: str | None = None,
        training_dp_size: int | None = None,
        serving_backend: str | None = None,
        serving_device: str | None = None,
        serving_num_replicas: int | None = None,
        admin_base_url: str | None = None,
        max_new_tokens: int | None = None,
        include_startup_divider: bool = False,
    ) -> None:
        """Log the start of a training run."""
        self._log_run_open(
            event_name="run_started",
            resume_checkpoint_path=None,
            dataset_size=dataset_size,
            batch_size=batch_size,
            max_epochs=max_epochs,
            total_batches=total_batches,
            device=device,
            dtype=dtype,
            cpu_threads=cpu_threads,
            runtime_shape=runtime_shape,
            reference_enabled=reference_enabled,
            reference_device=reference_device,
            group_size=group_size,
            clip_ratio=clip_ratio,
            prompts_per_step=prompts_per_step,
            steps_per_epoch=steps_per_epoch,
            total_planned_steps=total_planned_steps,
            training_backend=training_backend,
            training_device=training_device,
            training_dp_size=training_dp_size,
            serving_backend=serving_backend,
            serving_device=serving_device,
            serving_num_replicas=serving_num_replicas,
            admin_base_url=admin_base_url,
            max_new_tokens=max_new_tokens,
            include_startup_divider=include_startup_divider,
        )

    def resume_run(
        self,
        *,
        checkpoint_path: str,
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
        training_backend: str | None = None,
        training_device: str | None = None,
        training_dp_size: int | None = None,
        serving_backend: str | None = None,
        serving_device: str | None = None,
        serving_num_replicas: int | None = None,
        admin_base_url: str | None = None,
        max_new_tokens: int | None = None,
        include_startup_divider: bool = False,
    ) -> None:
        """Log the start of a resumed training run."""
        self._log_run_open(
            event_name="run_resumed",
            resume_checkpoint_path=checkpoint_path,
            dataset_size=dataset_size,
            batch_size=batch_size,
            max_epochs=max_epochs,
            total_batches=total_batches,
            device=device,
            dtype=dtype,
            cpu_threads=cpu_threads,
            runtime_shape=runtime_shape,
            reference_enabled=reference_enabled,
            reference_device=reference_device,
            group_size=group_size,
            clip_ratio=clip_ratio,
            prompts_per_step=prompts_per_step,
            steps_per_epoch=steps_per_epoch,
            total_planned_steps=total_planned_steps,
            training_backend=training_backend,
            training_device=training_device,
            training_dp_size=training_dp_size,
            serving_backend=serving_backend,
            serving_device=serving_device,
            serving_num_replicas=serving_num_replicas,
            admin_base_url=admin_base_url,
            max_new_tokens=max_new_tokens,
            include_startup_divider=include_startup_divider,
        )

    def _log_run_open(
        self,
        *,
        event_name: str,
        resume_checkpoint_path: str | None,
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
        training_backend: str | None = None,
        training_device: str | None = None,
        training_dp_size: int | None = None,
        serving_backend: str | None = None,
        serving_device: str | None = None,
        serving_num_replicas: int | None = None,
        admin_base_url: str | None = None,
        max_new_tokens: int | None = None,
        include_startup_divider: bool = False,
    ) -> None:
        """Emit the common run-open event and header for fresh or resumed runs."""
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
            "training_backend": training_backend,
            "training_device": training_device,
            "training_dp_size": training_dp_size,
            "serving_backend": serving_backend,
            "serving_device": serving_device,
            "serving_num_replicas": serving_num_replicas,
            "admin_base_url": admin_base_url,
            "run_dir": str(self.run_dir),
        }
        if resume_checkpoint_path is not None:
            payload["checkpoint_path"] = resume_checkpoint_path
        self._emit_event(event_name, payload)

        header_lines = self._format_compact_run_header(
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
            training_backend=training_backend,
            training_device=training_device,
            training_dp_size=training_dp_size,
            serving_backend=serving_backend,
            serving_device=serving_device,
            serving_num_replicas=serving_num_replicas,
            admin_base_url=admin_base_url,
            max_new_tokens=max_new_tokens,
        )
        if resume_checkpoint_path is not None:
            header_lines.insert(2, f"  resume   {resume_checkpoint_path}")

        if self.config.console_mode == "compact":
            if include_startup_divider:
                self._emit_console(SECTION_SEPARATOR)
            self._emit_console_lines(header_lines)
            return

        if include_startup_divider:
            self._emit_console(SECTION_SEPARATOR)
        self._emit_console_lines(header_lines)

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

    def replay_startup_lines(self, lines: list[str]) -> None:
        """Replay cached startup lines into console.log without reprinting them live."""
        for line in lines:
            self._emit_file_line(line)

    def observe_event(self, event: RuntimeEvent) -> None:
        """Consume one typed runtime event."""
        if event.kind == "step_start":
            self.log_step_start(event.payload)
            return
        if event.kind == "step_stage":
            self.log_step_stage(event.payload)
            return
        if event.kind == "step_done":
            self.log_step_done(event.payload)
            return
        if event.kind == "serving_debug_start":
            self.log_serving_debug_start(event.payload)
            return
        if event.kind == "serving_debug_chunk":
            self.log_serving_debug_chunk(event.payload)
            return
        if event.kind == "serving_debug_done":
            self.log_serving_debug_done(event.payload)

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

    def log_step_start(self, payload: dict[str, Any]) -> None:
        """Log the start of a training step before rollout begins."""
        if not self._should_log_step(int(payload["step"])):
            return

        self._emit_event("step_start", payload)
        self._begin_step_block(payload)

    def log_step_stage(self, payload: dict[str, Any]) -> None:
        """Log one completed training stage."""
        if not self._should_log_step(int(payload["step"])):
            return

        self._emit_event("step_stage", payload)
        self._ensure_step_header(payload)

        if self.config.console_mode == "compact":
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
        step = int(payload["step"])
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

        if not self._should_log_step(step):
            if self._current_step_header == step:
                self._step_block_complete = True
            return

        self._emit_event("step_done", payload)
        self._ensure_step_header(payload)

        if self.config.console_mode == "compact":
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
        batch_candidate_count = len(prompts)
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
                    "prompt": prompt,
                    "candidates": [],
                },
            )
            record["candidates"].append(
                {
                    "candidate_index": candidate_index,
                    "rollout": rollout,
                    "reward": reward,
                }
            )

        for prompt_index in sorted(grouped_records):
            record = grouped_records[prompt_index]
            record["candidates"].sort(key=lambda candidate: int(candidate["candidate_index"]))
            self._emit_rollout_record(
                self._build_rollout_record(
                    step=step,
                    epoch=epoch,
                    batch_index=batch_index,
                    batches_in_epoch=batches_in_epoch,
                    prompt_index=prompt_index,
                    prompt_count=prompt_count,
                    group_size=group_size,
                    batch_candidate_count=batch_candidate_count,
                    prompt=record["prompt"],
                    candidates=record["candidates"],
                )
            )

    def _build_rollout_record(
        self,
        *,
        step: int,
        epoch: int,
        batch_index: int,
        batches_in_epoch: int,
        prompt_index: int,
        prompt_count: int,
        group_size: int,
        batch_candidate_count: int,
        prompt: Any,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Serialize one prompt-group rollout record in schema v2."""
        return rollout_logging.build_rollout_record(
            run_id=self.run_id,
            run_index=self.run_index,
            step=step,
            epoch=epoch,
            batch_index=batch_index,
            batches_in_epoch=batches_in_epoch,
            prompt_index=prompt_index,
            prompt_count=prompt_count,
            group_size=group_size,
            batch_candidate_count=batch_candidate_count,
            prompt=prompt,
            candidates=candidates,
            serialize_for_json=self._serialize_for_json,
            truncate_text=self._truncate,
        )

    def log_serving_debug_start(self, payload: dict[str, Any]) -> None:
        """Log the start of one live serving candidate stream."""
        self._begin_step_block(payload)
        prompt_key = (int(payload["step"]), int(payload["prompt_index"]))
        candidate_index = int(payload["candidate_index"])
        prompt_text = str(payload.get("prompt_text") or payload.get("prompt_preview", ""))
        prompt_count = int(payload["prompt_count"])
        group_size = int(payload["group_size"])
        is_new_prompt = self._current_serving_prompt_key != prompt_key

        if is_new_prompt:
            if self._current_serving_prompt_key is not None and self._current_serving_prompt_key[0] == prompt_key[0]:
                self._emit_console("")
            self._emit_console(
                f"  prompt {int(payload['prompt_index']) + 1}/{prompt_count}  "
                f"{self._format_serving_prompt_gist(prompt_text)}"
            )
            self._current_serving_prompt_key = prompt_key

        self._emit_file_line(f"    rollout {candidate_index + 1}/{group_size} running...")
        self._start_terminal_serving_status(
            f"    rollout {candidate_index + 1}/{group_size} running..."
        )

    def log_serving_debug_chunk(self, payload: dict[str, Any]) -> None:
        """Ignore streamed fragments in the terminal and keep only structured events."""
        del payload

    def log_serving_debug_done(self, payload: dict[str, Any]) -> None:
        """Log the completion of one live serving candidate stream."""
        done_line = (
            f"    rollout {int(payload['candidate_index']) + 1}/{int(payload['group_size'])} done  "
            f"ttft={self._format_duration(float(payload['ttft_seconds']))}  "
            f"tpot={self._format_duration(float(payload['tpot_seconds']))}  "
            f"tokens={self._format_compact_scalar(payload['response_token_count'])}  "
            f"total={self._format_duration(float(payload['generation_seconds']))}"
        )
        self._finish_terminal_serving_status(done_line)

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

        self._emit_file_line(done_line)

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
        *,
        trigger: str = "manual",
    ) -> None:
        """Log checkpoint save/load operations."""
        payload = {
            "action": action,
            "trigger": trigger,
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
            f"checkpoint action={action} trigger={trigger} epoch={epoch} step={step}"
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
        training_backend: str | None,
        training_device: str | None,
        training_dp_size: int | None,
        serving_backend: str | None,
        serving_device: str | None,
        serving_num_replicas: int | None,
        admin_base_url: str | None,
        max_new_tokens: int | None,
    ) -> list[str]:
        reference_state = "enabled" if reference_enabled else "disabled"
        lines = [
            "FlashRL training run",
            f"  run      #{_format_run_index(self.run_index)}  {self.run_id}",
            f"  model    {self.model_name}  device={device} dtype={dtype} cpu={cpu_threads}",
            f"  data     dataset_prompts={dataset_size} epochs={max_epochs} total_batches={total_batches}",
            f"  runtime  {runtime_shape}  reference={reference_state} ref_device={reference_device}",
        ]
        training_line = self._format_compact_training_line(
            training_backend=training_backend,
            training_device=training_device,
            training_dp_size=training_dp_size,
        )
        if training_line is not None:
            lines.append(training_line)
        serving_line = self._format_compact_serving_line(
            serving_backend=serving_backend,
            serving_device=serving_device,
            serving_num_replicas=serving_num_replicas,
        )
        if serving_line is not None:
            lines.append(serving_line)
        if admin_base_url is not None:
            lines.append(f"  admin    {admin_base_url}")
        if group_size is not None:
            grpo_line = f"  grpo     completions_per_prompt={group_size}"
            if clip_ratio is not None:
                grpo_line += f"  clip={self._format_fixed(float(clip_ratio), 4)}"
            if max_new_tokens is not None:
                grpo_line += f"  max_new_tokens={max_new_tokens}"
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

    def _format_compact_training_line(
        self,
        *,
        training_backend: str | None,
        training_device: str | None,
        training_dp_size: int | None,
    ) -> str | None:
        if training_backend is None or training_device is None:
            return None
        line = f"  training backend={training_backend}  device={training_device}"
        if training_dp_size is not None:
            line += f"  dp_size={training_dp_size}"
        return line

    def _format_compact_serving_line(
        self,
        *,
        serving_backend: str | None,
        serving_device: str | None,
        serving_num_replicas: int | None,
    ) -> str | None:
        if serving_backend is None or serving_device is None:
            return None
        line = f"  serving  backend={serving_backend}  device={serving_device}"
        if serving_backend == "vllm" and serving_num_replicas is not None:
            line += f"  replicas={serving_num_replicas}"
        return line

    def _format_verbose_serving_line(
        self,
        *,
        serving_backend: str | None,
        serving_device: str | None,
        serving_num_replicas: int | None,
    ) -> str | None:
        if serving_backend is None or serving_device is None:
            return None
        line = f"  serving backend={serving_backend} device={serving_device}"
        if serving_backend == "vllm" and serving_num_replicas is not None:
            line += f" replicas={serving_num_replicas}"
        return line

    def _format_verbose_training_line(
        self,
        *,
        training_backend: str | None,
        training_device: str | None,
        training_dp_size: int | None,
    ) -> str | None:
        if training_backend is None or training_device is None:
            return None
        line = f"  training backend={training_backend} device={training_device}"
        if training_dp_size is not None:
            line += f" dp_size={training_dp_size}"
        return line

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
            f"completions_this_step={payload.get('completions_this_step', payload.get('samples_this_step', payload.get('batch_size', '?')))}/"
            f"{payload.get('planned_completions_per_step', payload.get('planned_samples_per_step', '?'))}"
        )

    def _format_step_header(self, payload: dict[str, Any]) -> str:
        """Render the stable step header that starts each visible step block."""
        return self._format_compact_step_header(payload)

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
            details = (
                f"mean {self._format_fixed(float(payload['reward_mean']), 4)}  "
                f"std {self._format_fixed(float(payload['reward_std']), 4)}  "
                f"min {self._format_compact_scalar(payload['reward_min'])}  "
                f"max {self._format_compact_scalar(payload['reward_max'])}"
            )
            if "accuracy_pass_rate" in payload:
                details += f"  acc {self._format_fixed(float(payload['accuracy_pass_rate']), 2)}"
            if "format_pass_rate" in payload:
                details += f"  fmt {self._format_fixed(float(payload['format_pass_rate']), 2)}"
            if "truncation_rate" in payload:
                details += f"  trunc {self._format_fixed(float(payload['truncation_rate']), 2)}"
            return details
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
        metric_suffix = []
        if "accuracy_pass_rate" in payload:
            metric_suffix.append(f"acc {self._format_fixed(float(payload['accuracy_pass_rate']), 2)}")
        if "format_pass_rate" in payload:
            metric_suffix.append(f"fmt {self._format_fixed(float(payload['format_pass_rate']), 2)}")
        if "truncation_rate" in payload:
            metric_suffix.append(f"trunc {self._format_fixed(float(payload['truncation_rate']), 2)}")
        if metric_suffix:
            lines[0] += "  " + "  ".join(metric_suffix)
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
                "accuracy_pass_rate",
                "format_pass_rate",
                "truncation_rate",
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
        self._emit_terminal_line(line)
        self._emit_file_line(line)

    def _emit_console_lines(self, lines: list[str]) -> None:
        for line in lines:
            self._emit_console(line)

    def _emit_terminal_line(self, line: str) -> None:
        """Write one styled line to the terminal only."""
        self._close_terminal_serving_status()
        if not self.config.console:
            return
        print(self._style_terminal_line(line))

    def _emit_file_line(self, line: str) -> None:
        """Write one plain line to console.log only."""
        if not self.config.file:
            return
        with self.console_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

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

    def _format_serving_prompt_gist(self, prompt_text: str) -> str:
        """Render one compact single-line gist for terminal serving status."""
        return self._truncate(prompt_text, limit=PROMPT_GIST_MAX_CHARS)

    def _start_terminal_serving_status(self, line: str) -> None:
        """Start one terminal-only live rollout status line."""
        if not self.config.console:
            return
        if self._should_interactively_update_terminal():
            sys.stdout.write(line)
            sys.stdout.flush()
            self._terminal_serving_status_open = True
            self._terminal_serving_status_length = len(line)
            return
        self._emit_terminal_line(line)

    def _finish_terminal_serving_status(self, line: str) -> None:
        """Finish one terminal-only live rollout status line."""
        if not self.config.console:
            return
        if self._terminal_serving_status_open:
            padding = " " * max(self._terminal_serving_status_length - len(line), 0)
            sys.stdout.write("\r")
            sys.stdout.write(line)
            if padding:
                sys.stdout.write(padding)
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._terminal_serving_status_open = False
            self._terminal_serving_status_length = 0
            return
        self._emit_terminal_line(line)

    def _close_terminal_serving_status(self) -> None:
        """Close any in-place terminal serving status before normal line output."""
        if not self._terminal_serving_status_open or not self.config.console:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._terminal_serving_status_open = False
        self._terminal_serving_status_length = 0

    def _maybe_emit_step_separator(self, step: int) -> None:
        """Insert one visible divider between completed step blocks."""
        if self._current_step_header is None:
            return
        if self._current_step_header == step:
            return
        if not self._step_block_complete:
            return
        self._emit_console(STEP_SEPARATOR)
        self._step_block_complete = False

    def _begin_step_block(self, payload: dict[str, Any]) -> None:
        """Emit the step header once before any rollout or stage detail."""
        step = int(payload["step"])
        self._maybe_emit_step_separator(step)
        if self._current_step_header == step:
            return
        self._emit_console(self._format_step_header(payload))
        self._current_step_header = step
        self._current_serving_prompt_key = None
        self._step_block_complete = False

    def _ensure_step_header(self, payload: dict[str, Any]) -> None:
        """Guarantee the current step header is already visible."""
        self._begin_step_block(payload)

    def _style_terminal_line(self, line: str) -> str:
        """Colorize terminal metadata while keeping file logs plain."""
        if not self._should_color_terminal():
            return line

        if line == "FlashRL training run":
            return self._color(line, "run_header")

        prompt_gist_match = re.match(r"^(  prompt \d+/\d+)(\s+)(.*)$", line)
        if prompt_gist_match:
            label, spacing, value = prompt_gist_match.groups()
            return f"{self._color(label, 'meta_label')}{spacing}{value}"
        rollout_status_match = re.match(r"^(    rollout \d+/\d+)(\s+)(.*)$", line)
        if rollout_status_match:
            label, spacing, value = rollout_status_match.groups()
            return f"{self._color(label, 'serving_candidate')}{spacing}{value}"
        if line in {SECTION_SEPARATOR, STEP_SEPARATOR}:
            return self._color(line, "divider")

        compact_label_match = re.match(
            r"^(  )(run|model|data|runtime|serving|admin|grpo|mapping|progress|logs)(\s+)(.*)$",
            line,
        )
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

    def _should_interactively_update_terminal(self) -> bool:
        """Use in-place serving status updates only for interactive terminals."""
        return self._should_color_terminal()

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
