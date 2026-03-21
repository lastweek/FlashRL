"""Metrics sinks for local FlashRL runs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from typing import Any, Callable, Protocol
import warnings

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except ImportError:  # pragma: no cover - depends on optional local metrics deps
    CollectorRegistry = None
    Gauge = None
    push_to_gateway = None

from flashrl.framework.config import (
    MetricsConfig,
    PushgatewayMetricsConfig,
    TensorBoardMetricsConfig,
)
from flashrl.framework.memory import extract_memory_counters
from flashrl.framework.observability import RuntimeEvent, observe_event


LABEL_NAMES = ("model", "algorithm", "runtime")


class MetricsSink(Protocol):
    """Runtime metrics sink contract shared by all backends."""

    def start_run(self, *, run_dir: Path, run_id: str) -> None:
        """Prepare the sink for a new run."""

    def observe_event(self, event: RuntimeEvent) -> None:
        """Observe one typed runtime event."""

    def observe_stage(self, payload: dict[str, Any]) -> None:
        """Observe one completed training stage."""

    def observe_step(self, payload: dict[str, Any]) -> None:
        """Observe one completed training step."""

    def observe_serving_debug(self, payload: dict[str, Any]) -> None:
        """Observe one serving-debug completion payload."""

    def push(self) -> None:
        """Flush or publish any staged metrics."""

    def finish_run(self) -> None:
        """Finish the current run."""

    def close(self) -> None:
        """Release sink-owned resources."""


class CompositeMetricsSink:
    """Forward metrics events to multiple concrete sinks."""

    def __init__(self, sinks: list[MetricsSink]) -> None:
        self.sinks = list(sinks)

    def start_run(self, *, run_dir: Path, run_id: str) -> None:
        for sink in self.sinks:
            sink.start_run(run_dir=run_dir, run_id=run_id)

    def observe_event(self, event: RuntimeEvent) -> None:
        for sink in self.sinks:
            observe_event(sink, event)

    def observe_stage(self, payload: dict[str, Any]) -> None:
        self.observe_event(RuntimeEvent(kind="step_stage", payload=payload))

    def observe_step(self, payload: dict[str, Any]) -> None:
        self.observe_event(RuntimeEvent(kind="step_done", payload=payload))

    def observe_serving_debug(self, payload: dict[str, Any]) -> None:
        self.observe_event(RuntimeEvent(kind="serving_debug_done", payload=payload))

    def push(self) -> None:
        for sink in self.sinks:
            sink.push()

    def finish_run(self) -> None:
        for sink in self.sinks:
            sink.finish_run()

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def build_metrics_sink(
    config: MetricsConfig,
    *,
    model_name: str,
    algorithm: str = "grpo",
    runtime: str = "framework_local",
) -> MetricsSink | None:
    """Build the enabled metrics sinks for one FlashRL runtime."""
    if not config.enabled:
        return None

    sinks: list[MetricsSink] = []
    if config.tensorboard.enabled:
        sinks.append(TensorBoardMetricsSink(config.tensorboard))
    if config.pushgateway.enabled:
        sinks.append(
            PrometheusMetricsSink(
                config.pushgateway,
                model_name=model_name,
                algorithm=algorithm,
                runtime=runtime,
            )
        )

    if not sinks:
        return None
    if len(sinks) == 1:
        return sinks[0]
    return CompositeMetricsSink(sinks)


def _parse_version_parts(version_text: str) -> tuple[int, ...]:
    """Parse numeric version components conservatively."""
    parts = [int(part) for part in re.findall(r"\d+", version_text)]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _load_summary_writer_class() -> type[Any]:
    """Load a usable TensorBoard SummaryWriter implementation."""
    try:
        tensorboard_version = version("tensorboard")
    except PackageNotFoundError as exc:
        raise ImportError(
            "TensorBoard metrics require the 'tensorboard' package (>=2.0.0)."
        ) from exc

    if _parse_version_parts(tensorboard_version) < (2, 0, 0):
        raise ImportError(
            "TensorBoard metrics require tensorboard>=2.0.0; "
            f"found tensorboard=={tensorboard_version}."
        )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard metrics require a working torch.utils.tensorboard "
            f"setup with tensorboard>=2.0.0; found tensorboard=={tensorboard_version}."
        ) from exc

    return SummaryWriter


def _create_summary_writer(log_dir: str) -> Any:
    """Instantiate the default TensorBoard writer."""
    summary_writer_cls = _load_summary_writer_class()
    return summary_writer_cls(log_dir=log_dir)


class TensorBoardMetricsSink:
    """TensorBoard-backed scalar metrics sink."""

    def __init__(
        self,
        config: TensorBoardMetricsConfig,
        *,
        writer_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.writer_factory = writer_factory or _create_summary_writer
        self._writer: Any | None = None

    def start_run(self, *, run_dir: Path, run_id: str) -> None:
        """Open a per-run TensorBoard event writer in the run root."""
        del run_id
        self.finish_run()
        self._writer = self.writer_factory(str(run_dir))

    def observe_event(self, event: RuntimeEvent) -> None:
        """Record one typed runtime event."""
        if event.kind == "step_stage":
            self.observe_stage(event.payload)
            return
        if event.kind == "step_done":
            self.observe_step(event.payload)
            return
        if event.kind == "serving_debug_done":
            self.observe_serving_debug(event.payload)

    def observe_stage(self, payload: dict[str, Any]) -> None:
        """Record stage-specific TensorBoard scalars."""
        writer = self._writer
        if writer is None:
            return

        step = int(payload["step"])
        stage = str(payload["stage"])
        self._add_scalar(writer, f"timing/stage/{stage}_seconds", payload["latency_seconds"], step)
        self._add_memory_scalars(
            writer,
            step=step,
            snapshot=(
                payload.get("memory", {}).get("after")
                if isinstance(payload.get("memory"), dict)
                else None
            ),
            prefix=f"memory/stage/{stage}",
        )

        if stage == "rollout":
            self._add_scalar(writer, "tokens/prompt_mean", payload["prompt_tokens_mean"], step)
            self._add_scalar(writer, "tokens/prompt_max", payload["prompt_tokens_max"], step)
            self._add_scalar(writer, "tokens/response_mean", payload["response_tokens_mean"], step)
            self._add_scalar(writer, "tokens/response_max", payload["response_tokens_max"], step)

            # Log LLM call rounds
            self._add_scalar(writer, "rollout/llm_call_rounds", payload.get("llm_call_rounds"), step)

            # Log tool calls
            self._add_scalar(writer, "rollout/tool_calls_total", payload.get("tool_calls_total"), step)

            return

        if stage == "reward":
            self._add_scalar(writer, "reward/mean", payload["reward_mean"], step)
            self._add_scalar(writer, "reward/std", payload["reward_std"], step)
            self._add_scalar(writer, "reward/min", payload["reward_min"], step)
            self._add_scalar(writer, "reward/max", payload["reward_max"], step)
            self._add_scalar(writer, "reward/accuracy_pass_rate", payload.get("accuracy_pass_rate"), step)
            self._add_scalar(writer, "reward/format_pass_rate", payload.get("format_pass_rate"), step)
            self._add_scalar(writer, "reward/truncation_rate", payload.get("truncation_rate"), step)
            self._add_scalar(
                writer,
                "timing/reward_per_item_mean_seconds",
                payload.get("reward_per_item_mean_seconds"),
                step,
            )
            return

        if stage == "advantage":
            self._add_scalar(writer, "advantage/mean", payload["advantage_mean"], step)
            self._add_scalar(writer, "advantage/std", payload["advantage_std"], step)
            self._add_scalar(writer, "advantage/min", payload["advantage_min"], step)
            self._add_scalar(writer, "advantage/max", payload["advantage_max"], step)
            return

        if stage == "prepare_inputs":
            self._add_scalar(writer, "tokens/full_mean", payload["full_tokens_mean"], step)
            self._add_scalar(writer, "tokens/full_max", payload["full_tokens_max"], step)
            self._add_scalar(writer, "tokens/response_total", payload["response_tokens_total"], step)
            return

        if stage == "loss_assembly":
            self._add_scalar(
                writer,
                "importance_sampling_ratio/mean",
                payload["importance_sampling_ratio_mean"],
                step,
            )
            self._add_scalar(
                writer,
                "importance_sampling_ratio/std",
                payload["importance_sampling_ratio_std"],
                step,
            )
            self._add_scalar(
                writer,
                "importance_sampling_ratio/min",
                payload["importance_sampling_ratio_min"],
                step,
            )
            self._add_scalar(
                writer,
                "importance_sampling_ratio/max",
                payload["importance_sampling_ratio_max"],
                step,
            )
            self._add_scalar(
                writer,
                "importance_sampling_ratio/clip_fraction",
                payload["clip_fraction"],
                step,
            )
            return

        if stage == "optimizer":
            self._add_scalar(writer, "train/learning_rate", payload["learning_rate"], step)

    def observe_step(self, payload: dict[str, Any]) -> None:
        """Record per-step TensorBoard scalars."""
        writer = self._writer
        if writer is None:
            return

        step = int(payload["step"])
        self._add_scalar(writer, "train/loss", payload["loss"], step)
        self._add_scalar(writer, "train/policy_loss", payload["policy_loss"], step)
        self._add_scalar(writer, "train/kl_divergence", payload["kl_divergence"], step)
        self._add_scalar(writer, "train/tokens_per_second", payload["tokens_per_second"], step)
        self._add_scalar(writer, "timing/step_duration_seconds", payload["step_duration_seconds"], step)
        memory_summary = payload.get("memory_summary")
        if isinstance(memory_summary, dict):
            self._add_memory_scalars(
                writer,
                step=step,
                snapshot=memory_summary.get("end"),
                prefix="memory/step/end",
            )
            self._add_scalar(
                writer,
                "memory/step/peak_process_rss_bytes",
                memory_summary.get("peak_process_rss_bytes"),
                step,
            )
            self._add_scalar(
                writer,
                "memory/step/lowest_system_available_bytes",
                memory_summary.get("lowest_system_available_bytes"),
                step,
            )
            self._add_scalar(
                writer,
                "memory/step/peak_device_current_allocated_bytes",
                memory_summary.get("peak_device_current_allocated_bytes"),
                step,
            )
            self._add_scalar(
                writer,
                "memory/step/peak_device_driver_allocated_bytes",
                memory_summary.get("peak_device_driver_allocated_bytes"),
                step,
            )

    def observe_serving_debug(self, payload: dict[str, Any]) -> None:
        """Record serving debug timings for the current step."""
        writer = self._writer
        if writer is None:
            return

        step = int(payload.get("step", 0))
        self._add_scalar(writer, "serving/ttft_seconds", payload["ttft_seconds"], step)
        self._add_scalar(writer, "serving/tpot_seconds", payload["tpot_seconds"], step)

    def push(self) -> None:
        """Flush the current TensorBoard writer."""
        if self._writer is not None:
            self._writer.flush()

    def finish_run(self) -> None:
        """Close the current run writer, if any."""
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None

    def close(self) -> None:
        """Release any writer resources."""
        self.finish_run()

    def _add_scalar(self, writer: Any, tag: str, value: Any, step: int) -> None:
        """Write one scalar when the payload field is present."""
        if value is None:
            return
        writer.add_scalar(tag, float(value), global_step=step)

    def _add_memory_scalars(
        self,
        writer: Any,
        *,
        step: int,
        snapshot: dict[str, Any] | None,
        prefix: str,
    ) -> None:
        counters = extract_memory_counters(snapshot)
        if not counters:
            return
        self._add_scalar(writer, f"{prefix}/process_rss_bytes", counters.get("process_rss_bytes"), step)
        self._add_scalar(
            writer,
            f"{prefix}/system_available_bytes",
            counters.get("system_available_bytes"),
            step,
        )
        self._add_scalar(
            writer,
            f"{prefix}/device_current_allocated_bytes",
            counters.get("device_current_allocated_bytes"),
            step,
        )
        self._add_scalar(
            writer,
            f"{prefix}/device_driver_allocated_bytes",
            counters.get("device_driver_allocated_bytes"),
            step,
        )
        self._add_scalar(
            writer,
            f"{prefix}/device_reserved_bytes",
            counters.get("device_reserved_bytes"),
            step,
        )
        self._add_scalar(
            writer,
            f"{prefix}/device_recommended_max_bytes",
            counters.get("device_recommended_max_bytes"),
            step,
        )


class PrometheusMetricsSink:
    """Pushgateway-backed Prometheus metrics for the local framework path."""

    def __init__(
        self,
        config: PushgatewayMetricsConfig,
        *,
        model_name: str,
        algorithm: str = "grpo",
        runtime: str = "framework_local",
        push_fn: Callable[..., None] | None = None,
    ) -> None:
        """Initialize the registry and gauges for one FlashRL process."""
        if CollectorRegistry is None or Gauge is None or push_to_gateway is None:
            raise RuntimeError(
                "prometheus_client is required when Pushgateway metrics are enabled."
            )
        self.config = config
        self.registry = CollectorRegistry()
        self.push_fn = push_fn or push_to_gateway
        self.labels = {
            "model": model_name,
            "algorithm": algorithm,
            "runtime": runtime,
        }
        self._push_warning_emitted = False

        self._train_loss = Gauge(
            "flashrl_train_loss",
            "Current FlashRL train loss.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._reward_mean = Gauge(
            "flashrl_reward_mean",
            "Current FlashRL reward mean.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._kl_mean = Gauge(
            "flashrl_kl_mean",
            "Current FlashRL KL mean.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._rollout_latency_seconds = Gauge(
            "flashrl_rollout_latency_seconds",
            "Current FlashRL rollout latency in seconds.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._reward_latency_seconds = Gauge(
            "flashrl_reward_latency_seconds",
            "Current FlashRL reward latency in seconds.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._step_duration_seconds = Gauge(
            "flashrl_step_duration_seconds",
            "Current FlashRL end-to-end step duration in seconds.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._serving_ttft_seconds = Gauge(
            "flashrl_serving_ttft_seconds",
            "Current FlashRL serving time-to-first-token in seconds.",
            LABEL_NAMES,
            registry=self.registry,
        )
        self._serving_tpot_seconds = Gauge(
            "flashrl_serving_tpot_seconds",
            "Current FlashRL serving time-per-output-token in seconds.",
            LABEL_NAMES,
            registry=self.registry,
        )

    def start_run(self, *, run_dir: Path, run_id: str) -> None:
        """Prometheus registry is process-scoped and needs no per-run setup."""
        del run_dir, run_id

    def observe_event(self, event: RuntimeEvent) -> None:
        """Update gauges from one typed runtime event."""
        if event.kind == "step_stage":
            self.observe_stage(event.payload)
            return
        if event.kind == "step_done":
            self.observe_step(event.payload)
            return
        if event.kind == "serving_debug_done":
            self.observe_serving_debug(event.payload)

    def observe_stage(self, payload: dict[str, Any]) -> None:
        """Update the latency gauges for stages tracked in v1."""
        stage = str(payload["stage"])
        latency_seconds = float(payload["latency_seconds"])
        if stage == "rollout":
            self._rollout_latency_seconds.labels(**self.labels).set(latency_seconds)
        elif stage == "reward":
            self._reward_latency_seconds.labels(**self.labels).set(latency_seconds)

    def observe_step(self, payload: dict[str, Any]) -> None:
        """Update the scalar gauges for one completed training step."""
        self._train_loss.labels(**self.labels).set(float(payload["loss"]))
        self._reward_mean.labels(**self.labels).set(float(payload["reward_mean"]))
        self._kl_mean.labels(**self.labels).set(float(payload["kl_divergence"]))
        self._step_duration_seconds.labels(**self.labels).set(
            float(payload["step_duration_seconds"])
        )

    def observe_serving_debug(self, payload: dict[str, Any]) -> None:
        """Update serving-debug gauges for one completed candidate generation."""
        self._serving_ttft_seconds.labels(**self.labels).set(float(payload["ttft_seconds"]))
        self._serving_tpot_seconds.labels(**self.labels).set(float(payload["tpot_seconds"]))

    def push(self) -> None:
        """Push the current registry to Pushgateway on a best-effort basis."""
        try:
            self.push_fn(
                self.config.url,
                job=self.config.job_name,
                registry=self.registry,
            )
        except Exception as exc:
            if not self._push_warning_emitted:
                warnings.warn(
                    (
                        "FlashRL metrics push failed and metrics will continue in "
                        f"best-effort mode: {exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._push_warning_emitted = True

    def finish_run(self) -> None:
        """Prometheus registry persists across runs and needs no teardown."""

    def close(self) -> None:
        """Prometheus registry has no external resources to close."""
