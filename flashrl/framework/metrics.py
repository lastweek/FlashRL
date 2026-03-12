"""Prometheus metrics sink for local FlashRL runs."""

from __future__ import annotations

from typing import Callable
import warnings

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from flashrl.framework.config import MetricsConfig


LABEL_NAMES = ("model", "algorithm", "runtime")


class PrometheusMetricsSink:
    """Pushgateway-backed Prometheus metrics for the local framework path."""

    def __init__(
        self,
        config: MetricsConfig,
        *,
        model_name: str,
        algorithm: str = "grpo",
        runtime: str = "framework_local",
        push_fn: Callable[..., None] | None = None,
    ) -> None:
        """Initialize the registry and gauges for one FlashRL process."""
        if config.backend != "pushgateway":
            raise ValueError(f"Unsupported metrics backend: {config.backend}")

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

    def observe_stage(self, stage: str, latency_seconds: float) -> None:
        """Update the latency gauges for stages we track in v1."""
        if stage == "rollout":
            self._rollout_latency_seconds.labels(**self.labels).set(latency_seconds)
        elif stage == "reward":
            self._reward_latency_seconds.labels(**self.labels).set(latency_seconds)

    def observe_step(
        self,
        *,
        loss: float,
        reward_mean: float,
        kl_mean: float,
        step_duration_seconds: float,
    ) -> None:
        """Update the scalar gauges for one completed training step."""
        self._train_loss.labels(**self.labels).set(loss)
        self._reward_mean.labels(**self.labels).set(reward_mean)
        self._kl_mean.labels(**self.labels).set(kl_mean)
        self._step_duration_seconds.labels(**self.labels).set(step_duration_seconds)

    def push(self) -> None:
        """Push the current registry to Pushgateway on a best-effort basis."""
        try:
            self.push_fn(
                self.config.pushgateway_url,
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
