"""Shared timing and event helpers for FlashRL internals."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Protocol, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeEvent:
    """One typed runtime event with a JSON-serializable payload."""

    kind: str
    payload: dict[str, Any]


@dataclass
class StageResult:
    """One measured training stage and its derived metrics."""

    name: str
    seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return the stage payload in the legacy logger/metrics shape."""
        return {
            "stage": self.name,
            "latency_seconds": float(self.seconds),
            **self.metrics,
        }


class EventObserver(Protocol):
    """Shared runtime event observer contract."""

    def observe_event(self, event: RuntimeEvent) -> None:
        """Consume one runtime event."""


def timed_call(operation: Callable[[], T]) -> tuple[T, float]:
    """Run one callable and return the result plus elapsed seconds."""
    started_at = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started_at


def elapsed_seconds(started_at: float) -> float:
    """Return the elapsed wall-clock seconds from one perf-counter start."""
    return time.perf_counter() - started_at


def observe_event(observer: EventObserver | None, event: RuntimeEvent) -> None:
    """Send one event to an observer when present."""
    if observer is None:
        return
    observe_method = getattr(observer, "observe_event", None)
    if observe_method is not None:
        observe_method(event)
        return

    # Keep compatibility with older sink interfaces used in tests.
    if event.kind == "step_stage" and hasattr(observer, "observe_stage"):
        observer.observe_stage(event.payload)
        return
    if event.kind == "step_done" and hasattr(observer, "observe_step"):
        observer.observe_step(event.payload)
        return
    if event.kind == "serving_debug_done" and hasattr(observer, "observe_serving_debug"):
        observer.observe_serving_debug(event.payload)


def observe_event_pair(
    first: EventObserver | None,
    second: EventObserver | None,
    event: RuntimeEvent,
) -> None:
    """Send one event to two observers in a stable order."""
    observe_event(first, event)
    observe_event(second, event)


def stage_timings(stages: list[StageResult]) -> dict[str, float]:
    """Return a mapping of stage name to measured seconds."""
    return {stage.name: float(stage.seconds) for stage in stages}


def stage_metrics(stages: list[StageResult]) -> dict[str, dict[str, Any]]:
    """Return a mapping of stage name to its metrics payload."""
    return {stage.name: dict(stage.metrics) for stage in stages}


def dominant_stage_name(stages: list[StageResult]) -> str:
    """Return the longest stage name or ``n/a`` when none exist."""
    if not stages:
        return "n/a"
    return max(stages, key=lambda stage: stage.seconds).name
