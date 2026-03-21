"""Shared FastAPI HTTP helpers for distributed FlashRL services."""

from __future__ import annotations

from collections import deque
from math import ceil
from threading import Lock
import time
from typing import Callable

from fastapi import FastAPI, HTTPException, Request

from flashrl.framework.admin.objects import build_admin_object, build_admin_object_list, utc_now_iso
from flashrl.framework.distributed.models import ComponentStatus


class ServiceRuntimeState:
    """Track pod-local readiness, drain state, and request load."""

    def __init__(
        self,
        *,
        queue_depth_getter: Callable[[], int] | None = None,
        sample_limit: int = 256,
    ) -> None:
        self._queue_depth_getter = queue_depth_getter or (lambda: 0)
        self._latencies: deque[float] = deque(maxlen=sample_limit)
        self._inflight_requests = 0
        self._draining = False
        self._last_observed_at = utc_now_iso()
        self._lock = Lock()

    def begin_request(self) -> None:
        with self._lock:
            self._inflight_requests += 1
            self._last_observed_at = utc_now_iso()

    def finish_request(self, *, elapsed_seconds: float) -> None:
        with self._lock:
            self._inflight_requests = max(self._inflight_requests - 1, 0)
            self._latencies.append(max(float(elapsed_seconds), 0.0))
            self._last_observed_at = utc_now_iso()

    def set_draining(self, draining: bool) -> None:
        with self._lock:
            self._draining = bool(draining)
            self._last_observed_at = utc_now_iso()

    def drain(self, *, wait_seconds: float) -> dict[str, object]:
        self.set_draining(True)
        deadline = time.monotonic() + max(wait_seconds, 0.0)
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if int(snapshot["inflightRequests"]) == 0:
                return snapshot
            time.sleep(0.05)
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latencies = list(self._latencies)
            inflight_requests = int(self._inflight_requests)
            draining = bool(self._draining)
            last_observed_at = self._last_observed_at
        return {
            "inflightRequests": inflight_requests,
            "queueDepth": max(int(self._queue_depth_getter()), 0),
            "p95LatencySeconds": _p95_latency(latencies),
            "draining": draining,
            "lastObservedAt": last_observed_at,
        }

    def ready(self, base_status: ComponentStatus) -> bool:
        return bool(base_status.healthy) and not bool(self.snapshot()["draining"])


def _p95_latency(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    values = sorted(float(value) for value in latencies)
    index = max(ceil(len(values) * 0.95) - 1, 0)
    return float(values[index])


def install_common_routes(
    app: FastAPI,
    *,
    status_getter,
    kind: str,
    name: str,
    drainable: bool = False,
    queue_depth_getter: Callable[[], int] | None = None,
) -> None:
    """Install shared health, readiness, status, and admin routes."""

    runtime_state = ServiceRuntimeState(queue_depth_getter=queue_depth_getter)
    control_paths = {
        "/healthz",
        "/readyz",
        "/v1/status",
        "/admin/v1/objects",
        "/v1/lifecycle/drain",
    }

    def _enriched_status() -> ComponentStatus:
        status = status_getter().model_copy(deep=True)
        if drainable:
            status.metadata.update(runtime_state.snapshot())
            status.healthy = status.healthy and not bool(status.metadata["draining"])
            if bool(status.metadata["draining"]) and status.phase == "Ready":
                status.phase = "Draining"
        return status

    @app.middleware("http")
    async def track_load(request: Request, call_next):
        track = request.url.path not in control_paths
        if track:
            runtime_state.begin_request()
        started_at = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            if track:
                runtime_state.finish_request(elapsed_seconds=time.perf_counter() - started_at)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        status = _enriched_status()
        if drainable and not runtime_state.ready(status_getter()):
            raise HTTPException(status_code=503, detail="draining")
        if not status.healthy:
            raise HTTPException(status_code=503, detail="not_ready")
        return {"status": "ok"}

    @app.get("/v1/status")
    def status():
        return {"status": _enriched_status().model_dump()}

    if drainable:

        @app.post("/v1/lifecycle/drain")
        def drain(wait_seconds: float = 25.0) -> dict[str, object]:
            snapshot = runtime_state.drain(wait_seconds=wait_seconds)
            return {
                "status": "draining",
                **snapshot,
            }

    @app.get("/admin/v1/objects")
    def admin_objects():
        status = _enriched_status()
        item = build_admin_object(
            kind,
            name,
            uid=f"{kind.lower()}:{name}",
            created_at=utc_now_iso(),
            spec={"component": name},
            status={
                "phase": status.phase,
                "healthy": status.healthy,
                "readyReplicaCount": status.ready_replica_count,
                "desiredReplicaCount": status.desired_replica_count,
                "activeWeightVersion": (
                    status.active_weight_version.model_dump()
                    if status.active_weight_version is not None
                    else None
                ),
                "lastError": status.last_error,
                **dict(status.metadata),
            },
        )
        return build_admin_object_list([item])


def unwrap_status(payload: dict[str, object]) -> ComponentStatus:
    """Parse a shared ``/v1/status`` payload."""
    return ComponentStatus.model_validate(payload["status"])
