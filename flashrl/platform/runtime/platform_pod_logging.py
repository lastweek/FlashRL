"""Platform pod logging helpers for durable job-scoped observability."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

from flashrl.framework.memory import capture_memory_snapshot
from flashrl.platform.k8s.job import FlashRLJob
from flashrl.platform.runtime.platform_shim_common import (
    component_log_metadata,
    pod_name_for,
    resolve_component_log_dir,
    resolve_job_log_root,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class PlatformPodLogger:
    """Dual-file pod logger used by all platform shim and service pods."""

    def __init__(self, *, job: FlashRLJob, component: str) -> None:
        self._job = job
        self._component = component
        self._job_log_root = resolve_job_log_root(job)
        self._component_log_dir = resolve_component_log_dir(job, component)
        self._console_path = self._component_log_dir / "console.log"
        self._events_path = self._component_log_dir / "events.jsonl"
        self._configured = False

    @property
    def job_log_root(self) -> Path:
        return self._job_log_root

    @property
    def component_log_dir(self) -> Path:
        return self._component_log_dir

    def configure_python_logging(self) -> None:
        """Configure root and uvicorn loggers to mirror to console.log."""
        if self._configured:
            return
        self._component_log_dir.mkdir(parents=True, exist_ok=True)
        self._events_path.touch(exist_ok=True)
        handlers = [logging.StreamHandler(sys.stdout)]
        handlers.append(logging.FileHandler(self._console_path, encoding="utf-8"))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=handlers,
            force=True,
        )
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
            logger = logging.getLogger(logger_name)
            logger.handlers = []
            logger.propagate = True
            logger.setLevel(logging.INFO)
        self._configured = True
        self.emit(
            "pod_logger_configured",
            message="Configured platform pod log files.",
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, str]:
        """Return stable log-discovery metadata for status payloads."""
        return component_log_metadata(self._job, self._component)

    def emit(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        message: str | None = None,
        level: str = "INFO",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one structured event and mirror it through Python logging."""
        merged_metadata = {**dict(payload or {}), **dict(metadata or {})}
        resolved_message = message
        if resolved_message is None:
            resolved_message = str(merged_metadata.pop("message", event))
        payload = {
            "timestamp": _utc_now(),
            "event": event,
            "level": level,
            "message": resolved_message,
            "job": self._job.name,
            "component": self._component,
            "pod": pod_name_for(self._component),
            "metadata": merged_metadata,
        }
        self._component_log_dir.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        logging.getLogger(f"flashrl.platform.{self._component}").log(
            getattr(logging, level.upper(), logging.INFO),
            "%s | %s",
            event,
            resolved_message,
        )

    def emit_request(
        self,
        *,
        path: str,
        method: str,
        latency_seconds: float,
        status_code: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one completed RPC request."""
        self.emit(
            "request_completed",
            message=f"{method} {path} status={status_code} latency={latency_seconds:.3f}s",
            metadata={
                "method": method,
                "path": path,
                "statusCode": int(status_code),
                "latencySeconds": float(latency_seconds),
                **dict(metadata or {}),
            },
        )

    def emit_exception(self, exc: BaseException, *, stage: str) -> None:
        """Record one uncaught exception."""
        memory_snapshot = getattr(exc, "memory_snapshot", None)
        if not isinstance(memory_snapshot, dict):
            memory_snapshot = capture_memory_snapshot(None)
        self.emit(
            "exception",
            level="ERROR",
            message=f"{type(exc).__name__}: {exc}",
            metadata={
                "stage": stage,
                "memory": memory_snapshot,
                "memoryReasonTags": list(getattr(exc, "reason_tags", []) or []),
            },
        )
