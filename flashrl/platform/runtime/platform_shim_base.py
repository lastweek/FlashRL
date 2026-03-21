"""Tiny base class for the platform shim layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fastapi import FastAPI
import uvicorn


class PlatformShim(ABC):
    """Base class for one platform-owned pod bootstrap layer."""

    def __init__(self, *, job_path: str | Path | None = None) -> None:
        self._job_path = job_path
        self._pod_logger: Any | None = None

    @abstractmethod
    def create_app(self) -> FastAPI:
        """Build the FastAPI app for one workload pod."""

    def run(self, *, host: str = "0.0.0.0", port: int = 8000) -> int:
        """Start the pod HTTP server for this shim."""
        app = self.create_app()
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except Exception as exc:
            if self._pod_logger is not None:
                self._pod_logger.emit_exception(exc, stage="uvicorn.run")
            raise
        return 0
