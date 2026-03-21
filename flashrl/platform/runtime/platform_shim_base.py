"""Tiny base class for the platform shim layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fastapi import FastAPI
import uvicorn


class PlatformShim(ABC):
    """Base class for one platform-owned pod bootstrap layer."""

    def __init__(self, *, job_path: str | Path | None = None) -> None:
        self._job_path = job_path

    @abstractmethod
    def create_app(self) -> FastAPI:
        """Build the FastAPI app for one workload pod."""

    def run(self, *, host: str = "0.0.0.0", port: int = 8000) -> int:
        """Start the pod HTTP server for this shim."""
        uvicorn.run(self.create_app(), host=host, port=port, log_level="info")
        return 0
