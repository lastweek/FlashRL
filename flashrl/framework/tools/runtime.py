"""Minimal subprocess-backed runtime for whitebox tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

from flashrl.framework.data_models import Prompt, ToolResult


@dataclass(frozen=True)
class Tool:
    """One built-in tool definition executed through the subprocess runtime."""

    name: str
    description: str
    entrypoint: str
    timeout_seconds: float | None = None
    memory_limit_mb: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool.name must be non-empty.")
        if not self.description.strip():
            raise ValueError("Tool.description must be non-empty.")
        module_name, separator, attr_name = self.entrypoint.partition(":")
        if separator == "" or not module_name or not attr_name:
            raise ValueError(
                "Tool.entrypoint must use the format 'module.submodule:function'."
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0.0:
            raise ValueError("Tool.timeout_seconds must be > 0 when provided.")
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError("Tool.memory_limit_mb must be > 0 when provided.")


def _resource_limiter(memory_limit_mb: int | None, timeout_seconds: float):
    """Build a POSIX-only preexec hook with conservative local limits."""

    def apply_limits() -> None:
        def safe_setrlimit(limit_name: str, soft: int, hard: int) -> None:
            limit = getattr(resource, limit_name, None)
            if limit is None:
                return
            try:
                resource.setrlimit(limit, (soft, hard))
            except (OSError, ValueError):
                return

        cpu_seconds = max(1, int(math.ceil(timeout_seconds)))
        safe_setrlimit("RLIMIT_CPU", cpu_seconds, cpu_seconds)
        safe_setrlimit("RLIMIT_FSIZE", 8 * 1024 * 1024, 8 * 1024 * 1024)
        safe_setrlimit("RLIMIT_NPROC", 16, 16)
        if memory_limit_mb is not None:
            address_space_bytes = int(memory_limit_mb) * 1024 * 1024
            safe_setrlimit("RLIMIT_AS", address_space_bytes, address_space_bytes)

    return apply_limits


class SubprocessToolRuntime:
    """Execute tool functions in short-lived subprocesses with local limits."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        default_timeout_seconds: float = 5.0,
        default_memory_limit_mb: int | None = 256,
    ) -> None:
        if default_timeout_seconds <= 0.0:
            raise ValueError("default_timeout_seconds must be > 0.")
        if default_memory_limit_mb is not None and default_memory_limit_mb <= 0:
            raise ValueError("default_memory_limit_mb must be > 0 when provided.")
        self.python_executable = python_executable or sys.executable
        self.default_timeout_seconds = float(default_timeout_seconds)
        self.default_memory_limit_mb = default_memory_limit_mb
        self.worker_path = Path(__file__).with_name("worker.py")

    def execute(
        self,
        tool: Tool,
        *,
        arguments: dict[str, Any],
        prompt: Prompt,
    ) -> ToolResult:
        """Run one tool call in a subprocess and normalize failures."""
        timeout_seconds = (
            float(tool.timeout_seconds)
            if tool.timeout_seconds is not None
            else self.default_timeout_seconds
        )
        memory_limit_mb = (
            int(tool.memory_limit_mb)
            if tool.memory_limit_mb is not None
            else self.default_memory_limit_mb
        )
        payload = {
            "entrypoint": tool.entrypoint,
            "arguments": dict(arguments),
            "prompt": prompt.model_dump(),
        }
        started_at = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="flashrl-tool-") as temp_dir:
            workdir = Path(temp_dir)
            kwargs = self._build_subprocess_kwargs(
                workdir=workdir,
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
            )
            try:
                completed = subprocess.run(
                    [
                        self.python_executable,
                        "-I",
                        "-B",
                        "-s",
                        str(self.worker_path),
                    ],
                    input=json.dumps(payload),
                    **kwargs,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    content=(
                        f"Tool '{tool.name}' timed out after {timeout_seconds:.2f}s."
                    ),
                    error=True,
                    metadata={
                        "status": "timeout",
                        "tool_name": tool.name,
                        "latency_seconds": time.perf_counter() - started_at,
                    },
                )

        latency_seconds = time.perf_counter() - started_at
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            return ToolResult(
                content=stderr or f"Tool '{tool.name}' exited with code {completed.returncode}.",
                error=True,
                metadata={
                    "status": "subprocess_error",
                    "tool_name": tool.name,
                    "returncode": int(completed.returncode),
                    "latency_seconds": latency_seconds,
                },
            )

        try:
            raw_result = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult(
                content=f"Tool '{tool.name}' returned invalid JSON: {exc}",
                error=True,
                metadata={
                    "status": "invalid_output",
                    "tool_name": tool.name,
                    "latency_seconds": latency_seconds,
                },
            )

        try:
            tool_result = ToolResult.model_validate(raw_result)
        except Exception as exc:
            return ToolResult(
                content=f"Tool '{tool.name}' returned an invalid payload: {exc}",
                error=True,
                metadata={
                    "status": "invalid_output",
                    "tool_name": tool.name,
                    "latency_seconds": latency_seconds,
                },
            )

        metadata = dict(tool_result.metadata)
        metadata.setdefault("status", "ok" if not tool_result.error else "tool_error")
        metadata.setdefault("tool_name", tool.name)
        metadata["latency_seconds"] = latency_seconds
        return tool_result.model_copy(update={"metadata": metadata})

    def _build_subprocess_kwargs(
        self,
        *,
        workdir: Path,
        timeout_seconds: float,
        memory_limit_mb: int | None,
    ) -> dict[str, object]:
        """Build one subprocess configuration shared by all tool calls."""
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        kwargs: dict[str, object] = {
            "cwd": str(workdir),
            "env": env,
            "text": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "timeout": timeout_seconds,
        }
        if os.name == "posix":
            kwargs["preexec_fn"] = _resource_limiter(memory_limit_mb, timeout_seconds)
        return kwargs
