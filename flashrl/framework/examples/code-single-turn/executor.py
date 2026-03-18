"""Local Python execution helpers for the reasoning-code example."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time


@dataclass
class ExecutionResult:
    """Compact execution outcome for one generated solution."""

    passed_tests: int
    total_tests: int
    pass_rate: float
    execution_seconds: float
    failure_reason: str | None
    checker_used: bool


def _normalize_output(text: str) -> str:
    """Normalize line endings and trailing whitespace for direct comparisons."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


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


def _build_subprocess_kwargs(
    *,
    workdir: Path,
    timeout_seconds: float,
    memory_limit_mb: int | None,
) -> dict[str, object]:
    """Build one subprocess configuration shared by solutions and checkers."""
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


def _run_checker(
    checker_code: str,
    *,
    workdir: Path,
    test_input: str,
    expected_output: str,
    actual_output: str,
    timeout_seconds: float,
    memory_limit_mb: int | None,
) -> tuple[bool, str | None]:
    """Run one dataset-provided Python checker against the current output."""
    checker_path = workdir / "checker.py"
    input_path = workdir / "checker-input.txt"
    expected_path = workdir / "checker-expected.txt"
    actual_path = workdir / "checker-actual.txt"
    checker_path.write_text(checker_code, encoding="utf-8")
    input_path.write_text(test_input, encoding="utf-8")
    expected_path.write_text(expected_output, encoding="utf-8")
    actual_path.write_text(actual_output, encoding="utf-8")

    if shutil.which(sys.executable) is None:  # pragma: no cover - defensive in live usage
        return False, "python_unavailable"

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-s",
                str(checker_path),
                str(input_path),
                str(expected_path),
                str(actual_path),
            ],
            **_build_subprocess_kwargs(
                workdir=workdir,
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
            ),
        )
    except subprocess.TimeoutExpired:
        return False, "checker_timeout"

    if completed.returncode != 0:
        return False, "checker_error"

    first_token = (completed.stdout or "").strip().split()
    if not first_token:
        return False, "checker_empty"
    try:
        score = float(first_token[0])
    except ValueError:
        return False, "checker_parse_error"
    return score > 0.0, None


def run_python_solution(
    code: str,
    *,
    official_tests: list[dict[str, str]],
    checker_code: str | None,
    timeout_seconds: float,
    memory_limit_mb: int | None,
) -> ExecutionResult:
    """Run one generated Python program against the selected official tests."""
    if not official_tests:
        return ExecutionResult(
            passed_tests=0,
            total_tests=0,
            pass_rate=0.0,
            execution_seconds=0.0,
            failure_reason="no_tests",
            checker_used=False,
        )

    started_at = time.perf_counter()
    passed_tests = 0
    checker_used = False
    failure_reason: str | None = None

    with tempfile.TemporaryDirectory(prefix="flashrl-code-") as temp_dir:
        workdir = Path(temp_dir)
        solution_path = workdir / "solution.py"
        solution_path.write_text(code, encoding="utf-8")

        for test_case in official_tests:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-B", "-s", str(solution_path)],
                    input=str(test_case["input"]),
                    **_build_subprocess_kwargs(
                        workdir=workdir,
                        timeout_seconds=timeout_seconds,
                        memory_limit_mb=memory_limit_mb,
                    ),
                )
            except subprocess.TimeoutExpired:
                failure_reason = "timeout"
                break

            if completed.returncode != 0:
                stderr = completed.stderr or ""
                failure_reason = "syntax_error" if "SyntaxError" in stderr else "runtime_error"
                break

            actual_output = completed.stdout or ""
            expected_output = str(test_case["output"])
            if checker_code:
                checker_used = True
                passed, checker_failure = _run_checker(
                    checker_code,
                    workdir=workdir,
                    test_input=str(test_case["input"]),
                    expected_output=expected_output,
                    actual_output=actual_output,
                    timeout_seconds=timeout_seconds,
                    memory_limit_mb=memory_limit_mb,
                )
                if not passed:
                    failure_reason = checker_failure or "wrong_answer"
                    break
            elif _normalize_output(actual_output) != _normalize_output(expected_output):
                failure_reason = "wrong_answer"
                break

            passed_tests += 1

    if failure_reason is None and passed_tests == len(official_tests):
        failure_reason = None

    total_tests = len(official_tests)
    return ExecutionResult(
        passed_tests=passed_tests,
        total_tests=total_tests,
        pass_rate=(passed_tests / total_tests) if total_tests else 0.0,
        execution_seconds=time.perf_counter() - started_at,
        failure_reason=failure_reason,
        checker_used=checker_used,
    )
