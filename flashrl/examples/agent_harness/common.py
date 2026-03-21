"""Shared config and runtime helpers for the agent harness examples."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

from flashrl.examples.agent_harness.config import AgentHarnessConfig

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXAMPLE_DIR / "config.yaml"


def find_default_vllm_python() -> str | None:
    candidates: list[Path] = []
    if sys.platform == "darwin" and os.uname().machine == "arm64":
        candidates.append(Path.home() / ".venv-vllm-metal" / "bin" / "python")
    candidates.append(Path.home() / ".venv-vllm" / "bin" / "python")
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    current_python = Path(sys.executable)
    sibling_vllm = shutil.which("vllm", path=str(current_python.parent))
    if sibling_vllm is None:
        return None
    try:
        __import__("vllm")
    except Exception:
        return None
    return str(current_python)


def load_yaml_mapping(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {config_path}.")
    return payload


def prepare_environment(config_path: str | Path) -> None:
    if os.environ.get("FLASHRL_VLLM_PYTHON"):
        return
    if not _config_uses_vllm(config_path):
        return
    runtime_python = find_default_vllm_python()
    if runtime_python is not None:
        os.environ["FLASHRL_VLLM_PYTHON"] = runtime_python


def load_example_config(config_path: str | Path) -> tuple[AgentHarnessConfig, dict[str, Any]]:
    payload = load_yaml_mapping(config_path)
    example = payload.get("example") or {}
    if not isinstance(example, dict):
        raise ValueError("config.example must be a mapping when present.")
    dataset = example.get("dataset") or {}
    if not isinstance(dataset, dict):
        raise ValueError("config.example.dataset must be a mapping when present.")
    harness = AgentHarnessConfig.from_mapping(example.get("harness"))
    return harness, dataset


def resolve_dataset_limit(
    explicit_limit: int | None,
    dataset_config: dict[str, Any],
    key: str,
) -> int | None:
    if explicit_limit is not None:
        return explicit_limit
    raw_limit = dataset_config.get(key)
    return int(raw_limit) if isinstance(raw_limit, int) else None


def _config_uses_vllm(config_path: str | Path) -> bool:
    payload = load_yaml_mapping(config_path)
    framework = payload.get("framework")
    if not isinstance(framework, dict):
        return False
    serving = framework.get("serving")
    if not isinstance(serving, dict):
        return False
    return serving.get("backend") == "vllm"
