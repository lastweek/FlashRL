"""Train the reference agent harness example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_CONFIG_PATH = EXAMPLE_DIR / "config.yaml"
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from flashrl.examples.agent_harness.config import CodingHarnessConfig
from flashrl.examples.agent_harness.harness import (
    build_coding_agent,
    build_coding_reward_fn,
    build_coding_train_dataset,
)
from flashrl.framework import FlashRL


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


def _config_uses_vllm(config_path: str | Path) -> bool:
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return False
    framework = payload.get("framework")
    if not isinstance(framework, dict):
        return False
    serving = framework.get("serving")
    if not isinstance(serving, dict):
        return False
    return serving.get("backend") == "vllm"


def prepare_environment(config_path: str | Path) -> None:
    if os.environ.get("FLASHRL_VLLM_PYTHON"):
        return
    if not _config_uses_vllm(config_path):
        return
    runtime_python = find_default_vllm_python()
    if runtime_python is not None:
        os.environ["FLASHRL_VLLM_PYTHON"] = runtime_python


def load_example_config(config_path: str | Path) -> tuple[CodingHarnessConfig, dict[str, Any]]:
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {config_path}.")
    example = payload.get("example") or {}
    if not isinstance(example, dict):
        raise ValueError("config.example must be a mapping when present.")
    dataset = example.get("dataset") or {}
    if not isinstance(dataset, dict):
        raise ValueError("config.example.dataset must be a mapping when present.")
    harness = CodingHarnessConfig.from_mapping(example.get("harness"))
    return harness, dataset


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reference agent harness example.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the FlashRL config YAML.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional override for the training task count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_environment(args.config)
    harness_config, dataset_config = load_example_config(args.config)
    train_limit = args.train_limit
    if train_limit is None:
        raw_train_limit = dataset_config.get("train_limit")
        train_limit = int(raw_train_limit) if isinstance(raw_train_limit, int) else None
    dataset = build_coding_train_dataset(limit=train_limit)
    flashrl: FlashRL | None = None
    try:
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=build_coding_agent(harness_config),
            reward_fn=build_coding_reward_fn(),
        )
        flashrl.train(dataset)
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
