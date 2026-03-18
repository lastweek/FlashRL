"""Serving backends and factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from flashrl.framework.config import ServingConfig
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.serving.huggingface import HuggingFaceServingBackend
from flashrl.framework.serving.vllm import VLLMServingBackend


def create_serving_backend(
    config: ServingConfig,
    startup_logger: Callable[[str], None] | None = None,
    log_dir: str | Path | None = None,
) -> ServingBackend:
    """Construct the requested serving backend implementation."""
    if config.backend == "huggingface":
        return HuggingFaceServingBackend(config, startup_logger=startup_logger, log_dir=log_dir)
    if config.backend == "vllm":
        return VLLMServingBackend(config, startup_logger=startup_logger, log_dir=log_dir)
    raise ValueError(f"Unsupported serving backend: {config.backend}")


__all__ = [
    "ServingBackend",
    "HuggingFaceServingBackend",
    "VLLMServingBackend",
    "create_serving_backend",
]
