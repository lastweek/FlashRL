"""Serving backends and factory helpers."""

from __future__ import annotations

from flashrl.framework.config import ServingConfig
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.serving.huggingface import HuggingFaceServingBackend
from flashrl.framework.serving.vllm import VLLMServingBackend


def create_serving_backend(config: ServingConfig) -> ServingBackend:
    """Construct the requested serving backend implementation."""
    if config.backend == "huggingface":
        return HuggingFaceServingBackend(config)
    if config.backend == "vllm":
        return VLLMServingBackend(config)
    raise ValueError(f"Unsupported serving backend: {config.backend}")


__all__ = [
    "ServingBackend",
    "HuggingFaceServingBackend",
    "VLLMServingBackend",
    "create_serving_backend",
]
