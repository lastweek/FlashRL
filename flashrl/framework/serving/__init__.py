"""Serving backends and factory helpers."""

from __future__ import annotations

from flashrl.framework.config import ServingConfig
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.serving.huggingface import HuggingFaceServingBackend
from flashrl.framework.serving.vllm_metal import VLLMMetalServingBackend


def create_serving_backend(
    config: ServingConfig,
    training_actor: ActorModel | None = None,
) -> ServingBackend:
    """Construct the requested serving backend implementation."""
    if config.backend == "huggingface":
        return HuggingFaceServingBackend(config)
    if config.backend == "vllm_metal":
        if training_actor is None:
            raise ValueError("vllm_metal serving requires the training actor during initialization.")
        return VLLMMetalServingBackend(config, training_actor=training_actor)
    raise ValueError(f"Unsupported serving backend: {config.backend}")


__all__ = [
    "ServingBackend",
    "HuggingFaceServingBackend",
    "VLLMMetalServingBackend",
    "create_serving_backend",
]
