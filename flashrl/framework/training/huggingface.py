"""In-process Hugging Face training backends."""

from __future__ import annotations

import time

import torch

from flashrl.framework.config import TrainingConfig
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import set_num_threads
from flashrl.framework.training.base import ActorTrainingBackend, TrainingBackend


class HuggingFaceTrainingBackend(ActorTrainingBackend):
    """Single-process trainable actor backend backed by a local Hugging Face model."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        learning_rate: float,
    ) -> None:
        super().__init__(config, learning_rate=learning_rate)

        set_num_threads(config.num_threads)

        started_at = time.perf_counter()
        self.model_copy = ActorModel(config)
        self.model_copy.train()
        self.device = self.model_copy.device
        self.optimizer = torch.optim.Adam(
            self.model_copy.model.parameters(),
            lr=learning_rate,
        )
        self.startup_events.append(
            {
                "component": self.component_name,
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": config.num_threads,
                    "duration_seconds": time.perf_counter() - started_at,
                },
            }
        )


class HuggingFaceReferenceBackend(TrainingBackend):
    """Single-process frozen reference backend backed by a local Hugging Face model."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config, role="reference")

        set_num_threads(config.num_threads)

        started_at = time.perf_counter()
        self.model_copy = ActorModel(config)
        self.model_copy.eval()
        self.model_copy.model.requires_grad_(False)
        self.device = self.model_copy.device
        self.startup_events.append(
            {
                "component": self.component_name,
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": config.num_threads,
                    "duration_seconds": time.perf_counter() - started_at,
                },
            }
        )
