"""In-process Hugging Face training backend."""

from __future__ import annotations

import time

import torch

from flashrl.framework.config import GrpoConfig, TrainingConfig
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import set_num_threads
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.training.base import TrainingBackend


class HuggingFaceTrainingBackend(TrainingBackend):
    """Single-process training backend backed by a local Hugging Face model."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        learning_rate: float,
        grpo_config: GrpoConfig,
        reference_enabled: bool = False,
        reference_device: str | None = None,
    ) -> None:
        super().__init__(
            config,
            learning_rate=learning_rate,
            grpo_config=grpo_config,
            reference_enabled=reference_enabled,
            reference_device=reference_device,
        )

        set_num_threads(config.num_threads)

        started_at = time.perf_counter()
        self.actor = ActorModel(config)
        self.actor.train()
        self.device = self.actor.device
        self.optimizer = torch.optim.Adam(
            self.actor.model.parameters(),
            lr=learning_rate,
        )
        self.startup_events.append(
            {
                "component": "training_backend",
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": config.num_threads,
                    "duration_seconds": time.perf_counter() - started_at,
                },
            }
        )

        if not reference_enabled:
            return

        started_at = time.perf_counter()
        reference_config = config.model_copy(
            update={"device": reference_device or config.device}
        )
        self.reference = ReferenceModel(reference_config)
        self.startup_events.append(
            {
                "component": "reference_model",
                "status": "completed",
                "metadata": {
                    "device": str(self.reference.device),
                    "cpu_threads": reference_config.num_threads,
                    "duration_seconds": time.perf_counter() - started_at,
                },
            }
        )
