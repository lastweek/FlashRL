"""FSDP2 training backend."""

from __future__ import annotations

from typing import Any

import torch

from flashrl.framework.admin import build_admin_object, utc_now_iso
from flashrl.framework.config import GrpoConfig, TrainingConfig
from flashrl.framework.training.huggingface import HuggingFaceTrainingBackend


class FSDP2TrainingBackend(HuggingFaceTrainingBackend):
    """Guarded FSDP2 training backend for single-node local runs."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        learning_rate: float,
        grpo_config: GrpoConfig,
        reference_enabled: bool = False,
        reference_device: str | None = None,
    ) -> None:
        if config.dp_size > 1:
            raise ValueError(
                "training.backend='fsdp2' currently supports only dp_size=1 in-process. "
                "The controller/training split and backend abstraction are ready, but "
                "multi-rank worker orchestration has not been enabled yet."
            )
        super().__init__(
            config,
            learning_rate=learning_rate,
            grpo_config=grpo_config,
            reference_enabled=reference_enabled,
            reference_device=reference_device,
        )
        self.world_size = config.dp_size
        self.rank = 0
        self.is_primary = True
        self._created_at = utc_now_iso()
        self._attempt_single_rank_fsdp_wrap()

    def _attempt_single_rank_fsdp_wrap(self) -> None:
        """Apply a best-effort single-rank FSDP2 wrap when CUDA is available."""
        if getattr(self.device, "type", None) != "cuda":
            return
        try:
            from torch.distributed._composable.fsdp import fully_shard
        except Exception:
            return
        try:
            fully_shard(self.actor.model)
        except Exception:
            return

    def list_admin_objects(self) -> list[dict[str, Any]]:
        """Expose one training replica object for distributed-style admin views."""
        return [
            build_admin_object(
                "TrainingReplica",
                f"training-replica-{self.rank}",
                uid=f"training-replica:{self.rank}",
                created_at=self._created_at,
                labels={"flashrl.dev/training-backend": self.backend_name},
                spec={
                    "backend": self.backend_name,
                    "rank": self.rank,
                    "worldSize": self.world_size,
                    "device": str(self.device),
                    "modelName": self.config.model_name,
                },
                status={
                    "phase": "Ready",
                    "isPrimary": self.is_primary,
                    "loaded": True,
                },
            )
        ]
