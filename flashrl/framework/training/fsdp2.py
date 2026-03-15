"""FSDP2 training backends."""

from __future__ import annotations

from typing import Any

from flashrl.framework.admin import build_admin_object, utc_now_iso
from flashrl.framework.config import TrainingConfig
from flashrl.framework.training.huggingface import (
    HuggingFaceReferenceBackend,
    HuggingFaceTrainingBackend,
)


class FSDP2TrainingBackend(HuggingFaceTrainingBackend):
    """Guarded FSDP2 actor backend for single-node local runs."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        learning_rate: float,
    ) -> None:
        if config.dp_size > 1:
            raise ValueError(
                "dp_size must be 1 when backend='fsdp2' in the current in-process runtime."
            )
        super().__init__(config, learning_rate=learning_rate)
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
            fully_shard(self.model_copy.model)
        except Exception:
            return

    def list_admin_objects(self) -> list[dict[str, Any]]:
        """Expose one training replica object for distributed-style admin views."""
        return [
            build_admin_object(
                "TrainingReplica",
                f"{self.role}-replica-{self.rank}",
                uid=f"{self.role}-replica:{self.rank}",
                created_at=self._created_at,
                labels={"flashrl.dev/training-backend": self.backend_name},
                spec={
                    "role": self.role,
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


class FSDP2ReferenceBackend(HuggingFaceReferenceBackend):
    """Guarded FSDP2-style frozen reference backend for single-node local runs."""

    def __init__(self, config: TrainingConfig) -> None:
        if config.dp_size > 1:
            raise ValueError(
                "dp_size must be 1 when backend='fsdp2' in the current in-process runtime."
            )
        super().__init__(config)
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
            fully_shard(self.model_copy.model)
        except Exception:
            return

    def list_admin_objects(self) -> list[dict[str, Any]]:
        """Expose one reference replica object for distributed-style admin views."""
        return [
            build_admin_object(
                "TrainingReplica",
                f"{self.role}-replica-{self.rank}",
                uid=f"{self.role}-replica:{self.rank}",
                created_at=self._created_at,
                labels={"flashrl.dev/training-backend": self.backend_name},
                spec={
                    "role": self.role,
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
