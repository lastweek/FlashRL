"""Managed checkpoint helpers for FlashRL runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from flashrl.framework.config import CheckpointingConfig


LATEST_CHECKPOINT = "latest"
CHECKPOINT_SCHEMA_VERSION = 4
LOGGER_STATE_SCHEMA_VERSION = 1


def _float_mapping(value: Any) -> dict[str, float]:
    """Normalize a JSON-like mapping into float values."""
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, item in value.items():
        try:
            normalized[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return normalized


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one JSON file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Managed resume details extracted from one checkpoint payload."""

    checkpoint_path: Path
    run_id: str
    run_index: int
    run_dir: Path
    lifecycle_totals: dict[str, float]
    run_logger_state: dict[str, Any]


class CheckpointManager:
    """Resolve managed checkpoint targets and manifests for one FlashRL instance."""

    def __init__(self, config: CheckpointingConfig | None = None) -> None:
        self.config = config or CheckpointingConfig()
        self._resume_consumed = False

    def managed_checkpointing_enabled(self) -> bool:
        """Return whether any managed checkpoint behavior is configured."""
        return any(
            (
                self.config.save_every_steps is not None,
                self.config.save_on_run_end,
                self.config.resume_from is not None,
            )
        )

    def should_save_interval(self, step: int) -> bool:
        """Return whether a completed step should emit an interval checkpoint."""
        every = self.config.save_every_steps
        return every is not None and step > 0 and (step % every) == 0

    def interval_checkpoint_path(self, *, run_dir: Path, step: int) -> Path:
        """Return the path for one managed interval checkpoint."""
        directory = self.checkpoint_directory(run_dir=run_dir)
        return directory / f"step-{step:08d}.pt"

    def final_checkpoint_path(self, *, run_dir: Path) -> Path:
        """Return the path for the managed final checkpoint."""
        if self.config.final_path is not None:
            return Path(self.config.final_path)
        directory = self.checkpoint_directory(run_dir=run_dir)
        return directory / "final.pt"

    def latest_manifest_path(self, *, run_dir: Path) -> Path:
        """Return the path for the managed latest-checkpoint manifest."""
        return self.checkpoint_directory(run_dir=run_dir) / "latest.json"

    def checkpoint_directory(self, *, run_dir: Path) -> Path:
        """Resolve the active checkpoint directory for one run."""
        if self.config.directory is not None:
            return Path(self.config.directory)
        return run_dir / "checkpoints"

    def resolve_resume_path(self) -> Path | None:
        """Resolve the configured one-shot resume target if present."""
        if self._resume_consumed or self.config.resume_from is None:
            return None

        resume_from = self.config.resume_from
        if resume_from == LATEST_CHECKPOINT:
            directory = Path(self.config.directory)
            manifest_path = directory / "latest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Managed latest checkpoint manifest does not exist: {manifest_path}"
                )
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            checkpoint_path = manifest.get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError(
                    f"Managed latest checkpoint manifest is missing checkpoint_path: {manifest_path}"
                )
            return Path(str(checkpoint_path))

        return Path(resume_from)

    def mark_resume_consumed(self) -> None:
        """Prevent the configured resume target from being used again."""
        self._resume_consumed = True

    def build_restored_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_metadata: dict[str, Any] | None,
    ) -> RestoredCheckpoint:
        """Validate the metadata required for managed append-resume."""
        if not isinstance(checkpoint_metadata, dict):
            raise ValueError(
                "Managed checkpoint resume requires role-based checkpoint metadata."
            )

        run_info = checkpoint_metadata.get("run")
        if not isinstance(run_info, dict):
            raise ValueError(
                "Managed checkpoint resume requires prior run metadata in checkpoint_metadata.run."
            )

        run_id = run_info.get("run_id")
        run_index = run_info.get("run_index")
        run_dir = run_info.get("run_dir")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Managed checkpoint resume requires checkpoint_metadata.run.run_id.")
        if not isinstance(run_index, int):
            raise ValueError("Managed checkpoint resume requires checkpoint_metadata.run.run_index.")
        if not isinstance(run_dir, str) or not run_dir:
            raise ValueError("Managed checkpoint resume requires checkpoint_metadata.run.run_dir.")

        resolved_run_dir = Path(run_dir)
        if not resolved_run_dir.exists():
            raise FileNotFoundError(
                f"Managed checkpoint resume requires the original run directory, but it does not exist: {resolved_run_dir}"
            )

        run_logger_state = checkpoint_metadata.get("run_logger_state")
        if not isinstance(run_logger_state, dict):
            raise ValueError(
                "Managed checkpoint resume requires checkpoint_metadata.run_logger_state."
            )

        return RestoredCheckpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            run_index=run_index,
            run_dir=resolved_run_dir,
            lifecycle_totals=_float_mapping(checkpoint_metadata.get("lifecycle_totals")),
            run_logger_state=dict(run_logger_state),
        )

    def write_latest_manifest(
        self,
        *,
        run_dir: Path,
        checkpoint_path: Path,
        epoch: int,
        step: int,
        trigger: str,
        run_id: str | None,
        run_index: int | None,
    ) -> None:
        """Atomically update the managed latest-checkpoint manifest."""
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_path": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "trigger": trigger,
            "run_id": run_id,
            "run_index": run_index,
            "run_dir": str(run_dir),
        }
        _atomic_write_json(self.latest_manifest_path(run_dir=run_dir), payload)
