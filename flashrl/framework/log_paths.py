"""Helpers for allocating and naming run-scoped log directories."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re


RUN_INDEX_WIDTH = 6


def sanitize_model_name(model_name: str) -> str:
    """Convert a model name into a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()
    return slug or "model"


def allocate_run_index(log_dir: Path) -> int:
    """Allocate the next global run index under the run root."""
    log_dir.mkdir(parents=True, exist_ok=True)
    counter_path = log_dir / ".run_counter"
    counter_path.touch(exist_ok=True)

    with counter_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        text = handle.read().strip()
        current = int(text) if text else 0
        run_index = current + 1
        handle.seek(0)
        handle.write(f"{run_index}\n")
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return run_index


def format_run_index(run_index: int) -> str:
    """Format a run index with stable zero padding."""
    return f"{run_index:0{RUN_INDEX_WIDTH}d}"
