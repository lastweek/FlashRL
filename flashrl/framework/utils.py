"""Generic utility functions shared across FlashRL."""

from typing import Any

import torch


def summary_stats(prefix: str, values: list[float]) -> dict[str, float]:
    """Compute mean/std/min/max stats for a list of floats."""
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        f"{prefix}_mean": float(tensor.mean().item()),
        f"{prefix}_std": float(tensor.std(unbiased=False).item()),
        f"{prefix}_min": float(tensor.min().item()),
        f"{prefix}_max": float(tensor.max().item()),
    }


def mean(values: list[float] | list[int]) -> float:
    """Return the arithmetic mean or zero for an empty list."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def truncate_preview(text: str, *, limit: int = 240) -> str:
    """Return a one-line preview clipped to the requested size."""
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."