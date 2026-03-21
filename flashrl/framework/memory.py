"""Runtime memory snapshot and summary helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psutil
import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _device_type_for(device: Any | None) -> str:
    if device is None:
        return "cpu"
    resolved = getattr(device, "type", device)
    if resolved is None:
        return "cpu"
    return str(resolved)


def _device_index_for(device: Any | None) -> int | None:
    if device is None:
        return None
    return _safe_int(getattr(device, "index", None))


def _safe_device_counter(getter) -> int | None:
    try:
        return _safe_int(getter())
    except Exception:
        return None


def capture_memory_snapshot(device: Any | None = None) -> dict[str, Any]:
    """Capture one stable JSON-serializable memory snapshot."""
    process_payload: dict[str, Any] = {}
    system_payload: dict[str, Any] = {}
    device_payload: dict[str, Any] = {}

    try:
        process_payload["rss_bytes"] = int(psutil.Process().memory_info().rss)
    except Exception:
        process_payload = {}

    try:
        virtual_memory = psutil.virtual_memory()
        system_payload = {
            "total_bytes": int(virtual_memory.total),
            "available_bytes": int(virtual_memory.available),
        }
    except Exception:
        system_payload = {}

    device_type = _device_type_for(device)
    if device_type == "mps" and hasattr(torch, "mps"):
        device_payload = {
            "current_allocated_bytes": _safe_device_counter(
                getattr(torch.mps, "current_allocated_memory", lambda: None)
            ),
            "driver_allocated_bytes": _safe_device_counter(
                getattr(torch.mps, "driver_allocated_memory", lambda: None)
            ),
            "recommended_max_bytes": _safe_device_counter(
                getattr(torch.mps, "recommended_max_memory", lambda: None)
            ),
        }
    elif device_type == "cuda" and hasattr(torch, "cuda"):
        device_index = _device_index_for(device)
        if device_index is None:
            try:
                device_index = int(torch.cuda.current_device())
            except Exception:
                device_index = 0
        if torch.cuda.is_available():
            device_payload = {
                "index": int(device_index),
                "current_allocated_bytes": _safe_device_counter(
                    lambda: torch.cuda.memory_allocated(device_index)
                ),
                "reserved_bytes": _safe_device_counter(
                    lambda: torch.cuda.memory_reserved(device_index)
                ),
                "max_allocated_bytes": _safe_device_counter(
                    lambda: torch.cuda.max_memory_allocated(device_index)
                ),
            }

    return {
        "captured_at": _utc_now_iso(),
        "device_type": device_type,
        "process": process_payload,
        "system": system_payload,
        "device": {key: value for key, value in device_payload.items() if value is not None},
    }


def extract_memory_counters(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten one nested snapshot into counter-oriented fields."""
    if not isinstance(snapshot, dict):
        return {}
    process = snapshot.get("process") if isinstance(snapshot.get("process"), dict) else {}
    system = snapshot.get("system") if isinstance(snapshot.get("system"), dict) else {}
    device = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    return {
        "device_type": snapshot.get("device_type"),
        "process_rss_bytes": _safe_int(process.get("rss_bytes")),
        "system_total_bytes": _safe_int(system.get("total_bytes")),
        "system_available_bytes": _safe_int(system.get("available_bytes")),
        "device_current_allocated_bytes": _safe_int(device.get("current_allocated_bytes")),
        "device_driver_allocated_bytes": _safe_int(device.get("driver_allocated_bytes")),
        "device_reserved_bytes": _safe_int(device.get("reserved_bytes")),
        "device_max_allocated_bytes": _safe_int(device.get("max_allocated_bytes")),
        "device_recommended_max_bytes": _safe_int(device.get("recommended_max_bytes")),
    }


def update_memory_summary(
    summary: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Update one peak/min summary mapping from a snapshot."""
    counters = extract_memory_counters(snapshot)
    if not counters:
        return dict(summary or {})

    updated = dict(summary or {})
    device_type = counters.get("device_type")
    if device_type:
        updated["device_type"] = device_type

    peak_fields = (
        ("process_rss_bytes", "peak_process_rss_bytes"),
        ("device_current_allocated_bytes", "peak_device_current_allocated_bytes"),
        ("device_driver_allocated_bytes", "peak_device_driver_allocated_bytes"),
        ("device_reserved_bytes", "peak_device_reserved_bytes"),
        ("device_max_allocated_bytes", "peak_device_max_allocated_bytes"),
    )
    for source_key, target_key in peak_fields:
        value = counters.get(source_key)
        if value is None:
            continue
        updated[target_key] = max(int(updated.get(target_key, 0) or 0), int(value))

    low_fields = (
        ("system_available_bytes", "lowest_system_available_bytes"),
    )
    for source_key, target_key in low_fields:
        value = counters.get(source_key)
        if value is None:
            continue
        current = updated.get(target_key)
        updated[target_key] = int(value) if current is None else min(int(current), int(value))

    constant_fields = (
        ("system_total_bytes", "system_total_bytes"),
        ("device_recommended_max_bytes", "device_recommended_max_bytes"),
    )
    for source_key, target_key in constant_fields:
        value = counters.get(source_key)
        if value is not None:
            updated[target_key] = int(value)

    return updated


def summarize_memory_window(
    *snapshots: dict[str, Any] | None,
    start: dict[str, Any] | None = None,
    end: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one sequence of snapshots for step-level payloads."""
    summary: dict[str, Any] = {}
    for snapshot in snapshots:
        summary = update_memory_summary(summary, snapshot)
    if start is not None:
        summary["start"] = start
    if end is not None:
        summary["end"] = end
    return summary


def memory_pressure_tags(
    exc: BaseException,
    *,
    snapshot: dict[str, Any] | None,
    shared_device_pressure: bool = False,
) -> list[str]:
    """Return simple heuristic memory-pressure tags for one failure."""
    tags: list[str] = []
    message = str(exc).lower()
    if isinstance(exc, MemoryError) or "out of memory" in message:
        tags.append("oom_during_stage")

    counters = extract_memory_counters(snapshot)
    device_type = counters.get("device_type")
    device_current = counters.get("device_current_allocated_bytes")
    device_driver = counters.get("device_driver_allocated_bytes")
    device_reserved = counters.get("device_reserved_bytes")
    device_limit = counters.get("device_recommended_max_bytes")
    if device_type in {"mps", "cuda"} and device_limit:
        usage = max(
            float(device_current or 0),
            float(device_driver or 0),
            float(device_reserved or 0),
        )
        if usage >= float(device_limit) * 0.85:
            tags.append("device_pressure")
    elif device_type in {"mps", "cuda"} and (
        device_current is not None or device_driver is not None or device_reserved is not None
    ):
        tags.append("device_pressure")

    if shared_device_pressure and device_type == "mps":
        tags.append("shared_device_pressure")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def _format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    scaled = float(value)
    unit_index = 0
    while scaled >= 1024.0 and unit_index < len(units) - 1:
        scaled /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(scaled)}{units[unit_index]}"
    return f"{scaled:.2f}{units[unit_index]}"


def format_memory_brief(snapshot: dict[str, Any] | None) -> str:
    """Render one bounded human-readable memory suffix."""
    counters = extract_memory_counters(snapshot)
    if not counters:
        return ""

    parts: list[str] = []
    rss = _format_bytes(counters.get("process_rss_bytes"))
    if rss is not None:
        parts.append(f"rss={rss}")

    available = _format_bytes(counters.get("system_available_bytes"))
    if available is not None:
        parts.append(f"avail={available}")

    device_type = counters.get("device_type")
    if device_type == "mps":
        current = _format_bytes(counters.get("device_current_allocated_bytes"))
        limit = _format_bytes(counters.get("device_recommended_max_bytes"))
        driver = _format_bytes(counters.get("device_driver_allocated_bytes"))
        if current is not None or limit is not None:
            ratio = "/".join(part for part in (current, limit) if part is not None)
            parts.append(f"mps={ratio}")
        elif driver is not None:
            parts.append(f"mps_driver={driver}")
    elif device_type == "cuda":
        current = _format_bytes(counters.get("device_current_allocated_bytes"))
        reserved = _format_bytes(counters.get("device_reserved_bytes"))
        if current is not None or reserved is not None:
            ratio = "/".join(part for part in (current, reserved) if part is not None)
            parts.append(f"cuda={ratio}")

    if not parts:
        return ""
    return "mem " + " ".join(parts)
