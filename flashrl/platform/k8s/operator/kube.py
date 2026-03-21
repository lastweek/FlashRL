"""Minimal Kubernetes client bootstrap and normalization helpers."""

from __future__ import annotations

from typing import Any, Callable


def load_client(client_module: Any | None) -> Any:
    """Load the Kubernetes client module lazily."""
    if client_module is not None:
        return client_module
    try:
        from kubernetes import client, config
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "FlashRL platform operations require the optional `kubernetes` package."
        ) from exc
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    return client


def load_watch_factory(watch_factory: Callable[[], Any] | None) -> Callable[[], Any]:
    """Load the Kubernetes watch factory lazily."""
    if watch_factory is not None:
        return watch_factory
    try:
        from kubernetes import watch
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "FlashRL operator watch mode requires the optional `kubernetes` package."
        ) from exc
    return watch.Watch


def is_not_found(exc: Exception) -> bool:
    """Return whether one Kubernetes client exception represents a 404."""
    return int(getattr(exc, "status", 0) or 0) == 404


def to_plain(value: Any) -> Any:
    """Convert Kubernetes client objects into plain dict/list data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_plain(to_dict())
    items = getattr(value, "items", None)
    if callable(items):
        return {key: to_plain(item) for key, item in items()}
    return value


def get_path(payload: Any, *path: str, default: Any = None) -> Any:
    """Read one nested value from normalized payload data."""
    current = to_plain(payload)
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
