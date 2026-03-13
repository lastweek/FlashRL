"""Helpers for Kubernetes-like admin objects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ADMIN_API_VERSION = "flashrl.dev/v1alpha1"


def utc_now_iso() -> str:
    """Return the current UTC time in RFC3339 form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_admin_object(
    kind: str,
    name: str,
    *,
    uid: str,
    created_at: str,
    spec: dict[str, Any],
    status: dict[str, Any],
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one normalized admin object envelope."""
    return {
        "apiVersion": ADMIN_API_VERSION,
        "kind": kind,
        "metadata": {
            "name": name,
            "uid": uid,
            "creationTimestamp": created_at,
            "labels": dict(labels or {}),
        },
        "spec": dict(spec),
        "status": dict(status),
    }


def build_admin_object_list(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the normalized list envelope for admin objects."""
    return {
        "apiVersion": ADMIN_API_VERSION,
        "kind": "ObjectList",
        "items": list(items),
    }
