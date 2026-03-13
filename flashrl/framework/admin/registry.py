"""Read-only registry for live admin objects."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable


AdminObjectProvider = Callable[[], list[dict[str, Any]]]


class AdminRegistry:
    """Registry that gathers live objects from provider callables on demand."""

    def __init__(self) -> None:
        self._providers: list[AdminObjectProvider] = []
        self._lock = Lock()

    def register(self, provider: AdminObjectProvider) -> None:
        """Register one provider that returns a list of admin objects."""
        with self._lock:
            self._providers.append(provider)

    def list_objects(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Return all current objects, optionally filtered by kind."""
        with self._lock:
            providers = list(self._providers)

        items: list[dict[str, Any]] = []
        for provider in providers:
            provided = provider() or []
            items.extend(provided)

        if kind is not None:
            items = [item for item in items if item.get("kind") == kind]

        return sorted(
            items,
            key=lambda item: (
                str(item.get("kind", "")),
                str(item.get("metadata", {}).get("name", "")),
            ),
        )

    def get_object(self, kind: str, name: str) -> dict[str, Any] | None:
        """Return one current object by kind and name."""
        for item in self.list_objects(kind):
            if item.get("metadata", {}).get("name") == name:
                return item
        return None
