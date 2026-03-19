"""Base interface for agent context managers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from flashrl.framework.data_models import Message


class BaseContextManager(ABC):
    """Minimal context-management interface used by agent sessions."""

    @abstractmethod
    def build_messages(self, state) -> list[Message]:
        """Return the messages that should be visible to the next generation step."""

    @abstractmethod
    def observe(self, state, new_messages: list[Message]) -> None:
        """Observe newly appended messages after each traced step."""
