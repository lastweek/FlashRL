"""Simple rolling-window context manager."""

from __future__ import annotations

from dataclasses import dataclass

from flashrl.framework.agent.context.base import BaseContextManager
from flashrl.framework.data_models import Message


@dataclass
class WindowedContextManager(BaseContextManager):
    """Keep system messages plus the most recent non-system messages."""

    max_messages: int = 8

    def __post_init__(self) -> None:
        if self.max_messages < 1:
            raise ValueError("WindowedContextManager.max_messages must be >= 1.")

    def build_messages(self, state) -> list[Message]:
        """Return the current windowed view over the conversation."""
        system_messages = [
            message.model_copy(deep=True)
            for message in state.conversation.messages
            if message.role == "system"
        ]
        recent_messages = [
            message.model_copy(deep=True)
            for message in state.conversation.messages
            if message.role != "system"
        ]
        return system_messages + recent_messages[-self.max_messages:]

    def observe(self, state, new_messages: list[Message]) -> None:
        """Windowing is derived from conversation state, so observe is a no-op."""
        del state, new_messages
        return None
