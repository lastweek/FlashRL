"""Deterministic rolling-summary compaction for long agent traces."""

from __future__ import annotations

from dataclasses import dataclass

from flashrl.framework.agent.context.base import BaseContextManager
from flashrl.framework.data_models import AgentTraceEvent, Message


@dataclass(frozen=True)
class CompactionPolicy:
    """Thresholds controlling deterministic session compaction."""

    trigger_message_count: int = 12
    preserve_recent_messages: int = 6
    max_summary_chars: int = 1200

    def __post_init__(self) -> None:
        if self.trigger_message_count < 2:
            raise ValueError("CompactionPolicy.trigger_message_count must be >= 2.")
        if self.preserve_recent_messages < 1:
            raise ValueError("CompactionPolicy.preserve_recent_messages must be >= 1.")
        if self.preserve_recent_messages >= self.trigger_message_count:
            raise ValueError(
                "CompactionPolicy.preserve_recent_messages must be smaller than trigger_message_count."
            )
        if self.max_summary_chars < 64:
            raise ValueError("CompactionPolicy.max_summary_chars must be >= 64.")


@dataclass
class CompactionManager(BaseContextManager):
    """Keep recent raw turns plus a deterministic rolling summary."""

    policy: CompactionPolicy = CompactionPolicy()

    def build_messages(self, state) -> list[Message]:
        system_messages = [
            message.model_copy(deep=True)
            for message in state.conversation.messages
            if message.role == "system"
        ]
        if state.session_context.rolling_summary:
            system_messages.append(
                Message(
                    role="system",
                    content=(
                        "Session summary:\n"
                        f"{state.session_context.rolling_summary}"
                    ),
                    metadata={"summary": True},
                )
            )
        recent_messages = [
            message.model_copy(deep=True)
            for message in state.conversation.messages
            if message.role != "system"
        ]
        return system_messages + recent_messages[-self.policy.preserve_recent_messages :]

    def observe(self, state, new_messages: list[Message]) -> None:
        del new_messages
        non_system_messages = [
            message
            for message in state.conversation.messages
            if message.role != "system"
        ]
        if len(non_system_messages) <= self.policy.trigger_message_count:
            return

        preserved = non_system_messages[-self.policy.preserve_recent_messages :]
        compacted = non_system_messages[: -self.policy.preserve_recent_messages]
        summary_lines: list[str] = []
        for message in compacted:
            label = message.role.title()
            content = message.content.strip().replace("\n", " ")
            if len(content) > 160:
                content = content[:157].rstrip() + "..."
            summary_lines.append(f"{label}: {content}")
        previous_summary = state.session_context.rolling_summary.strip()
        if previous_summary:
            summary_lines.insert(0, previous_summary)
        summary_text = "\n".join(line for line in summary_lines if line).strip()
        summary_text = summary_text[: self.policy.max_summary_chars].rstrip()
        state.session_context.rolling_summary = summary_text
        state.agent_trace.events.append(
            AgentTraceEvent(
                event_type="compaction",
                step_index=len(state.assistant_turns),
                prompt_index=state.session_context.prompt_index,
                candidate_index=state.session_context.candidate_index,
                payload={
                    "trigger_message_count": self.policy.trigger_message_count,
                    "compacted_message_count": len(compacted),
                    "preserved_message_count": len(preserved),
                    "summary_char_count": len(summary_text),
                },
            )
        )

