"""Session-level state shared across one live agent trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flashrl.framework.data_models import Conversation


@dataclass(frozen=True)
class LoadedSkill:
    """One loaded skill bundle available to the current session."""

    name: str
    description: str
    content: str
    source_path: str | None = None


@dataclass
class SessionContext:
    """Structured per-trajectory session context."""

    conversation: Conversation
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_index: int | None = None
    candidate_index: int | None = None
    tool_profile: str = "default"
    pinned_skill_names: list[str] = field(default_factory=list)
    active_skills: dict[str, LoadedSkill] = field(default_factory=dict)
    rolling_summary: str = ""

