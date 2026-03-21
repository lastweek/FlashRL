"""Example-local config objects for the reference agent harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flashrl.framework.agent import ToolProfile


@dataclass(frozen=True)
class AgentHarnessConfig:
    """Serializable example-local harness config used by train/eval and ablations."""

    tool_profile: str = ToolProfile.DEFAULT.value
    enable_skills: bool = True
    enable_compaction: bool = True
    enable_subagents: bool = True
    max_steps: int = 6
    max_parallel_calls: int = 4
    compaction_trigger_message_count: int = 10
    compaction_preserve_recent_messages: int = 5
    subagent_max_parallel: int = 2
    subagent_max_per_parent_turn: int = 2
    subagent_timeout_seconds: float = 10.0
    pinned_skills: tuple[str, ...] = ()
    extra_skill_roots: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "AgentHarnessConfig":
        if payload is None:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown agent harness config keys: {', '.join(unknown)}")
        normalized = dict(payload)
        if "pinned_skills" in normalized:
            normalized["pinned_skills"] = tuple(str(item) for item in normalized["pinned_skills"])
        if "extra_skill_roots" in normalized:
            normalized["extra_skill_roots"] = tuple(
                str(Path(item))
                for item in normalized["extra_skill_roots"]
            )
        return cls(**normalized)

    def with_overrides(self, payload: dict[str, Any] | None) -> "AgentHarnessConfig":
        if not payload:
            return self
        merged = {**self.__dict__, **payload}
        return type(self).from_mapping(merged)
