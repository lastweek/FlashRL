"""Structured tool registration and visibility resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence

from flashrl.framework.agent.tools.runtime import Tool

if False:  # pragma: no cover
    from flashrl.framework.agent.runtime import AgentState


class ToolProfile(str, Enum):
    """Built-in visibility profiles for serious harnesses."""

    DEFAULT = "default"
    READONLY = "readonly"
    SUBAGENT = "subagent"


ToolPredicate = Callable[["AgentState"], bool]


@dataclass(frozen=True)
class ToolRegistration:
    """One tool plus the profiles where it should be visible."""

    tool: Tool
    profiles: tuple[str, ...] = (ToolProfile.DEFAULT.value,)
    predicate: ToolPredicate | None = None


@dataclass
class ToolRegistry:
    """Resolve visible tools from a structured registry instead of ad hoc lists."""

    registrations: list[ToolRegistration] = field(default_factory=list)

    def __init__(
        self,
        registrations: Sequence[ToolRegistration] | None = None,
    ) -> None:
        self.registrations = list(registrations or [])
        self._validate()

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[Tool],
        *,
        profiles: Sequence[str] | None = None,
    ) -> "ToolRegistry":
        resolved_profiles = tuple(profiles or (ToolProfile.DEFAULT.value,))
        return cls(
            [
                ToolRegistration(tool=tool, profiles=resolved_profiles)
                for tool in tools
            ]
        )

    def register(
        self,
        tool: Tool,
        *,
        profiles: Sequence[str] | None = None,
        predicate: ToolPredicate | None = None,
    ) -> None:
        self.registrations.append(
            ToolRegistration(
                tool=tool,
                profiles=tuple(profiles or (ToolProfile.DEFAULT.value,)),
                predicate=predicate,
            )
        )
        self._validate()

    def resolve(
        self,
        *,
        state: "AgentState",
        profile: str | ToolProfile = ToolProfile.DEFAULT.value,
    ) -> list[Tool]:
        resolved_profile = str(getattr(profile, "value", profile))
        visible: list[Tool] = []
        seen_names: set[str] = set()
        for registration in self.registrations:
            if resolved_profile not in registration.profiles:
                continue
            if registration.predicate is not None and not registration.predicate(state):
                continue
            if registration.tool.name in seen_names:
                raise ValueError(f"Duplicate tool name: {registration.tool.name}")
            seen_names.add(registration.tool.name)
            visible.append(registration.tool)
        return visible

    def names_for_profile(self, profile: str | ToolProfile) -> list[str]:
        resolved_profile = str(getattr(profile, "value", profile))
        return [
            registration.tool.name
            for registration in self.registrations
            if resolved_profile in registration.profiles
        ]

    def _validate(self) -> None:
        seen_names: set[str] = set()
        for registration in self.registrations:
            tool_name = registration.tool.name
            if tool_name in seen_names:
                raise ValueError(f"Duplicate tool name in ToolRegistry: {tool_name}")
            seen_names.add(tool_name)
            if not registration.profiles:
                raise ValueError(f"Tool '{tool_name}' must be registered in at least one profile.")

