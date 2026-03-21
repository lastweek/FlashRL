"""Explicit skill discovery and loading for serious harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from flashrl.framework.agent.session import LoadedSkill
from flashrl.framework.data_models import Prompt, ToolResult

_SKILL_MENTION_PATTERN = re.compile(r"@skill:([a-zA-Z0-9_.-]+)")


@dataclass
class SkillManager:
    """Discover and load explicit skill bundles from configured roots."""

    roots: Sequence[str | Path]

    def __post_init__(self) -> None:
        self._skills_by_name = self._discover_skills()

    def discover(self) -> dict[str, LoadedSkill]:
        return dict(self._skills_by_name)

    def parse_mentions(self, text: str) -> list[str]:
        return list(dict.fromkeys(match.group(1) for match in _SKILL_MENTION_PATTERN.finditer(text)))

    def render_catalog(self) -> str:
        if not self._skills_by_name:
            return "No explicit skills are available."
        lines = ["Available skills:"]
        for skill in sorted(self._skills_by_name.values(), key=lambda item: item.name):
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def preload(self, agent, names: Sequence[str]) -> list[LoadedSkill]:
        loaded: list[LoadedSkill] = []
        for name in names:
            skill = self.load(agent, name)
            if skill is not None:
                loaded.append(skill)
        return loaded

    def load(self, agent, name: str) -> LoadedSkill | None:
        skill = self._skills_by_name.get(str(name).strip())
        if skill is None:
            return None
        if skill.name in agent.session.active_skills:
            return agent.session.active_skills[skill.name]
        agent.session.active_skills[skill.name] = skill
        agent.add_message(
            "system",
            f"Loaded skill `{skill.name}`.\n{skill.content}",
            metadata={"skill_name": skill.name, "skill_source_path": skill.source_path},
        )
        agent.record_trace_event(
            "skill_load",
            payload={
                "skill_name": skill.name,
                "skill_source_path": skill.source_path,
            },
        )
        return skill

    def execute(
        self,
        tool_call,
        *,
        agent,
        tool,
    ) -> ToolResult:
        del tool
        name = str(tool_call.arguments.get("name", "")).strip()
        skill = self.load(agent, name)
        if skill is None:
            return ToolResult(
                content=f"Skill not found: {name}",
                error=True,
                metadata={"status": "skill_not_found"},
            )
        return ToolResult(
            content=f"Loaded skill `{skill.name}`.",
            error=False,
            metadata={
                "status": "skill_loaded",
                "skill_name": skill.name,
                "skill_source_path": skill.source_path,
            },
        )

    def _discover_skills(self) -> dict[str, LoadedSkill]:
        discovered: dict[str, LoadedSkill] = {}
        for root in self.roots:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                continue
            for skill_file in sorted(root_path.rglob("SKILL.md")):
                skill_name = skill_file.parent.name
                if skill_name in discovered:
                    continue
                text = skill_file.read_text(encoding="utf-8")
                description = self._extract_description(text)
                discovered[skill_name] = LoadedSkill(
                    name=skill_name,
                    description=description,
                    content=text.strip(),
                    source_path=str(skill_file),
                )
        return discovered

    def _extract_description(self, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped[:160]
        return "Explicit skill bundle."


def load_skill_placeholder(arguments: dict[str, object], prompt: Prompt) -> str:
    """Placeholder entrypoint for the model-visible load_skill tool."""
    del arguments, prompt
    raise RuntimeError("load_skill must be executed through SkillManager, not the subprocess tool runtime.")
