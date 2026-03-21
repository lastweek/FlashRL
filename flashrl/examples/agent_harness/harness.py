"""Reference agent harness assembled from generic FlashRL agent primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
from uuid import uuid4

from flashrl.examples.agent_harness.config import AgentHarnessConfig
from flashrl.framework.agent import (
    Agent,
    AgentToolExecutor,
    CompactionManager,
    CompactionPolicy,
    SkillManager,
    SubagentManager,
    Tool,
    ToolProfile,
    ToolRegistry,
    WindowedContextManager,
)
from flashrl.framework.agent.tools import SubprocessToolRuntime
from flashrl.framework.data_models import ToolCall

EXAMPLE_DIR = Path(__file__).resolve().parent
SKILLS_DIR = EXAMPLE_DIR / "skills"
TOOL_HELPER_MODULE = "flashrl.examples.agent_harness.tool_helpers"
READ_ONLY_PROFILES = (
    ToolProfile.DEFAULT.value,
    ToolProfile.READONLY.value,
    ToolProfile.SUBAGENT.value,
)
REPO_TOOL_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("list_repo_files", "List repository files.", "list_repo_files", READ_ONLY_PROFILES),
    ("read_repo_file", "Read one repository file.", "read_repo_file", READ_ONLY_PROFILES),
    ("search_repo_text", "Search for text in repository files.", "search_repo_text", READ_ONLY_PROFILES),
    (
        "run_repo_shell",
        "Run a small allowlisted shell command in the repo.",
        "run_repo_shell",
        (ToolProfile.DEFAULT.value,),
    ),
)


@dataclass(frozen=True)
class _Decision:
    kind: Literal["action", "final"]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    parse_error: str | None = None


def build_agent_harness(
    config: AgentHarnessConfig,
    *,
    runtime: SubprocessToolRuntime | None = None,
) -> Agent:
    """Build the reference agent harness from the generic framework primitives."""
    effective_runtime = runtime or SubprocessToolRuntime(
        default_timeout_seconds=8.0,
        default_memory_limit_mb=256,
    )
    skill_manager = _build_skill_manager(config)
    child_agent_builder = _build_child_agent_builder(config, effective_runtime)
    tool_executor = _build_tool_executor(
        config=config,
        runtime=effective_runtime,
        skill_manager=skill_manager,
        child_agent_builder=child_agent_builder,
    )
    return Agent(
        run_fn=_build_run_fn(
            config=config,
            skill_manager=skill_manager,
            tool_executor=tool_executor,
        ),
        tools=_build_tool_registry(config),
        context_manager=_build_context_manager(config),
        max_steps=config.max_steps,
        runtime=effective_runtime,
        max_parallel_calls=config.max_parallel_calls,
        tool_executor=tool_executor,
    )


def run_agent_step_loop(
    agent: Agent,
    *,
    tool_executor: AgentToolExecutor,
) -> None:
    while not agent.done:
        available_tools = agent.available_tools()
        sample = agent.generate(agent.build_prompt(tools=available_tools))
        decision = _parse_response(sample.text)
        if decision.kind == "action":
            if decision.parse_error:
                _record_parse_error(agent, sample_text=sample.text, error_text=decision.parse_error)
                continue
            agent.record_generation(sample, tool_calls=decision.tool_calls)
            agent.run_tools(decision.tool_calls, tools=available_tools, executor=tool_executor)
            continue
        agent.record_generation(sample)
        agent.finish(decision.final_text)


def _build_run_fn(
    *,
    config: AgentHarnessConfig,
    skill_manager: SkillManager | None,
    tool_executor: AgentToolExecutor,
) -> Callable[[Agent], None]:
    def run(agent: Agent) -> None:
        agent.session.tool_profile = config.tool_profile
        agent.add_message(
            "system",
            _build_system_prompt(
                skill_catalog=skill_manager.render_catalog() if skill_manager is not None else None
            ),
        )
        _preload_skills(agent, config=config, skill_manager=skill_manager)
        run_agent_step_loop(agent, tool_executor=tool_executor)

    return run


def _build_child_agent_builder(
    config: AgentHarnessConfig,
    runtime: SubprocessToolRuntime,
) -> Callable[[Agent, ToolCall], Agent]:
    def build_child_agent(parent_agent: Agent, tool_call: ToolCall) -> Agent:
        del tool_call
        child_config = config.with_overrides(
            {
                "enable_subagents": False,
                "tool_profile": ToolProfile.SUBAGENT.value,
            }
        )
        child_agent = build_agent_harness(child_config, runtime=runtime)
        parent_active_skills = tuple(parent_agent.session.active_skills)
        if parent_active_skills:
            child_agent.run_fn = _wrap_child_run_fn(
                child_agent.run_fn,
                pinned_skill_names=parent_active_skills,
            )  # type: ignore[method-assign]
        return child_agent

    return build_child_agent


def _build_tool_executor(
    *,
    config: AgentHarnessConfig,
    runtime: SubprocessToolRuntime,
    skill_manager: SkillManager | None,
    child_agent_builder: Callable[[Agent, ToolCall], Agent],
) -> AgentToolExecutor:
    handlers: dict[str, Any] = {}
    if skill_manager is not None:
        handlers["load_skill"] = skill_manager
    if config.enable_subagents:
        handlers["run_subagent"] = SubagentManager(
            build_agent=child_agent_builder,
            max_parallel=config.subagent_max_parallel,
            max_per_parent_turn=config.subagent_max_per_parent_turn,
            timeout_seconds=config.subagent_timeout_seconds,
        )
    return AgentToolExecutor(
        runtime=runtime,
        max_parallel_calls=config.max_parallel_calls,
        handlers=handlers,
    )


def _build_context_manager(config: AgentHarnessConfig):
    if config.enable_compaction:
        return CompactionManager(
            policy=CompactionPolicy(
                trigger_message_count=config.compaction_trigger_message_count,
                preserve_recent_messages=config.compaction_preserve_recent_messages,
                max_summary_chars=1200,
            )
        )
    return WindowedContextManager(max_messages=10)


def _build_skill_manager(config: AgentHarnessConfig) -> SkillManager | None:
    if not config.enable_skills:
        return None
    return SkillManager([SKILLS_DIR, *config.extra_skill_roots])


def _preload_skills(
    agent: Agent,
    *,
    config: AgentHarnessConfig,
    skill_manager: SkillManager | None,
) -> None:
    if skill_manager is None:
        return
    preload_skills = _resolve_preload_skill_names(
        agent,
        config=config,
        skill_manager=skill_manager,
    )
    agent.session.pinned_skill_names = preload_skills
    skill_manager.preload(agent, preload_skills)


def _resolve_preload_skill_names(
    agent: Agent,
    *,
    config: AgentHarnessConfig,
    skill_manager: SkillManager,
) -> list[str]:
    preload_skills = list(config.pinned_skills)
    preload_skills.extend(
        str(item)
        for item in agent.prompt.metadata.get("preload_skills", [])
        if str(item).strip()
    )
    preload_skills.extend(skill_manager.parse_mentions(agent.prompt.text))
    return list(dict.fromkeys(preload_skills))


def _record_parse_error(
    agent: Agent,
    *,
    sample_text: str,
    error_text: str,
) -> None:
    agent.add_message("assistant", sample_text)
    agent.add_message(
        "tool",
        error_text,
        metadata={
            "tool_name": "invalid_action",
            "tool_id": uuid4().hex,
            "error": True,
            "status": "parse_error",
        },
    )


def _build_tool_registry(config: AgentHarnessConfig) -> ToolRegistry:
    registry = ToolRegistry()
    for name, description, entrypoint_name, profiles in REPO_TOOL_SPECS:
        registry.register(
            Tool(
                name=name,
                description=description,
                entrypoint=f"{TOOL_HELPER_MODULE}:{entrypoint_name}",
            ),
            profiles=profiles,
        )
    if config.enable_skills:
        registry.register(
            Tool(
                name="load_skill",
                description="Load one explicit skill bundle by name.",
                entrypoint="flashrl.framework.agent.skills:load_skill_placeholder",
            ),
            profiles=(ToolProfile.DEFAULT.value, ToolProfile.SUBAGENT.value),
        )
    if config.enable_subagents:
        registry.register(
            Tool(
                name="run_subagent",
                description="Delegate one bounded sub-task to a local child agent.",
                entrypoint="flashrl.framework.agent.subagents:run_subagent_placeholder",
            ),
            profiles=(ToolProfile.DEFAULT.value,),
        )
    return registry


def _build_system_prompt(skill_catalog: str | None) -> str:
    lines = [
        "You are a coding agent working against a small local repository.",
        "Use tools before answering repository questions.",
        "When you need tools, respond with `Action:` followed by either one JSON object or a JSON array of objects.",
        "Each object must have the keys `tool` and `arguments`.",
        "When you are ready to answer, respond with `Final:` followed by the exact final answer and nothing else.",
        "For multi-part tasks, you may use `run_subagent` to delegate bounded sub-questions.",
    ]
    if skill_catalog:
        lines.append("")
        lines.append(skill_catalog)
    return "\n".join(lines)


def _parse_response(raw_text: str) -> _Decision:
    stripped = raw_text.strip()
    if not stripped.startswith("Action:"):
        if stripped.startswith("Final:"):
            return _Decision(kind="final", final_text=stripped[len("Final:") :].strip())
        return _Decision(kind="final", final_text=stripped)
    return _parse_action_response(stripped[len("Action:") :].strip())


def _parse_action_response(payload_text: str) -> _Decision:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return _Decision(kind="action", parse_error=f"Invalid Action payload: {exc}")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return _Decision(kind="action", parse_error="Action payload must be a JSON object or array.")

    tool_calls: list[ToolCall] = []
    for entry in payload:
        if not isinstance(entry, dict):
            return _Decision(kind="action", parse_error="Each Action entry must be a JSON object.")
        tool_name = entry.get("tool")
        arguments = entry.get("arguments", {})
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            return _Decision(
                kind="action",
                parse_error="Each Action entry must include string `tool` and object `arguments`.",
            )
        tool_calls.append(ToolCall(name=tool_name, arguments=dict(arguments), tool_id=uuid4().hex))
    return _Decision(kind="action", tool_calls=tool_calls)


def _wrap_child_run_fn(
    run_fn: Callable[[Agent], None],
    *,
    pinned_skill_names: Sequence[str],
) -> Callable[[Agent], None]:
    def wrapped(agent: Agent) -> None:
        agent.prompt.metadata.setdefault("preload_skills", list(pinned_skill_names))
        return run_fn(agent)

    return wrapped
