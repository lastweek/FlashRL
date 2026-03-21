"""Reference agent harness assembled from generic FlashRL agent primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal, Sequence
from uuid import uuid4

from flashrl.examples.agent_harness.config import CodingHarnessConfig
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
from flashrl.framework.agent.session import LoadedSkill
from flashrl.framework.agent.tools import SubprocessToolRuntime
from flashrl.framework.data_models import Prompt, RewardOutput, RolloutOutput, ToolCall

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EXAMPLE_DIR / "fixtures"
SKILLS_DIR = EXAMPLE_DIR / "skills"


@dataclass(frozen=True)
class RepoTask:
    """One deterministic repo-inspection task used by the agent harness example."""

    task_id: str
    split: Literal["train", "eval"]
    repo_name: str
    question: str
    expected_answer: str
    preload_skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Decision:
    kind: Literal["action", "final"]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    parse_error: str | None = None


TASKS: tuple[RepoTask, ...] = (
    RepoTask(
        task_id="inventory-version",
        split="train",
        repo_name="inventory_repo",
        question=(
            "Inspect the repository and answer with the raw APP_VERSION value only. "
            "@skill:repo_triage"
        ),
        expected_answer="v3",
        preload_skills=("repo_triage",),
    ),
    RepoTask(
        task_id="inventory-port",
        split="train",
        repo_name="inventory_repo",
        question="Inspect the repository and answer with the default port integer only.",
        expected_answer="8080",
    ),
    RepoTask(
        task_id="service-header",
        split="train",
        repo_name="service_repo",
        question=(
            "Inspect the repository and answer with the function name that builds the auth header only. "
            "@skill:repo_triage"
        ),
        expected_answer="build_auth_header",
        preload_skills=("repo_triage",),
    ),
    RepoTask(
        task_id="service-retry",
        split="train",
        repo_name="service_repo",
        question="Inspect the repository and answer with the retry count integer only.",
        expected_answer="3",
    ),
    RepoTask(
        task_id="inventory-combo",
        split="eval",
        repo_name="inventory_repo",
        question=(
            "Inspect the repository and answer exactly `version=v3;port=8080`."
        ),
        expected_answer="version=v3;port=8080",
        preload_skills=("repo_triage",),
    ),
    RepoTask(
        task_id="service-combo",
        split="eval",
        repo_name="service_repo",
        question=(
            "Inspect the repository and answer exactly `retry=3;header=build_auth_header`. "
            "If the work can be decomposed, you may delegate one sub-question with run_subagent."
        ),
        expected_answer="retry=3;header=build_auth_header",
        preload_skills=("repo_triage",),
    ),
)


def build_coding_train_dataset(limit: int | None = None) -> list[Prompt]:
    return _build_dataset(split="train", limit=limit)


def build_coding_eval_dataset(limit: int | None = None) -> list[Prompt]:
    return _build_dataset(split="eval", limit=limit)


def _build_dataset(split: Literal["train", "eval"], limit: int | None) -> list[Prompt]:
    prompts: list[Prompt] = []
    for task in TASKS:
        if task.split != split:
            continue
        prompt = Prompt(
            text=task.question,
            metadata={
                "task_id": task.task_id,
                "source": "flashrl/examples/agent_harness",
                "split": split,
                "repo_root": str((FIXTURES_DIR / "repos" / task.repo_name).resolve()),
                "expected_answer": task.expected_answer,
                "preload_skills": list(task.preload_skills),
                "verifier": "string_exact",
            },
        )
        prompts.append(prompt)
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def build_coding_reward_fn():
    def reward_fn(rollout: RolloutOutput) -> RewardOutput:
        prompt_metadata = dict(rollout.metadata.get("prompt_metadata", {}))
        expected_answer = _normalize_answer(str(prompt_metadata.get("expected_answer", "")))
        answer = _normalize_answer(str(rollout.text))
        accuracy_pass = bool(answer == expected_answer and expected_answer)
        return RewardOutput(
            reward=1.0 if accuracy_pass else 0.0,
            metadata={
                "expected_answer": expected_answer,
                "normalized_answer": answer,
                "accuracy_pass": accuracy_pass,
            },
        )

    return reward_fn


def build_coding_agent(
    config: CodingHarnessConfig,
    *,
    runtime: SubprocessToolRuntime | None = None,
) -> Agent:
    """Build the reference agent harness from the generic framework primitives."""
    runtime = runtime or SubprocessToolRuntime(default_timeout_seconds=8.0, default_memory_limit_mb=256)
    registry = _build_tool_registry(config)
    skill_manager = SkillManager([SKILLS_DIR, *config.extra_skill_roots]) if config.enable_skills else None

    def build_child_agent(parent_agent: Agent, tool_call: ToolCall) -> Agent:
        del tool_call
        child_config = config.with_overrides(
            {
                "enable_subagents": False,
                "tool_profile": ToolProfile.SUBAGENT.value,
            }
        )
        child = build_coding_agent(child_config, runtime=runtime)
        parent_active_skills = tuple(parent_agent.session.active_skills)
        child.run_fn = _wrap_child_run_fn(child.run_fn, pinned_skill_names=parent_active_skills)  # type: ignore[method-assign]
        return child

    handlers: dict[str, Any] = {}
    if skill_manager is not None:
        handlers["load_skill"] = skill_manager
    if config.enable_subagents:
        handlers["run_subagent"] = SubagentManager(
            build_agent=build_child_agent,
            max_parallel=config.subagent_max_parallel,
            max_per_parent_turn=config.subagent_max_per_parent_turn,
            timeout_seconds=config.subagent_timeout_seconds,
        )
    tool_executor = AgentToolExecutor(
        runtime=runtime,
        max_parallel_calls=config.max_parallel_calls,
        handlers=handlers,
    )
    context_manager = (
        CompactionManager(
            policy=CompactionPolicy(
                trigger_message_count=config.compaction_trigger_message_count,
                preserve_recent_messages=config.compaction_preserve_recent_messages,
                max_summary_chars=1200,
            )
        )
        if config.enable_compaction
        else WindowedContextManager(max_messages=10)
    )

    def run(agent: Agent) -> None:
        agent.session.tool_profile = config.tool_profile
        agent.add_message(
            "system",
            _build_system_prompt(skill_catalog=skill_manager.render_catalog() if skill_manager else None),
        )
        preload_skills = list(config.pinned_skills)
        preload_skills.extend(str(item) for item in agent.prompt.metadata.get("preload_skills", []) if str(item).strip())
        if skill_manager is not None:
            preload_skills.extend(skill_manager.parse_mentions(agent.prompt.text))
            deduped_preloads = list(dict.fromkeys(preload_skills))
            agent.session.pinned_skill_names = deduped_preloads
            skill_manager.preload(agent, deduped_preloads)
        while not agent.done:
            available_tools = agent.available_tools()
            sample = agent.generate(agent.build_prompt(tools=available_tools))
            decision = _parse_response(sample.text)
            if decision.kind == "action":
                if decision.parse_error:
                    agent.record_generation(sample)
                    agent.add_message(
                        "tool",
                        decision.parse_error,
                        metadata={
                            "tool_name": "invalid_action",
                            "tool_id": uuid4().hex,
                            "error": True,
                            "status": "parse_error",
                        },
                    )
                    continue
                agent.record_generation(sample, tool_calls=decision.tool_calls)
                agent.run_tools(decision.tool_calls, tools=available_tools, executor=tool_executor)
                continue
            agent.record_generation(sample)
            agent.finish(_strip_final_prefix(decision.final_text))

    return Agent(
        run_fn=run,
        tools=registry,
        context_manager=context_manager,
        max_steps=config.max_steps,
        runtime=runtime,
        max_parallel_calls=config.max_parallel_calls,
        tool_executor=tool_executor,
    )


def evaluate_rollouts(
    rollouts: Sequence[RolloutOutput],
    rewards: Sequence[RewardOutput],
) -> dict[str, Any]:
    count = len(rollouts)
    accuracy = (
        float(sum(1.0 for reward in rewards if reward.metadata.get("accuracy_pass")) / count)
        if count
        else 0.0
    )
    token_counts = [
        int(rollout.metadata.get("response_token_count", len(rollout.response_token_ids)))
        for rollout in rollouts
    ]
    rollout_seconds = [
        float(rollout.metadata.get("generation_seconds", 0.0))
        for rollout in rollouts
    ]
    tool_calls = [
        sum(len(message.tool_calls) for message in rollout.conversation.messages if message.role == "assistant")
        for rollout in rollouts
    ]
    tool_errors = [
        sum(1 for message in rollout.conversation.messages if message.role == "tool" and message.metadata.get("error"))
        for rollout in rollouts
    ]
    skill_loads = [
        sum(1 for event in rollout.agent_trace.events if event.event_type == "skill_load")
        for rollout in rollouts
    ]
    compactions = [
        sum(1 for event in rollout.agent_trace.events if event.event_type == "compaction")
        for rollout in rollouts
    ]
    subagents = [len(rollout.agent_trace.subagents) for rollout in rollouts]
    assistant_turns = [len(rollout.assistant_turns) for rollout in rollouts]
    return {
        "eval_accuracy": accuracy,
        "mean_total_model_tokens": _mean(token_counts),
        "mean_rollout_seconds": _mean(rollout_seconds),
        "mean_assistant_turns": _mean(assistant_turns),
        "mean_tool_calls": _mean(tool_calls),
        "mean_tool_error_count": _mean(tool_errors),
        "mean_skill_loads": _mean(skill_loads),
        "mean_compactions": _mean(compactions),
        "mean_subagent_calls": _mean(subagents),
    }


def _build_tool_registry(config: CodingHarnessConfig) -> ToolRegistry:
    entrypoint_module = "flashrl.examples.agent_harness.tool_helpers"
    registrations = [
        (Tool(name="list_repo_files", description="List repository files.", entrypoint=f"{entrypoint_module}:list_repo_files"), (ToolProfile.DEFAULT.value, ToolProfile.READONLY.value, ToolProfile.SUBAGENT.value)),
        (Tool(name="read_repo_file", description="Read one repository file.", entrypoint=f"{entrypoint_module}:read_repo_file"), (ToolProfile.DEFAULT.value, ToolProfile.READONLY.value, ToolProfile.SUBAGENT.value)),
        (Tool(name="search_repo_text", description="Search for text in repository files.", entrypoint=f"{entrypoint_module}:search_repo_text"), (ToolProfile.DEFAULT.value, ToolProfile.READONLY.value, ToolProfile.SUBAGENT.value)),
        (Tool(name="run_repo_shell", description="Run a small allowlisted shell command in the repo.", entrypoint=f"{entrypoint_module}:run_repo_shell"), (ToolProfile.DEFAULT.value,)),
    ]
    if config.enable_skills:
        registrations.append(
            (
                Tool(
                    name="load_skill",
                    description="Load one explicit skill bundle by name.",
                    entrypoint="flashrl.framework.agent.skills:load_skill_placeholder",
                ),
                (ToolProfile.DEFAULT.value, ToolProfile.SUBAGENT.value),
            )
        )
    if config.enable_subagents:
        registrations.append(
            (
                Tool(
                    name="run_subagent",
                    description="Delegate one bounded sub-task to a local child agent.",
                    entrypoint="flashrl.framework.agent.subagents:run_subagent_placeholder",
                ),
                (ToolProfile.DEFAULT.value,),
            )
        )
    registry = ToolRegistry()
    for tool, profiles in registrations:
        registry.register(tool, profiles=profiles)
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
    if stripped.startswith("Action:"):
        payload_text = stripped[len("Action:") :].strip()
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
    if stripped.startswith("Final:"):
        return _Decision(kind="final", final_text=stripped[len("Final:") :].strip())
    return _Decision(kind="final", final_text=stripped)


def _normalize_answer(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized.casefold()


def _strip_final_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("Final:"):
        return stripped[len("Final:") :].strip()
    return stripped


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _wrap_child_run_fn(run_fn, *, pinned_skill_names: Sequence[str]):
    def wrapped(agent: Agent) -> None:
        if pinned_skill_names:
            agent.prompt.metadata.setdefault("preload_skills", list(pinned_skill_names))
        return run_fn(agent)

    return wrapped
