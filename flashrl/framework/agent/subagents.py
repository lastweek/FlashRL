"""Bounded local child-agent execution for serious harnesses."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Callable, Sequence

from flashrl.framework.data_models import Prompt, ToolCall, ToolResult

if False:  # pragma: no cover
    from flashrl.framework.agent.runtime import Agent


SubagentBuilder = Callable[["Agent", ToolCall], "Agent"]


@dataclass
class SubagentManager:
    """Execute local child agents and keep their traces off the trainable path."""

    build_agent: SubagentBuilder
    max_parallel: int = 2
    max_per_parent_turn: int = 2
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.max_parallel < 1:
            raise ValueError("SubagentManager.max_parallel must be >= 1.")
        if self.max_per_parent_turn < 1:
            raise ValueError("SubagentManager.max_per_parent_turn must be >= 1.")
        if self.timeout_seconds <= 0.0:
            raise ValueError("SubagentManager.timeout_seconds must be > 0.")

    def execute_batch(
        self,
        tool_calls: list[ToolCall],
        *,
        agent: "Agent",
        tool,
    ) -> list[ToolResult]:
        del tool
        if len(tool_calls) > self.max_per_parent_turn:
            return [
                ToolResult(
                    content=(
                        "Too many subagent calls requested in one assistant turn "
                        f"(max {self.max_per_parent_turn}, got {len(tool_calls)})."
                    ),
                    error=True,
                    metadata={"status": "subagent_limit"},
                )
            ] + [
                ToolResult(
                    content="Subagent call skipped after limit error.",
                    error=True,
                    metadata={"status": "subagent_skipped"},
                )
                for _ in range(len(tool_calls) - 1)
            ]

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), self.max_parallel)) as executor:
            futures = [
                executor.submit(self._run_one, agent, tool_call)
                for tool_call in tool_calls
            ]
            results: list[ToolResult] = []
            for tool_call, future in zip(tool_calls, futures, strict=True):
                try:
                    result = future.result(timeout=self.timeout_seconds)
                except TimeoutError:
                    result = ToolResult(
                        content=f"Subagent timed out after {self.timeout_seconds:.1f}s.",
                        error=True,
                        metadata={"status": "subagent_timeout"},
                    )
                agent.agent_trace.subagents.append(
                    {
                        "tool_id": tool_call.tool_id,
                        "task": str(tool_call.arguments.get("task", "")),
                        "result": result.model_dump(mode="json"),
                    }
                )
                agent.record_trace_event(
                    "subagent_run",
                    payload={
                        "tool_id": tool_call.tool_id,
                        "status": result.metadata.get("status", "ok"),
                        "error": bool(result.error),
                    },
                )
                results.append(result)
        return results

    def _run_one(self, parent_agent: "Agent", tool_call: ToolCall) -> ToolResult:
        task = str(tool_call.arguments.get("task", "")).strip()
        if not task:
            return ToolResult(
                content="run_subagent requires a non-empty `task` argument.",
                error=True,
                metadata={"status": "invalid_arguments"},
            )
        child_agent = self.build_agent(parent_agent, tool_call)
        child_prompt = Prompt(
            text=task,
            metadata={
                **dict(parent_agent.prompt.metadata),
                "subagent_parent_tool_id": tool_call.tool_id,
            },
        )
        rollout = child_agent.run_batch([child_prompt], parent_agent._serving_backend)[0]
        return ToolResult(
            content=rollout.text,
            error=False,
            metadata={
                "status": "ok",
                "assistant_turn_count": len(rollout.assistant_turns),
                "stop_reason": rollout.metadata.get("stop_reason"),
                "child_trace": rollout.agent_trace.model_dump(mode="json"),
                "child_conversation": rollout.conversation.model_dump(mode="json"),
            },
        )


def run_subagent_placeholder(arguments: dict[str, object], prompt: Prompt) -> str:
    """Placeholder entrypoint for the model-visible run_subagent tool."""
    del arguments, prompt
    raise RuntimeError("run_subagent must be executed through SubagentManager, not the subprocess tool runtime.")
