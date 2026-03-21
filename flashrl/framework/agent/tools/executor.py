"""Dedicated tool execution layer for agent harnesses."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence
from uuid import uuid4

from flashrl.framework.agent.tools.runtime import SubprocessToolRuntime, Tool
from flashrl.framework.data_models import ToolCall, ToolResult

if False:  # pragma: no cover
    from flashrl.framework.agent.runtime import Agent


class ToolHandler(Protocol):
    """Special-case tool handler interface."""

    def execute(
        self,
        tool_call: ToolCall,
        *,
        agent: "Agent",
        tool: Tool,
    ) -> ToolResult:
        """Execute one tool call."""


class BatchToolHandler(Protocol):
    """Batch special-case tool handler interface."""

    def execute_batch(
        self,
        tool_calls: list[ToolCall],
        *,
        agent: "Agent",
        tool: Tool,
    ) -> list[ToolResult]:
        """Execute one consecutive batch of tool calls."""


@dataclass
class AgentToolExecutor:
    """Validate, execute, and record one assistant step worth of tool calls."""

    runtime: SubprocessToolRuntime
    max_parallel_calls: int = 4
    handlers: dict[str, ToolHandler | BatchToolHandler] | None = None

    def __post_init__(self) -> None:
        if self.max_parallel_calls < 1:
            raise ValueError("AgentToolExecutor.max_parallel_calls must be >= 1.")
        if self.handlers is None:
            self.handlers = {}

    def execute(
        self,
        agent: "Agent",
        tool_calls: list[ToolCall],
        *,
        tools: Sequence[Tool],
    ) -> list[tuple[ToolCall, ToolResult]]:
        if not tool_calls:
            return []
        if len(tool_calls) > self.max_parallel_calls:
            result = ToolResult(
                content=(
                    "Too many parallel tool calls requested "
                    f"(max {self.max_parallel_calls}, got {len(tool_calls)})."
                ),
                error=True,
                metadata={"status": "parallel_limit"},
            )
            call = ToolCall(name="parallel_limit", arguments={}, tool_id=uuid4().hex)
            self._append_tool_messages(agent, [(call, result)])
            self._record_trace(agent, [(call, result)])
            return [(call, result)]

        available_tool_map = self._build_tool_map(list(tools))
        if any(self._supports_batch(tool_call, available_tool_map) for tool_call in tool_calls):
            results = self._execute_segmented(agent, tool_calls, available_tool_map=available_tool_map)
        else:
            results = self._execute_parallel(agent, tool_calls, available_tool_map=available_tool_map)

        self._append_tool_messages(agent, results)
        self._record_trace(agent, results)
        return results

    def _execute_parallel(
        self,
        agent: "Agent",
        tool_calls: list[ToolCall],
        *,
        available_tool_map: dict[str, Tool],
    ) -> list[tuple[ToolCall, ToolResult]]:
        def run_tool_call(tool_call: ToolCall) -> tuple[ToolCall, ToolResult]:
            return self._execute_one(agent, tool_call, available_tool_map=available_tool_map)

        with ThreadPoolExecutor(
            max_workers=min(len(tool_calls), self.max_parallel_calls)
        ) as executor:
            return list(executor.map(run_tool_call, tool_calls))

    def _execute_segmented(
        self,
        agent: "Agent",
        tool_calls: list[ToolCall],
        *,
        available_tool_map: dict[str, Tool],
    ) -> list[tuple[ToolCall, ToolResult]]:
        results: list[tuple[ToolCall, ToolResult]] = []
        index = 0
        while index < len(tool_calls):
            tool_call = tool_calls[index]
            tool = available_tool_map.get(tool_call.name)
            handler = self.handlers.get(tool_call.name) if tool is not None else None
            batch_handler = getattr(handler, "execute_batch", None)
            if callable(batch_handler):
                batch_calls = [tool_call]
                index += 1
                while index < len(tool_calls) and tool_calls[index].name == tool_call.name:
                    batch_calls.append(tool_calls[index])
                    index += 1
                batch_results = batch_handler(batch_calls, agent=agent, tool=tool)
                if len(batch_results) != len(batch_calls):
                    raise ValueError(
                        f"Batch tool handler for '{tool_call.name}' returned "
                        f"{len(batch_results)} results for {len(batch_calls)} calls."
                    )
                results.extend(zip(batch_calls, batch_results, strict=True))
                continue

            results.append(self._execute_one(agent, tool_call, available_tool_map=available_tool_map))
            index += 1
        return results

    def _execute_one(
        self,
        agent: "Agent",
        tool_call: ToolCall,
        *,
        available_tool_map: dict[str, Tool],
    ) -> tuple[ToolCall, ToolResult]:
        tool = available_tool_map.get(tool_call.name)
        if tool is None:
            return (
                tool_call,
                ToolResult(
                    content=f"Tool not available on this step: {tool_call.name}",
                    error=True,
                    metadata={"status": "tool_unavailable"},
                ),
            )

        handler = self.handlers.get(tool_call.name)
        execute = getattr(handler, "execute", None)
        if callable(execute):
            return tool_call, execute(tool_call, agent=agent, tool=tool)
        return (
            tool_call,
            self.runtime.execute(tool, arguments=tool_call.arguments, prompt=agent.prompt),
        )

    def _supports_batch(
        self,
        tool_call: ToolCall,
        available_tool_map: dict[str, Tool],
    ) -> bool:
        if tool_call.name not in available_tool_map:
            return False
        handler = self.handlers.get(tool_call.name)
        return callable(getattr(handler, "execute_batch", None))

    def _append_tool_messages(
        self,
        agent: "Agent",
        results: list[tuple[ToolCall, ToolResult]],
    ) -> None:
        for tool_call, tool_result in results:
            tool_metadata = {
                **dict(tool_result.metadata),
                "tool_id": tool_call.tool_id,
                "tool_name": tool_call.name,
                "error": bool(tool_result.error),
            }
            available_names = list(getattr(agent, "_last_available_tool_names", []))
            if available_names:
                tool_metadata.setdefault("available_tool_names", available_names)
            agent.add_message("tool", tool_result.content, metadata=tool_metadata)

    def _record_trace(
        self,
        agent: "Agent",
        results: list[tuple[ToolCall, ToolResult]],
    ) -> None:
        agent.record_trace_event(
            "tool_batch",
            payload={
                "tool_calls": [
                    {
                        "tool_name": tool_call.name,
                        "tool_id": tool_call.tool_id,
                        "arguments": dict(tool_call.arguments),
                        "status": str(tool_result.metadata.get("status", "")),
                        "error": bool(tool_result.error),
                    }
                    for tool_call, tool_result in results
                ]
            },
        )

    def _build_tool_map(self, tools: list[Tool]) -> dict[str, Tool]:
        tool_map: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in tool_map:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            tool_map[tool.name] = tool
        return tool_map

