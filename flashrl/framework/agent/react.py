"""Minimal whitebox ReAct rollout with subprocess-backed tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any, Callable, Sequence
from uuid import uuid4

from flashrl.framework.data_models import (
    AssistantTurn,
    Conversation,
    Message,
    Prompt,
    RolloutOutput,
    ToolCall,
    ToolResult,
)
from flashrl.framework.tools import SubprocessToolRuntime, Tool


SystemPromptValue = str | Callable[[Prompt], str | None] | None

DEFAULT_REACT_INSTRUCTION = (
    "You are a tool-using assistant.\n"
    "When you need tools, respond with exactly one line that starts with "
    "`Action:` followed by either one JSON object or a JSON array of objects.\n"
    "Each object must have the keys `tool` and `arguments`.\n"
    "When you are ready to answer the user, respond with `Final:` followed by "
    "the final answer content.\n"
    "Do not use any other output format."
)


class ReActRollout:
    """Callable whitebox rollout that runs a simple ReAct loop."""

    def __init__(
        self,
        *,
        tools: Sequence[Tool],
        max_steps: int,
        system_prompt: SystemPromptValue = None,
        instruction: str | None = None,
        runtime: SubprocessToolRuntime | None = None,
        max_parallel_calls: int = 4,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if max_parallel_calls < 1:
            raise ValueError("max_parallel_calls must be >= 1.")
        self.tools = list(tools)
        self.tool_map = self._build_tool_map(self.tools)
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.instruction = instruction or DEFAULT_REACT_INSTRUCTION
        self.runtime = runtime or SubprocessToolRuntime()
        self.max_parallel_calls = max_parallel_calls

    def __call__(
        self,
        prompts: list[Prompt],
        serving_backend: Any,
    ) -> list[RolloutOutput]:
        """Run one whitebox rollout per prompt."""
        if not prompts:
            return []

        states = [self._build_state(prompt) for prompt in prompts]
        active_indices = list(range(len(states)))

        for _step_number in range(self.max_steps):
            if not active_indices:
                break

            rendered_prompts = [
                self._render_prompt(states[index]["prompt"], states[index]["conversation"])
                for index in active_indices
            ]
            samples = serving_backend.generate_batch(rendered_prompts)
            next_active_indices: list[int] = []

            for state_index, sample in zip(active_indices, samples, strict=True):
                state = states[state_index]
                state["assistant_turns"].append(
                    AssistantTurn(
                        prompt_token_ids=list(sample.prompt_token_ids),
                        response_token_ids=list(sample.response_token_ids),
                        response_token_logprobs=list(sample.response_token_logprobs),
                    )
                )
                state["last_sample"] = sample
                raw_text = str(sample.text)
                decision = self._parse_decision(raw_text)

                if decision["kind"] == "action":
                    requested_calls = [
                        ToolCall(
                            name=str(call["tool"]),
                            arguments=dict(call["arguments"]),
                            tool_id=uuid4().hex,
                        )
                        for call in decision["calls"]
                    ]
                    state["conversation"].add_message(
                        Message(
                            role="assistant",
                            content=raw_text,
                            tool_calls=requested_calls,
                        )
                    )
                    state["last_assistant_text"] = raw_text
                    parse_error = decision.get("parse_error")
                    if parse_error:
                        tool_results = [
                            (
                                ToolCall(
                                    name="invalid_action",
                                    arguments={},
                                    tool_id=uuid4().hex,
                                ),
                                ToolResult(
                                    content=str(parse_error),
                                    error=True,
                                    metadata={"status": "parse_error"},
                                ),
                            )
                        ]
                    else:
                        tool_results = self._execute_tool_calls(
                            requested_calls,
                            prompt=state["prompt"],
                        )
                    for tool_call, tool_result in tool_results:
                        tool_metadata = {
                            **dict(tool_result.metadata),
                            "tool_id": tool_call.tool_id,
                            "tool_name": tool_call.name,
                            "error": bool(tool_result.error),
                        }
                        state["conversation"].add_message(
                            Message(
                                role="tool",
                                content=tool_result.content,
                                metadata=tool_metadata,
                            )
                        )
                    next_active_indices.append(state_index)
                    continue

                final_text = str(decision["final_text"])
                state["conversation"].add_message(
                    Message(role="assistant", content=raw_text)
                )
                state["last_assistant_text"] = raw_text
                state["final_text"] = final_text
                state["stop_reason"] = str(decision["stop_reason"])
                state["done"] = True

            active_indices = next_active_indices

        for state in states:
            if state["done"]:
                continue
            state["done"] = True
            state["stop_reason"] = "max_steps"
            if not state["final_text"]:
                state["final_text"] = str(state["last_assistant_text"])

        return [self._build_rollout_output(state) for state in states]

    def _build_tool_map(self, tools: Sequence[Tool]) -> dict[str, Tool]:
        """Build one name-indexed tool map and reject duplicates."""
        tool_map: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in tool_map:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            tool_map[tool.name] = tool
        return tool_map

    def _build_state(self, prompt: Prompt) -> dict[str, Any]:
        """Build one rollout state for a prompt."""
        messages: list[Message] = []
        system_prompt = self._resolve_system_prompt(prompt)
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt.text))
        return {
            "prompt": prompt,
            "conversation": Conversation(messages=messages),
            "assistant_turns": [],
            "done": False,
            "stop_reason": None,
            "final_text": "",
            "last_assistant_text": "",
            "last_sample": None,
        }

    def _resolve_system_prompt(self, prompt: Prompt) -> str | None:
        """Resolve the optional user-supplied system prompt."""
        if self.system_prompt is None:
            return None
        if callable(self.system_prompt):
            resolved = self.system_prompt(prompt)
            if resolved is None:
                return None
            return str(resolved)
        return str(self.system_prompt)

    def _render_prompt(self, prompt: Prompt, conversation: Conversation) -> str:
        """Render one string prompt for the current serving backend."""
        system_messages = [
            message.content for message in conversation.messages if message.role == "system"
        ]
        transcript_messages = [
            message for message in conversation.messages if message.role != "system"
        ]
        transcript_lines: list[str] = []
        for message in transcript_messages:
            if message.role == "user":
                transcript_lines.append(f"User: {message.content}")
            elif message.role == "assistant":
                transcript_lines.append(f"Assistant: {message.content}")
            elif message.role == "tool":
                tool_name = str(message.metadata.get("tool_name", "tool"))
                transcript_lines.append(f"Tool[{tool_name}]: {message.content}")
            else:
                transcript_lines.append(f"{message.role.title()}: {message.content}")

        tool_lines = [
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ] or ["- no tools available"]
        sections: list[str] = []
        if system_messages:
            sections.append("System Prompt:\n" + "\n\n".join(system_messages))
        sections.append(
            "Tool Contract:\n"
            f"{self.instruction}\n"
            f"Maximum parallel tool calls per step: {self.max_parallel_calls}\n"
            "Available tools:\n"
            + "\n".join(tool_lines)
        )
        sections.append("Transcript:\n" + "\n".join(transcript_lines))
        sections.append(
            "Reply with either `Action: {...}`, `Action: [{...}, {...}]`, or `Final: ...`."
        )
        return "\n\n".join(section for section in sections if section).strip()

    def _parse_decision(self, raw_text: str) -> dict[str, Any]:
        """Parse one assistant response into an action or final answer."""
        stripped = raw_text.strip()
        if stripped.startswith("Action:"):
            payload_text = stripped[len("Action:"):].strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                return {
                    "kind": "action",
                    "calls": [],
                    "parse_error": f"Invalid Action payload: {exc}",
                }
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                return {
                    "kind": "action",
                    "calls": [],
                    "parse_error": "Action payload must be a JSON object or array.",
                }
            normalized_calls: list[dict[str, Any]] = []
            for entry in payload:
                if not isinstance(entry, dict):
                    return {
                        "kind": "action",
                        "calls": [],
                        "parse_error": "Each Action entry must be a JSON object.",
                    }
                tool_name = entry.get("tool")
                arguments = entry.get("arguments", {})
                if not isinstance(tool_name, str) or not tool_name:
                    return {
                        "kind": "action",
                        "calls": [],
                        "parse_error": "Each Action entry must include a non-empty string `tool`.",
                    }
                if not isinstance(arguments, dict):
                    return {
                        "kind": "action",
                        "calls": [],
                        "parse_error": "Each Action entry must use a JSON object for `arguments`.",
                    }
                normalized_calls.append({"tool": tool_name, "arguments": dict(arguments)})
            return {"kind": "action", "calls": normalized_calls, "parse_error": None}
        if stripped.startswith("Final:"):
            return {
                "kind": "final",
                "final_text": stripped[len("Final:"):].strip(),
                "stop_reason": "final",
            }
        return {
            "kind": "final",
            "final_text": raw_text,
            "stop_reason": "unstructured_final",
        }

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        prompt: Prompt,
    ) -> list[tuple[ToolCall, ToolResult]]:
        """Execute one assistant step worth of tool calls."""
        if not tool_calls:
            return []
        if len(tool_calls) > self.max_parallel_calls:
            message = (
                "Too many parallel tool calls requested "
                f"(max {self.max_parallel_calls}, got {len(tool_calls)})."
            )
            return [
                (
                    ToolCall(name="parallel_limit", arguments={}, tool_id=uuid4().hex),
                    ToolResult(
                        content=message,
                        error=True,
                        metadata={"status": "parallel_limit"},
                    ),
                )
            ]

        def run_tool_call(tool_call: ToolCall) -> tuple[ToolCall, ToolResult]:
            tool = self.tool_map.get(tool_call.name)
            if tool is None:
                return (
                    tool_call,
                    ToolResult(
                        content=f"Unknown tool: {tool_call.name}",
                        error=True,
                        metadata={"status": "unknown_tool"},
                    ),
                )
            return (
                tool_call,
                self.runtime.execute(tool, arguments=tool_call.arguments, prompt=prompt),
            )

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), self.max_parallel_calls)) as executor:
            return list(executor.map(run_tool_call, tool_calls))

    def _build_rollout_output(self, state: dict[str, Any]) -> RolloutOutput:
        """Convert one finished state into the rollout shape expected by FlashRL."""
        last_sample = state["last_sample"]
        assistant_turns = list(state["assistant_turns"])
        final_prompt_token_ids = (
            list(last_sample.prompt_token_ids)
            if last_sample is not None
            else []
        )
        final_response_token_ids = (
            list(last_sample.response_token_ids)
            if last_sample is not None
            else []
        )
        final_response_logprobs = (
            list(last_sample.response_token_logprobs)
            if last_sample is not None
            else []
        )
        log_prob = (
            float(last_sample.log_prob)
            if last_sample is not None
            else 0.0
        )
        sample_metadata = dict(getattr(last_sample, "metadata", {}) or {})
        rollout_metadata = {
            **sample_metadata,
            "prompt_metadata": dict(state["prompt"].metadata),
            "stop_reason": state["stop_reason"],
            "assistant_turn_count": len(assistant_turns),
        }
        return RolloutOutput(
            text=str(state["final_text"]),
            log_prob=log_prob,
            prompt_token_ids=final_prompt_token_ids,
            response_token_ids=final_response_token_ids,
            response_token_logprobs=final_response_logprobs,
            assistant_turns=assistant_turns,
            conversation=state["conversation"],
            metadata=rollout_metadata,
        )
