"""Traced agent building blocks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from uuid import uuid4

from flashrl.framework.agent.context import BaseContextManager
from flashrl.framework.agent.tools import SubprocessToolRuntime, Tool
from flashrl.framework.data_models import (
    AssistantTurn,
    Conversation,
    Message,
    Prompt,
    RolloutOutput,
    ToolCall,
    ToolResult,
)


@dataclass
class AgentState:
    """Per-prompt mutable state shared across one agent run."""

    prompt: Prompt
    conversation: Conversation
    metadata: dict[str, Any] = field(default_factory=dict)
    assistant_turns: list[AssistantTurn] = field(default_factory=list)
    done: bool = False
    stop_reason: str | None = None
    final_text: str = ""
    last_assistant_text: str = ""
    last_sample: AgentSample | None = None


@dataclass(frozen=True)
class AgentSample:
    """Stable generation sample returned to custom agent loops."""

    text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_token_logprobs: list[float]
    log_prob: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_backend_sample(cls, sample: Any) -> AgentSample:
        """Normalize one serving-backend sample into a stable public type."""
        return cls(
            text=str(getattr(sample, "text", "")),
            prompt_token_ids=[int(token_id) for token_id in getattr(sample, "prompt_token_ids", [])],
            response_token_ids=[int(token_id) for token_id in getattr(sample, "response_token_ids", [])],
            response_token_logprobs=[
                float(value) for value in getattr(sample, "response_token_logprobs", [])
            ],
            log_prob=float(getattr(sample, "log_prob", 0.0)),
            metadata=dict(getattr(sample, "metadata", {}) or {}),
        )


_ToolSource = Sequence[Tool] | Callable[[AgentState], Sequence[Tool]] | None
_RunFn = Callable[["Agent"], Any]


class _AgentStop(RuntimeError):
    """Internal control-flow signal used to stop one live agent run."""


class Agent:
    """Public whitebox agent type.

    Constructed directly, ``Agent`` is the rollout callable passed to FlashRL.
    Inside ``run_fn(agent)``, the same public type is used as the live per-prompt
    runtime object.
    """

    def __init__(
        self,
        run_fn: _RunFn,
        *,
        tools: _ToolSource = None,
        context_manager: BaseContextManager | None = None,
        max_steps: int,
        runtime: SubprocessToolRuntime | None = None,
        max_parallel_calls: int = 4,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if max_parallel_calls < 1:
            raise ValueError("max_parallel_calls must be >= 1.")
        self.run_fn = run_fn
        self.tools = tools
        self.context_manager = context_manager
        self.max_steps = max_steps
        self.runtime = runtime or SubprocessToolRuntime()
        self.max_parallel_calls = max_parallel_calls
        if tools is not None and not callable(tools):
            self._validate_unique_tools(list(tools))
        self._is_runtime = False
        self._serving_backend: Any | None = None
        self._state: AgentState | None = None
        self._generation_count = 0
        self._pending_sample: AgentSample | None = None
        self._last_available_tool_names: list[str] = []

    @classmethod
    def _create_runtime(
        cls,
        blueprint: Agent,
        *,
        prompt: Prompt,
        serving_backend: Any,
    ) -> Agent:
        runtime = cls.__new__(cls)
        runtime.run_fn = blueprint.run_fn
        runtime.tools = blueprint.tools
        runtime.context_manager = blueprint.context_manager
        runtime.max_steps = blueprint.max_steps
        runtime.runtime = blueprint.runtime
        runtime.max_parallel_calls = blueprint.max_parallel_calls
        runtime._is_runtime = True
        runtime._serving_backend = serving_backend
        runtime._state = AgentState(
            prompt=prompt,
            conversation=Conversation(messages=[Message(role="user", content=prompt.text)]),
        )
        runtime._generation_count = 0
        runtime._pending_sample = None
        runtime._last_available_tool_names = []
        return runtime

    @property
    def state(self) -> AgentState:
        """Return the advanced mutable state for the current live run."""
        self._ensure_runtime_context("state")
        assert self._state is not None
        return self._state

    @property
    def prompt(self) -> Prompt:
        """Return the current prompt."""
        return self.state.prompt

    @property
    def conversation(self) -> Conversation:
        """Return the traced conversation."""
        return self.state.conversation

    @property
    def metadata(self) -> dict[str, Any]:
        """Return rollout metadata for the current prompt."""
        return self.state.metadata

    @property
    def done(self) -> bool:
        """Return whether the current live run has been finished."""
        return self.state.done

    def __call__(
        self,
        prompts: list[Prompt],
        serving_backend: Any,
    ) -> list[RolloutOutput]:
        """Convenience adapter for direct/manual rollout invocation."""
        self._ensure_blueprint_context("__call__")
        return self.run_batch(prompts, serving_backend)

    def run_batch(
        self,
        prompts: list[Prompt],
        serving_backend: Any,
    ) -> list[RolloutOutput]:
        """Run the traced agent loop once per prompt."""
        self._ensure_blueprint_context("run_batch")
        rollouts: list[RolloutOutput] = []
        for prompt in prompts:
            agent = self._create_runtime(
                self,
                prompt=prompt,
                serving_backend=serving_backend,
            )
            try:
                self.run_fn(agent)
            except _AgentStop:
                pass

            if agent._pending_sample is not None:
                raise RuntimeError(
                    "Agent run returned with an unrecorded sample. "
                    "Call agent.record_generation(sample, ...) before finishing or returning."
                )

            state = agent.state
            if state.stop_reason == "max_steps":
                state.final_text = str(state.last_assistant_text)
                state.done = True
            elif not state.done:
                state.final_text = str(state.last_assistant_text)
                state.stop_reason = "run_returned"
                state.done = True
            rollouts.append(self._build_rollout_output(state))
        return rollouts

    def available_tools(self) -> list[Tool]:
        """Resolve the current step-local tool set and record it for the trace."""
        resolved_tools = self._resolve_tools(self.state)
        self._last_available_tool_names = [tool.name for tool in resolved_tools]
        self._record_visible_tools(self.state, self._last_available_tool_names)
        return resolved_tools

    def build_prompt(
        self,
        *,
        tools: Sequence[Tool] | None = None,
        footer: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> str:
        """Build one plain-text, completion-ready agent prompt."""
        sections: list[str] = []
        if tools is not None:
            tool_lines = [f"- {tool.name}: {tool.description}" for tool in tools]
            if not tool_lines:
                tool_lines = ["- no tools available on this step"]
            sections.append("Available tools:\n" + "\n".join(tool_lines))

        visible_messages = list(messages) if messages is not None else self._build_context_messages()
        rendered_transcript = self._render_messages(visible_messages)
        transcript_section = "Transcript:"
        if rendered_transcript:
            transcript_section += "\n" + rendered_transcript
        sections.append(transcript_section)

        if footer is not None:
            normalized_footer = str(footer).strip()
            if normalized_footer:
                sections.append(normalized_footer)

        sections.append("Assistant:")
        return "\n\n".join(section for section in sections if section).strip()

    def generate(self, prompt: str) -> AgentSample:
        """Generate one assistant sample for the current step."""
        self._ensure_not_finished("generate")
        self._ensure_no_pending("generate")
        if self._generation_count >= self.max_steps:
            self.state.done = True
            self.state.stop_reason = "max_steps"
            raise _AgentStop("Agent reached max_steps.")
        if self._serving_backend is None:
            raise RuntimeError("Agent.generate() is only available inside run_fn(agent).")

        samples = self._serving_backend.generate_batch([str(prompt)])
        if len(samples) != 1:
            raise ValueError(
                "Agent.generate expected exactly one backend sample "
                f"(got {len(samples)})."
            )
        sample = AgentSample.from_backend_sample(samples[0])
        self._pending_sample = sample
        self._generation_count += 1
        return sample

    def record_generation(
        self,
        sample: AgentSample,
        *,
        tool_calls: list[ToolCall] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Append one assistant message and one trainable assistant turn."""
        self._ensure_not_finished("record_generation")
        if self._pending_sample is None:
            raise RuntimeError(
                "Agent.record_generation() requires the most recent sample returned by generate()."
            )
        if sample is not self._pending_sample:
            raise RuntimeError(
                "Agent.record_generation() must use the latest sample returned by generate()."
            )
        message = self._append_message(
            "assistant",
            sample.text,
            tool_calls=tool_calls,
            metadata=self._merge_message_metadata(metadata),
        )
        self.state.assistant_turns.append(
            AssistantTurn(
                prompt_token_ids=list(sample.prompt_token_ids),
                response_token_ids=list(sample.response_token_ids),
                response_token_logprobs=list(sample.response_token_logprobs),
            )
        )
        self.state.last_sample = sample
        self.state.last_assistant_text = sample.text
        self._pending_sample = None
        return message

    def run_tools(
        self,
        tool_calls: list[ToolCall],
        *,
        tools: Sequence[Tool] | None = None,
    ) -> list[tuple[ToolCall, ToolResult]]:
        """Execute tools safely, record tool messages, and preserve declared order."""
        self._ensure_not_finished("run_tools")
        self._ensure_no_pending("run_tools")
        resolved_tools = list(tools) if tools is not None else self.available_tools()
        self._validate_unique_tools(list(resolved_tools))
        if tools is not None:
            self._last_available_tool_names = [tool.name for tool in resolved_tools]
        available_tool_map = self._build_tool_map(list(resolved_tools))
        tool_results = self._execute_tool_calls(
            tool_calls,
            available_tool_map=available_tool_map,
            prompt=self.state.prompt,
        )
        for tool_call, tool_result in tool_results:
            tool_metadata = {
                **dict(tool_result.metadata),
                "tool_id": tool_call.tool_id,
                "tool_name": tool_call.name,
                "error": bool(tool_result.error),
            }
            if self._last_available_tool_names:
                tool_metadata.setdefault("available_tool_names", list(self._last_available_tool_names))
            self._append_message("tool", tool_result.content, metadata=tool_metadata)
        return tool_results

    def add_message(
        self,
        role: str,
        content: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Append one explicit trace message outside the generated assistant path."""
        self._ensure_not_finished("add_message")
        self._ensure_no_pending("add_message")
        return self._append_message(
            role,
            content,
            tool_calls=tool_calls,
            metadata=metadata,
        )

    def finish(
        self,
        text: str | None = None,
        *,
        stop_reason: str = "final",
    ) -> None:
        """Finalize the current live agent run."""
        self._ensure_runtime_context("finish")
        if self.state.done:
            return
        self._ensure_no_pending("finish")
        self.state.final_text = str(text if text is not None else self.state.last_assistant_text)
        self.state.stop_reason = str(stop_reason)
        self.state.done = True

    def _build_context_messages(self) -> list[Message]:
        self._ensure_runtime_context("build_prompt")
        if self.context_manager is None:
            system_messages = [
                message.model_copy(deep=True)
                for message in self.state.conversation.messages
                if message.role == "system"
            ]
            other_messages = [
                message.model_copy(deep=True)
                for message in self.state.conversation.messages
                if message.role != "system"
            ]
            return system_messages + other_messages
        return [message.model_copy(deep=True) for message in self.context_manager.build_messages(self.state)]

    def _append_message(
        self,
        role: str,
        content: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        self._ensure_runtime_context("add_message")
        message = Message(
            role=str(role),
            content=str(content),
            tool_calls=list(tool_calls or []),
            metadata=dict(metadata or {}),
        )
        if message.role == "system":
            system_count = sum(
                1
                for existing in self.state.conversation.messages
                if existing.role == "system"
            )
            self.state.conversation.messages.insert(system_count, message)
        else:
            self.state.conversation.add_message(message)
        if message.role == "assistant":
            self.state.last_assistant_text = message.content
        if self.context_manager is not None:
            self.context_manager.observe(self.state, [message])
        return message

    def _merge_message_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(metadata or {})
        if self._last_available_tool_names:
            merged.setdefault("available_tool_names", list(self._last_available_tool_names))
        return merged

    def _render_messages(self, messages: Sequence[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            if message.role == "system":
                lines.append(f"System: {message.content}")
            elif message.role == "user":
                lines.append(f"User: {message.content}")
            elif message.role == "assistant":
                lines.append(f"Assistant: {message.content}")
            elif message.role == "tool":
                tool_name = str(message.metadata.get("tool_name", "tool"))
                lines.append(f"Tool[{tool_name}]: {message.content}")
            else:
                lines.append(f"{message.role.title()}: {message.content}")
        return "\n".join(lines).strip()

    def _ensure_runtime_context(self, action: str) -> None:
        if self._is_runtime and self._state is not None:
            return
        raise RuntimeError(f"Agent.{action}() is only available inside run_fn(agent).")

    def _ensure_blueprint_context(self, action: str) -> None:
        if not self._is_runtime:
            return
        raise RuntimeError(f"Agent.{action}() is only available on the rollout blueprint.")

    def _ensure_not_finished(self, action: str) -> None:
        self._ensure_runtime_context(action)
        if self.state.done:
            raise RuntimeError(f"Cannot call Agent.{action}() after finish().")

    def _ensure_no_pending(self, action: str) -> None:
        if self._pending_sample is None:
            return
        raise RuntimeError(
            "Cannot call Agent."
            f"{action}() while the latest sample from generate() is still unrecorded. "
            "Call record_generation(sample, ...) first."
        )

    def _resolve_tools(self, state: AgentState) -> list[Tool]:
        """Resolve the tool set visible to the current step."""
        if self.tools is None:
            return []
        if callable(self.tools):
            resolved_tools = list(self.tools(state))
        else:
            resolved_tools = list(self.tools)
        self._validate_unique_tools(resolved_tools)
        return resolved_tools

    def _validate_unique_tools(self, tools: list[Tool]) -> None:
        """Reject duplicate tool names within one visible tool set."""
        tool_names: set[str] = set()
        for tool in tools:
            if tool.name in tool_names:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            tool_names.add(tool.name)

    def _build_tool_map(self, tools: list[Tool]) -> dict[str, Tool]:
        """Build one name-indexed tool map for the current step."""
        return {tool.name: tool for tool in tools}

    def _record_visible_tools(
        self,
        state: AgentState,
        available_tool_names: list[str],
    ) -> None:
        """Persist the step-local visible tool list for later inspection."""
        rollout_visible_tools = state.metadata.get("visible_tools_by_step")
        if not isinstance(rollout_visible_tools, list):
            rollout_visible_tools = []
            state.metadata["visible_tools_by_step"] = rollout_visible_tools
        rollout_visible_tools.append(list(available_tool_names))

        conversation_visible_tools = state.conversation.metadata.get("visible_tools_by_step")
        if not isinstance(conversation_visible_tools, list):
            conversation_visible_tools = []
            state.conversation.metadata["visible_tools_by_step"] = conversation_visible_tools
        conversation_visible_tools.append(list(available_tool_names))

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        available_tool_map: dict[str, Tool],
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
            return (
                tool_call,
                self.runtime.execute(tool, arguments=tool_call.arguments, prompt=prompt),
            )

        with ThreadPoolExecutor(
            max_workers=min(len(tool_calls), self.max_parallel_calls)
        ) as executor:
            return list(executor.map(run_tool_call, tool_calls))

    def _build_rollout_output(self, state: AgentState) -> RolloutOutput:
        """Convert one finished state into the rollout shape expected by FlashRL."""
        last_sample = state.last_sample
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
        log_prob = float(last_sample.log_prob) if last_sample is not None else 0.0
        sample_metadata = dict(last_sample.metadata) if last_sample is not None else {}
        rollout_metadata = {
            **sample_metadata,
            **dict(state.metadata),
            "prompt_metadata": dict(state.prompt.metadata),
            "stop_reason": state.stop_reason,
            "assistant_turn_count": len(state.assistant_turns),
        }
        return RolloutOutput(
            text=str(state.final_text),
            log_prob=log_prob,
            prompt_token_ids=final_prompt_token_ids,
            response_token_ids=final_response_token_ids,
            response_token_logprobs=final_response_logprobs,
            assistant_turns=list(state.assistant_turns),
            conversation=state.conversation,
            metadata=rollout_metadata,
        )
