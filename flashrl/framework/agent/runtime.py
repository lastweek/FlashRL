"""Traced agent building blocks."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Sequence

from flashrl.framework.agent.context import BaseContextManager
from flashrl.framework.agent.session import SessionContext
from flashrl.framework.agent.tools import (
    AgentToolExecutor,
    SubprocessToolRuntime,
    Tool,
    ToolProfile,
    ToolRegistry,
)
from flashrl.framework.data_models import (
    AgentTrace,
    AgentTraceEvent,
    AssistantTurn,
    Conversation,
    Message,
    Prompt,
    RolloutOutput,
    ToolCall,
)


@dataclass
class AgentState:
    """Per-prompt mutable state shared across one agent run."""

    prompt: Prompt
    conversation: Conversation
    metadata: dict[str, Any] = field(default_factory=dict)
    session_context: SessionContext | None = None
    agent_trace: AgentTrace = field(default_factory=AgentTrace)
    assistant_turns: list[AssistantTurn] = field(default_factory=list)
    done: bool = False
    stop_reason: str | None = None
    final_text: str = ""
    last_assistant_text: str = ""
    last_sample: AgentSample | None = None

    def __post_init__(self) -> None:
        if self.session_context is None:
            self.session_context = SessionContext(
                conversation=self.conversation,
                metadata=self.metadata,
            )
        else:
            self.session_context.conversation = self.conversation
            self.session_context.metadata = self.metadata


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


ResolvedToolSource = Sequence[Tool] | ToolRegistry
ToolSource = ResolvedToolSource | Callable[[AgentState], ResolvedToolSource] | None
RunFn = Callable[["Agent"], Any]


class _AgentStop(RuntimeError):
    """Internal control-flow signal used to stop one live agent run."""


@dataclass
class _GroupedGenerationRequest:
    """One pending generation request waiting for the grouped scheduler."""

    runtime: "Agent"
    prompt_text: str
    step_index: int
    prompt_index: int
    candidate_index: int
    future: Future[AgentSample] = field(default_factory=Future)


class _GroupedAgentScheduler:
    """Cooperative ready-trajectory scheduler for grouped whitebox rollouts."""

    def __init__(
        self,
        *,
        serving_backend: Any,
        group_size: int,
        active_runtime_count: int,
    ) -> None:
        self.serving_backend = serving_backend
        self.group_size = group_size
        self._active_runtime_count = active_runtime_count
        self._pending_requests: list[_GroupedGenerationRequest] = []
        self._condition = threading.Condition()
        self._error: BaseException | None = None

    def submit(
        self,
        runtime: "Agent",
        *,
        prompt_text: str,
        step_index: int,
    ) -> AgentSample:
        if runtime.state.session_context.prompt_index is None:
            raise RuntimeError("Grouped agent runtime is missing prompt_index.")
        if runtime.state.session_context.candidate_index is None:
            raise RuntimeError("Grouped agent runtime is missing candidate_index.")
        request = _GroupedGenerationRequest(
            runtime=runtime,
            prompt_text=prompt_text,
            step_index=step_index,
            prompt_index=runtime.state.session_context.prompt_index,
            candidate_index=runtime.state.session_context.candidate_index,
        )
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Grouped scheduler has already failed.") from self._error
            self._pending_requests.append(request)
            self._condition.notify_all()
        return request.future.result()

    def mark_runtime_finished(self, error: BaseException | None = None) -> None:
        with self._condition:
            if error is not None and self._error is None:
                self._error = error
            self._active_runtime_count -= 1
            self._condition.notify_all()

    def run(self) -> None:
        while True:
            batch = self._wait_for_batch()
            if not batch:
                break
            self._dispatch_batch(batch)
        if self._error is not None:
            raise RuntimeError("Grouped agent scheduling failed.") from self._error

    def _wait_for_batch(self) -> list[_GroupedGenerationRequest]:
        with self._condition:
            while not self._pending_requests and self._active_runtime_count > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                pending = list(self._pending_requests)
                self._pending_requests.clear()
                for request in pending:
                    if not request.future.done():
                        request.future.set_exception(
                            RuntimeError("Grouped scheduler aborted because another trajectory failed.")
                        )
                return []
            if not self._pending_requests and self._active_runtime_count == 0:
                return []
            self._condition.wait(timeout=0.01)
            batch = list(self._pending_requests)
            self._pending_requests.clear()
            return batch

    def _dispatch_batch(self, batch: list[_GroupedGenerationRequest]) -> None:
        remaining = list(batch)
        grouped_requests: list[tuple[int, list[_GroupedGenerationRequest]]] = []
        by_prompt: dict[int, list[_GroupedGenerationRequest]] = {}
        for request in batch:
            if request.step_index != 0:
                continue
            by_prompt.setdefault(request.prompt_index, []).append(request)
        for prompt_index, requests in by_prompt.items():
            if len(requests) != self.group_size:
                continue
            prompt_texts = {request.prompt_text for request in requests}
            if len(prompt_texts) != 1:
                continue
            grouped_requests.append((prompt_index, sorted(requests, key=lambda item: item.candidate_index)))
            remaining = [request for request in remaining if request.prompt_index != prompt_index]

        try:
            if grouped_requests:
                grouped_prompts = [requests[0].prompt_text for _, requests in grouped_requests]
                grouped_samples = self.serving_backend.generate_grouped(grouped_prompts, self.group_size)
                for (_, requests), prompt_samples in zip(grouped_requests, grouped_samples, strict=True):
                    for request, sample in zip(requests, prompt_samples, strict=True):
                        agent_sample = AgentSample.from_backend_sample(sample)
                        request.runtime.record_trace_event(
                            "scheduler_batch",
                            payload={
                                "mode": "grouped_step0",
                                "batch_size": len(grouped_prompts),
                                "group_size": self.group_size,
                            },
                        )
                        request.future.set_result(agent_sample)

            if remaining:
                prompts = [request.prompt_text for request in remaining]
                samples = self.serving_backend.generate_batch(prompts)
                for request, sample in zip(remaining, samples, strict=True):
                    agent_sample = AgentSample.from_backend_sample(sample)
                    request.runtime.record_trace_event(
                        "scheduler_batch",
                        payload={
                            "mode": "dynamic_batch",
                            "batch_size": len(remaining),
                        },
                    )
                    request.future.set_result(agent_sample)
        except BaseException as exc:
            self._error = exc
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)


class Agent:
    """Public whitebox agent type.

    Constructed directly, ``Agent`` is the rollout callable passed to FlashRL.
    Inside ``run_fn(agent)``, the same public type is used as the live per-prompt
    runtime object.
    """

    def __init__(
        self,
        run_fn: RunFn,
        *,
        tools: ToolSource = None,
        context_manager: BaseContextManager | None = None,
        max_steps: int,
        runtime: SubprocessToolRuntime | None = None,
        max_parallel_calls: int = 4,
        tool_executor: AgentToolExecutor | None = None,
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
        self.tool_executor = tool_executor or AgentToolExecutor(
            runtime=self.runtime,
            max_parallel_calls=max_parallel_calls,
        )
        if isinstance(tools, ToolRegistry):
            pass
        elif tools is not None and not callable(tools):
            self._validate_unique_tools(list(tools))
        self._is_runtime = False
        self._serving_backend: Any | None = None
        self._state: AgentState | None = None
        self._generation_count = 0
        self._pending_sample: AgentSample | None = None
        self._last_available_tool_names: list[str] = []
        self._scheduler: _GroupedAgentScheduler | None = None

    @classmethod
    def _create_runtime(
        cls,
        blueprint: "Agent",
        *,
        prompt: Prompt,
        serving_backend: Any,
        prompt_index: int | None = None,
        candidate_index: int | None = None,
        scheduler: _GroupedAgentScheduler | None = None,
    ) -> "Agent":
        runtime = cls.__new__(cls)
        runtime.run_fn = blueprint.run_fn
        runtime.tools = blueprint.tools
        runtime.context_manager = blueprint.context_manager
        runtime.max_steps = blueprint.max_steps
        runtime.runtime = blueprint.runtime
        runtime.max_parallel_calls = blueprint.max_parallel_calls
        runtime.tool_executor = blueprint.tool_executor
        runtime._is_runtime = True
        runtime._serving_backend = serving_backend
        conversation = Conversation(messages=[Message(role="user", content=prompt.text)])
        metadata: dict[str, Any] = {}
        runtime._state = AgentState(
            prompt=prompt,
            conversation=conversation,
            metadata=metadata,
            session_context=SessionContext(
                conversation=conversation,
                metadata=metadata,
                prompt_index=prompt_index,
                candidate_index=candidate_index,
            ),
        )
        runtime._generation_count = 0
        runtime._pending_sample = None
        runtime._last_available_tool_names = []
        runtime._scheduler = scheduler
        return runtime

    @property
    def state(self) -> AgentState:
        """Return the advanced mutable state for the current live run."""
        self._ensure_runtime_context("state")
        assert self._state is not None
        return self._state

    @property
    def session(self) -> SessionContext:
        """Return the current structured session context."""
        return self.state.session_context

    @property
    def agent_trace(self) -> AgentTrace:
        """Return the current structured agent trace."""
        return self.state.agent_trace

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
        for prompt_index, prompt in enumerate(prompts):
            runtime = self._create_runtime(
                self,
                prompt=prompt,
                serving_backend=serving_backend,
                prompt_index=prompt_index,
                candidate_index=0,
            )
            self._run_runtime(runtime)
            rollouts.append(self._build_rollout_output(runtime.state))
        return rollouts

    def run_grouped(
        self,
        prompts: list[Prompt],
        serving_backend: Any,
        group_size: int,
    ) -> tuple[list[Prompt], list[RolloutOutput], list[int], list[int]]:
        """Run grouped whitebox rollouts with dynamic ready-trajectory batching."""
        self._ensure_blueprint_context("run_grouped")
        if group_size < 1:
            raise ValueError("group_size must be >= 1.")
        scheduler = _GroupedAgentScheduler(
            serving_backend=serving_backend,
            group_size=group_size,
            active_runtime_count=len(prompts) * group_size,
        )
        runtimes: list[Agent] = []
        threads: list[threading.Thread] = []
        for prompt_index, prompt in enumerate(prompts):
            for candidate_index in range(group_size):
                runtime = self._create_runtime(
                    self,
                    prompt=prompt,
                    serving_backend=serving_backend,
                    prompt_index=prompt_index,
                    candidate_index=candidate_index,
                    scheduler=scheduler,
                )
                runtimes.append(runtime)
                thread = threading.Thread(
                    target=self._run_runtime_thread,
                    args=(runtime, scheduler),
                    daemon=True,
                )
                threads.append(thread)
                thread.start()

        scheduler.run()
        for thread in threads:
            thread.join()

        flat_prompts: list[Prompt] = []
        flat_rollouts: list[RolloutOutput] = []
        prompt_indices: list[int] = []
        candidate_indices: list[int] = []
        for prompt_index, prompt in enumerate(prompts):
            matching = [
                runtime
                for runtime in runtimes
                if runtime.session.prompt_index == prompt_index
            ]
            matching.sort(key=lambda runtime: runtime.session.candidate_index or 0)
            for runtime in matching:
                flat_prompts.append(prompt)
                flat_rollouts.append(self._build_rollout_output(runtime.state))
                prompt_indices.append(prompt_index)
                candidate_indices.append(runtime.session.candidate_index or 0)
        return flat_prompts, flat_rollouts, prompt_indices, candidate_indices

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

        if self._scheduler is not None:
            sample = self._scheduler.submit(
                self,
                prompt_text=str(prompt),
                step_index=self._generation_count,
            )
        else:
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
        executor: AgentToolExecutor | None = None,
    ) -> list[tuple[ToolCall, Any]]:
        """Execute tools safely, record tool messages, and preserve declared order."""
        self._ensure_not_finished("run_tools")
        self._ensure_no_pending("run_tools")
        resolved_tools = list(tools) if tools is not None else self.available_tools()
        if tools is not None:
            self._last_available_tool_names = [tool.name for tool in resolved_tools]
        active_executor = executor or self.tool_executor
        return active_executor.execute(self, tool_calls, tools=resolved_tools)

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

    def record_trace_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one structured non-trainable trace event."""
        self._ensure_runtime_context("record_trace_event")
        self.state.agent_trace.events.append(
            AgentTraceEvent(
                event_type=str(event_type),
                step_index=len(self.state.assistant_turns),
                prompt_index=self.session.prompt_index,
                candidate_index=self.session.candidate_index,
                payload=dict(payload or {}),
            )
        )

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
        source = self.tools
        if source is None:
            return []
        if callable(source):
            source = source(state)
        if isinstance(source, ToolRegistry):
            resolved_profile = str(state.session_context.tool_profile or ToolProfile.DEFAULT.value)
            return source.resolve(state=state, profile=resolved_profile)
        resolved_tools = list(source)
        self._validate_unique_tools(resolved_tools)
        return resolved_tools

    def _validate_unique_tools(self, tools: list[Tool]) -> None:
        """Reject duplicate tool names within one visible tool set."""
        tool_names: set[str] = set()
        for tool in tools:
            if tool.name in tool_names:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            tool_names.add(tool.name)

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

        self.record_trace_event(
            "tool_visibility",
            payload={"tool_names": list(available_tool_names)},
        )

    def _run_runtime(self, runtime: "Agent") -> None:
        try:
            self.run_fn(runtime)
        except _AgentStop:
            pass

        if runtime._pending_sample is not None:
            raise RuntimeError(
                "Agent run returned with an unrecorded sample. "
                "Call agent.record_generation(sample, ...) before finishing or returning."
            )

        self._finalize_state(runtime.state)

    def _run_runtime_thread(
        self,
        runtime: "Agent",
        scheduler: _GroupedAgentScheduler,
    ) -> None:
        error: BaseException | None = None
        try:
            self._run_runtime(runtime)
        except BaseException as exc:
            error = exc
        finally:
            scheduler.mark_runtime_finished(error)

    def _finalize_state(self, state: AgentState) -> None:
        if state.stop_reason == "max_steps":
            state.final_text = str(state.last_assistant_text)
            state.done = True
        elif not state.done:
            state.final_text = str(state.last_assistant_text)
            state.stop_reason = "run_returned"
            state.done = True

    def _build_rollout_output(self, state: AgentState) -> RolloutOutput:
        """Convert one finished state into the rollout shape expected by FlashRL."""
        last_sample = state.last_sample
        final_prompt_token_ids = list(last_sample.prompt_token_ids) if last_sample is not None else []
        final_response_token_ids = list(last_sample.response_token_ids) if last_sample is not None else []
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
            "prompt_index": state.session_context.prompt_index,
            "candidate_index": state.session_context.candidate_index,
        }
        return RolloutOutput(
            text=str(state.final_text),
            log_prob=log_prob,
            prompt_token_ids=final_prompt_token_ids,
            response_token_ids=final_response_token_ids,
            response_token_logprobs=final_response_logprobs,
            assistant_turns=list(state.assistant_turns),
            conversation=state.conversation,
            agent_trace=state.agent_trace,
            metadata=rollout_metadata,
        )

