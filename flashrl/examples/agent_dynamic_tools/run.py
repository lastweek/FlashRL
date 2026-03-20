"""Dynamic tool gating demo built from the core agent primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from flashrl.framework.agent import Agent, Tool, WindowedContextManager
from flashrl.framework.data_models import Prompt, ToolCall


@dataclass(frozen=True)
class _Decision:
    kind: Literal["action", "final"]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    parse_error: str | None = None


class DemoServingBackend:
    """Scripted backend that uses tools once, then finishes without tools."""

    def __init__(self) -> None:
        self.generation_defaults: dict[str, object] = {}
        self._call_index = 0

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        responses_by_step = [
            'Action: [{"tool": "lookup_note", "arguments": {"note_id": "alpha"}}, '
            '{"tool": "lookup_note", "arguments": {"note_id": "beta"}}]',
            "Final: Alpha is the stronger default choice because it leads Beta on reliability.",
        ]
        response_text = responses_by_step[min(self._call_index, len(responses_by_step) - 1)]
        self._call_index += 1
        outputs = []
        for prompt_text in prompts:
            prompt_token_ids = [((ord(char) % 30) + 1) for char in prompt_text[:32]] or [1]
            response_token_ids = [((ord(char) % 30) + 1) for char in response_text[:32]] or [1]
            response_token_logprobs = [-0.1 for _ in response_token_ids]
            outputs.append(
                SimpleNamespace(
                    text=response_text,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    log_prob=float(sum(response_token_logprobs)),
                    metadata={"finish_reason": "stop"},
                )
            )
        return outputs


def _parse_response(raw_text: str) -> _Decision:
    stripped = raw_text.strip()
    if stripped.startswith("Action:"):
        payload_text = stripped[len("Action:"):].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return _Decision(kind="action", parse_error=f"Invalid Action payload: {exc}")
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return _Decision(
                kind="action",
                parse_error="Action payload must be a JSON object or array.",
            )
        tool_calls: list[ToolCall] = []
        for entry in payload:
            if not isinstance(entry, dict):
                return _Decision(
                    kind="action",
                    parse_error="Each Action entry must be a JSON object.",
                )
            tool_name = entry.get("tool")
            arguments = entry.get("arguments", {})
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                return _Decision(
                    kind="action",
                    parse_error="Each Action entry must include string `tool` and object `arguments`.",
                )
            tool_calls.append(
                ToolCall(name=tool_name, arguments=dict(arguments), tool_id=uuid4().hex)
            )
        return _Decision(kind="action", tool_calls=tool_calls)
    if stripped.startswith("Final:"):
        return _Decision(kind="final", final_text=stripped[len("Final:"):].strip())
    return _Decision(kind="final", final_text=raw_text)


def _dynamic_tools(state) -> list[Tool]:
    if any(message.role == "tool" for message in state.conversation.messages):
        return []
    entrypoint_module = "flashrl.examples.agent_tools_helpers"
    return [
        Tool(
            name="lookup_note",
            description="Read one short comparison note by id.",
            entrypoint=f"{entrypoint_module}:lookup_note_tool",
        )
    ]


def build_demo_agent() -> Agent:
    """Build the dynamic-tools example agent."""

    def run(agent: Agent) -> None:
        agent.add_message(
            "system",
            "Inspect the available notes before giving a concise recommendation.\n"
            "Use tools when you need more information.\n"
            "Reply with `Action:` when calling tools and `Final:` when you are ready to answer.",
        )
        while not agent.done:
            available_tools = agent.available_tools()
            prompt = agent.build_prompt(tools=available_tools)
            sample = agent.generate(prompt)
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
                agent.run_tools(decision.tool_calls, tools=available_tools)
                continue
            agent.record_generation(sample)
            agent.finish(decision.final_text)

    return Agent(
        run_fn=run,
        tools=_dynamic_tools,
        context_manager=WindowedContextManager(max_messages=4),
        max_steps=3,
    )


def main() -> int:
    """Run the dynamic-tools example and print the rollout trace."""
    prompt = Prompt(text="Compare Alpha and Beta, then recommend the safer default.")
    rollout_output = build_demo_agent()([prompt], DemoServingBackend())[0]
    print(json.dumps(rollout_output.model_dump(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
