"""Minimal custom-loop demo built from explicit agent primitives."""

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

from flashrl.framework.agent import Agent, Tool
from flashrl.framework.data_models import Prompt, ToolCall


@dataclass(frozen=True)
class _Decision:
    kind: Literal["action", "final"]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    parse_error: str | None = None


class DemoServingBackend:
    """Scripted backend that demonstrates one parallel tool step then one final step."""

    def __init__(self) -> None:
        self.generation_defaults: dict[str, object] = {}
        self._call_index = 0

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        responses_by_step = [
            'Action: [{"tool": "add", "arguments": {"a": 20, "b": 22}}, '
            '{"tool": "multiply", "arguments": {"a": 6, "b": 7}}]',
            "Final: The sum is 42 and the product is 42.",
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


def build_demo_agent() -> Agent:
    """Build one explicit custom agent loop using the core Agent methods."""
    entrypoint_module = "flashrl.examples.agent_tools_helpers"

    def run(agent: Agent) -> None:
        agent.add_message(
            "system",
            "You are a tool-using assistant.\n"
            "Use `Action:` with JSON tool calls when you need tools.\n"
            "Use `Final:` when you are ready to answer.\n"
            "After any tool use, finish with a concise answer after `Final:`.",
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
        tools=[
            Tool(
                name="add",
                description="Add two integers.",
                entrypoint=f"{entrypoint_module}:add_tool",
            ),
            Tool(
                name="multiply",
                description="Multiply two integers.",
                entrypoint=f"{entrypoint_module}:multiply_tool",
            ),
        ],
        max_steps=3,
    )


def main() -> int:
    """Run one demo rollout and print the resulting transcript."""
    prompt = Prompt(text="Compute 20 + 22 and 6 * 7.")
    rollout_output = build_demo_agent()([prompt], DemoServingBackend())[0]
    print(
        json.dumps(rollout_output.model_dump(), ensure_ascii=True, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
