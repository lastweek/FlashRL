"""Small whitebox agent-tools demo with parallel subprocess-backed calls."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from flashrl.framework import Prompt, ReActRollout, Tool


def add_tool(arguments: dict[str, int], prompt: Prompt) -> str:
    """Return the sum of two integers."""
    del prompt
    return str(int(arguments["a"]) + int(arguments["b"]))


def multiply_tool(arguments: dict[str, int], prompt: Prompt) -> str:
    """Return the product of two integers."""
    del prompt
    return str(int(arguments["a"]) * int(arguments["b"]))


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


def main() -> int:
    """Run one demo rollout and print the resulting transcript."""
    entrypoint_module = "flashrl.framework.examples.agent-tools.run"
    rollout = ReActRollout(
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
        system_prompt=(
            "Use tools when needed. After any tool use, finish with a concise answer "
            "after `Final:`."
        ),
    )
    prompt = Prompt(text="Compute 20 + 22 and 6 * 7.")
    rollout_output = rollout([prompt], DemoServingBackend())[0]
    print(
        json.dumps(rollout_output.model_dump(), ensure_ascii=True, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
