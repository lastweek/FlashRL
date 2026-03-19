"""Unit tests for whitebox ReAct rollouts and subprocess-backed tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from flashrl.framework import LoggingConfig, ReActRollout, SubprocessToolRuntime, Tool
from flashrl.framework.config import GrpoConfig, TrainerConfig
from flashrl.framework.data_models import (
    AssistantTurn,
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    ToolCall,
    TrainingBatch,
)
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.run_logger import RunLogger
from flashrl.framework.trainer.grpo.trainer import GRPOTrainer
from tests.conftest import TinyServingBackend, TinyTrainingBackend, reward_fn

pytestmark = pytest.mark.unit


def echo_tool(arguments: dict[str, object], prompt: Prompt) -> str:
    """Return one echoed argument for subprocess runtime tests."""
    del prompt
    return str(arguments["text"])


def slow_tool(arguments: dict[str, object], prompt: Prompt) -> str:
    """Sleep briefly and return one tagged result."""
    del prompt
    time.sleep(float(arguments.get("delay", 0.05)))
    return str(arguments["text"])


def failing_tool(arguments: dict[str, object], prompt: Prompt) -> str:
    """Raise a deterministic tool failure."""
    del arguments, prompt
    raise RuntimeError("boom")


class ScriptedServingBackend:
    """Scripted serving backend for deterministic ReAct tests."""

    def __init__(self, responses_by_step: list[list[str]]) -> None:
        self.responses_by_step = [list(batch) for batch in responses_by_step]
        self.rendered_prompts: list[list[str]] = []
        self.generation_defaults: dict[str, object] = {}

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        call_index = len(self.rendered_prompts)
        self.rendered_prompts.append(list(prompts))
        batch = self.responses_by_step[call_index]
        assert len(batch) == len(prompts)
        outputs = []
        for prompt_text, response_text in zip(prompts, batch, strict=True):
            prompt_token_ids = [((ord(char) % 30) + 1) for char in prompt_text[:16]] or [1]
            response_token_ids = [((ord(char) % 30) + 1) for char in response_text[:16]] or [1]
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


def load_script_module(module_name: str, relative_path: str):
    """Load one script module by path for example tests."""
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_trainer() -> GRPOTrainer:
    """Build one minimal trainer for learner-batch expansion tests."""
    training_backend = TinyTrainingBackend(learning_rate=1e-2, group_size=2)
    serving_backend = TinyServingBackend()
    reward = UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace())
    return GRPOTrainer(
        config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        grpo_config=GrpoConfig(group_size=2, clip_ratio=0.2, kl_coefficient=0.0),
        actor_backend=training_backend,
        reference_backend=None,
        serving_backend=serving_backend,
        reward_fn=reward,
        rollout_generator=SimpleNamespace(),
        run_logger=None,
        metrics_sink=None,
    )


def test_subprocess_tool_runtime_executes_valid_tool() -> None:
    """The subprocess runtime should import and execute one tool entrypoint."""
    runtime = SubprocessToolRuntime(default_timeout_seconds=3.0, default_memory_limit_mb=64)
    result = runtime.execute(
        Tool(
            name="echo",
            description="Echo text.",
            entrypoint="tests.test_agent_tools:echo_tool",
        ),
        arguments={"text": "hello"},
        prompt=Prompt(text="prompt"),
    )

    assert result.error is False
    assert result.content == "hello"
    assert result.metadata["status"] == "ok"


def test_subprocess_tool_runtime_converts_exceptions_into_tool_results() -> None:
    """Tool failures should be surfaced as ToolResult(error=True)."""
    runtime = SubprocessToolRuntime(default_timeout_seconds=3.0, default_memory_limit_mb=64)
    result = runtime.execute(
        Tool(
            name="failing",
            description="Always fail.",
            entrypoint="tests.test_agent_tools:failing_tool",
        ),
        arguments={},
        prompt=Prompt(text="prompt"),
    )

    assert result.error is True
    assert "Tool execution failed" in result.content
    assert result.metadata["status"] in {"tool_error", "exception"}


def test_react_rollout_runs_parallel_tools_and_preserves_declared_order() -> None:
    """Parallel tool calls should execute concurrently but append results deterministically."""
    rollout = ReActRollout(
        tools=[
            Tool(
                name="slow",
                description="Sleep and return text.",
                entrypoint="tests.test_agent_tools:slow_tool",
            ),
            Tool(
                name="fast",
                description="Sleep and return text.",
                entrypoint="tests.test_agent_tools:slow_tool",
            ),
        ],
        max_steps=3,
        system_prompt="Use tools before answering.",
    )
    backend = ScriptedServingBackend(
        responses_by_step=[
            [
                'Action: [{"tool": "slow", "arguments": {"text": "first", "delay": 0.15}}, '
                '{"tool": "fast", "arguments": {"text": "second", "delay": 0.01}}]'
            ],
            ["Final: <answer>done</answer>"],
        ]
    )

    rollout_output = rollout([Prompt(text="demo")], backend)[0]
    tool_messages = [
        message for message in rollout_output.conversation.messages if message.role == "tool"
    ]

    assert rollout_output.text == "<answer>done</answer>"
    assert [message.content for message in tool_messages] == ["first", "second"]
    assert len(rollout_output.assistant_turns) == 2


def test_react_rollout_inserts_system_prompt_and_supports_callable_system_prompts() -> None:
    """Rendered prompts should include the resolved user system prompt."""
    rollout = ReActRollout(
        tools=[],
        max_steps=1,
        system_prompt=lambda prompt: f"system::{prompt.metadata['tag']}",
    )
    backend = ScriptedServingBackend(
        responses_by_step=[["Final: answer-a", "Final: answer-b"]]
    )

    rollout(
        [
            Prompt(text="question a", metadata={"tag": "a"}),
            Prompt(text="question b", metadata={"tag": "b"}),
        ],
        backend,
    )

    assert "System Prompt:\nsystem::a" in backend.rendered_prompts[0][0]
    assert "System Prompt:\nsystem::b" in backend.rendered_prompts[0][1]


def test_react_rollout_handles_parse_errors_unknown_tools_and_max_steps() -> None:
    """Malformed actions, unknown tools, and max-steps termination should stay in-band."""
    malformed = ReActRollout(
        tools=[],
        max_steps=2,
        system_prompt="system",
    )
    malformed_backend = ScriptedServingBackend(
        responses_by_step=[["Action: not-json"], ["Final: done"]]
    )
    malformed_output = malformed([Prompt(text="demo")], malformed_backend)[0]
    malformed_tool_messages = [
        message for message in malformed_output.conversation.messages if message.role == "tool"
    ]
    assert malformed_tool_messages[0].metadata["status"] == "parse_error"

    unknown = ReActRollout(
        tools=[],
        max_steps=2,
        system_prompt="system",
    )
    unknown_backend = ScriptedServingBackend(
        responses_by_step=[
            ['Action: {"tool": "missing", "arguments": {}}'],
            ["Final: done"],
        ]
    )
    unknown_output = unknown([Prompt(text="demo")], unknown_backend)[0]
    unknown_tool_messages = [
        message for message in unknown_output.conversation.messages if message.role == "tool"
    ]
    assert unknown_tool_messages[0].metadata["status"] == "unknown_tool"

    max_steps_rollout = ReActRollout(
        tools=[
            Tool(
                name="echo",
                description="Echo text.",
                entrypoint="tests.test_agent_tools:echo_tool",
            )
        ],
        max_steps=1,
        system_prompt="system",
    )
    max_steps_backend = ScriptedServingBackend(
        responses_by_step=[['Action: {"tool": "echo", "arguments": {"text": "hi"}}']]
    )
    max_steps_output = max_steps_rollout([Prompt(text="demo")], max_steps_backend)[0]
    assert max_steps_output.metadata["stop_reason"] == "max_steps"
    assert max_steps_output.text.startswith("Action:")


def test_learner_batch_expands_whitebox_assistant_turns() -> None:
    """Learner-batch construction should flatten assistant turns and repeat advantages."""
    trainer = build_trainer()
    rollout = RolloutOutput(
        text="<answer>42</answer>",
        log_prob=-0.2,
        prompt_token_ids=[1, 2],
        response_token_ids=[3, 4],
        response_token_logprobs=[-0.1, -0.1],
        assistant_turns=[
            AssistantTurn(
                prompt_token_ids=[1, 2],
                response_token_ids=[3],
                response_token_logprobs=[-0.1],
            ),
            AssistantTurn(
                prompt_token_ids=[1, 2, 3],
                response_token_ids=[4, 5],
                response_token_logprobs=[-0.2, -0.2],
            ),
        ],
        conversation=Conversation(
            messages=[
                Message(role="user", content="prompt"),
                Message(role="assistant", content='Action: {"tool": "echo", "arguments": {"text": "x"}}'),
                Message(role="tool", content="x"),
                Message(role="assistant", content="<answer>42</answer>"),
            ]
        ),
    )
    batch = TrainingBatch(
        prompts=[Prompt(text="prompt")],
        conversations=[rollout.conversation],
        rollouts=[rollout],
        rewards=[RewardOutput(reward=1.0)],
        group_size=2,
        prompt_count=1,
        prompt_indices=[0],
        candidate_indices=[1],
    )

    learner_batch = trainer._build_learner_batch(batch, torch.tensor([0.75]))

    assert learner_batch.prompt_token_ids == [[1, 2], [1, 2, 3]]
    assert learner_batch.response_token_ids == [[3], [4, 5]]
    assert learner_batch.advantages == pytest.approx([0.75, 0.75])
    assert learner_batch.prompt_indices == [0, 0]
    assert learner_batch.candidate_indices == [1, 1]


def test_rollout_logging_preserves_whitebox_output_metadata(tmp_path: Path) -> None:
    """Whitebox rollout metadata should survive rollouts.jsonl serialization."""
    logger = RunLogger(LoggingConfig(log_dir=tmp_path, console=False), model_name="fake/model")
    rollout = RolloutOutput(
        text="<answer>42</answer>",
        log_prob=-0.2,
        prompt_token_ids=[1, 2],
        response_token_ids=[3, 4],
        response_token_logprobs=[-0.1, -0.1],
        assistant_turns=[
            AssistantTurn(
                prompt_token_ids=[1, 2],
                response_token_ids=[3],
                response_token_logprobs=[-0.1],
            )
        ],
        conversation=Conversation(
            messages=[
                Message(role="system", content="system"),
                Message(role="user", content="question"),
                Message(
                    role="assistant",
                    content='Action: {"tool": "echo", "arguments": {"text": "x"}}',
                    tool_calls=[ToolCall(name="echo", arguments={"text": "x"}, tool_id="tool-1")],
                ),
                Message(
                    role="tool",
                    content="x",
                    metadata={"tool_name": "echo", "tool_id": "tool-1", "status": "ok"},
                ),
                Message(role="assistant", content="<answer>42</answer>"),
            ]
        ),
        metadata={
            "prompt_metadata": {"task_id": "task-1", "source": "tests", "split": "train"},
            "stop_reason": "final",
            "assistant_turn_count": 2,
        },
    )
    logger.log_rollout_batch(
        step=1,
        epoch=1,
        batch_index=1,
        batches_in_epoch=1,
        prompts=[Prompt(text="question", metadata={"task_id": "task-1", "source": "tests", "split": "train"})],
        rollouts=[rollout],
        rewards=[RewardOutput(reward=1.0)],
        prompt_indices=[0],
        candidate_indices=[0],
        group_size=1,
        prompt_count=1,
    )

    record = json.loads((logger.run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert record["candidates"][0]["output"]["stop_reason"] == "final"
    assert record["candidates"][0]["output"]["metadata"]["assistant_turn_count"] == 2
    all_messages = list(record["input"]["shared_messages"]) + list(
        record["candidates"][0]["completion_messages"]
    )
    assert any(message.get("role") == "tool" for message in all_messages)


def test_math_example_supports_blackbox_and_whitebox_rollout_builders() -> None:
    """The math example should expose explicit blackbox and whitebox rollout construction."""
    module = load_script_module(
        "flashrl_reasoning_math_train_whitebox",
        "flashrl/framework/examples/math/train.py",
    )

    blackbox = module.build_math_rollout(
        rollout_mode="blackbox",
        training_mode="math",
        system_prompt=module.build_math_system_prompt("math"),
    )
    whitebox = module.build_math_rollout(
        rollout_mode="whitebox",
        training_mode="math",
        system_prompt=module.build_math_system_prompt("math"),
    )

    assert callable(blackbox)
    assert isinstance(whitebox, ReActRollout)
    assert whitebox.tools[0].name == "calculator"
    assert module.calculator_tool({"expression": "20 + 22"}, Prompt(text="prompt")) == "42"


def test_agent_tools_example_runs_and_prints_rollout_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The small agent-tools example should run offline and print one rollout payload."""
    module = load_script_module(
        "flashrl_agent_tools_demo",
        "flashrl/framework/examples/agent-tools/run.py",
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "The sum is 42 and the product is 42."
    tool_messages = [message for message in payload["conversation"]["messages"] if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages] == ["42", "42"]
