"""Unit tests for the public agent building blocks and example flows."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from flashrl.framework import LoggingConfig
from flashrl.framework.agent import (
    Agent,
    BaseContextManager,
    SubprocessToolRuntime,
    Tool,
    WindowedContextManager,
)
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
from flashrl.framework.tools import SubprocessToolRuntime as ShimToolRuntime
from flashrl.framework.tools import Tool as ShimTool
from flashrl.framework.trainer.grpo.trainer import GRPOTrainer
from tests.conftest import TinyServingBackend, TinyTrainingBackend, reward_fn


pytestmark = pytest.mark.unit


class ScriptedServingBackend:
    """Scripted serving backend for deterministic agent-runtime tests."""

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


class AssistantCueServingBackend:
    """Probe backend that only returns text when the prompt ends at the assistant turn."""

    def __init__(self) -> None:
        self.rendered_prompts: list[list[str]] = []
        self.generation_defaults: dict[str, object] = {}

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        self.rendered_prompts.append(list(prompts))
        outputs = []
        for prompt_text in prompts:
            response_text = "ready answer" if prompt_text.endswith("Assistant:") else ""
            response_token_ids = [((ord(char) % 30) + 1) for char in response_text[:16]] or [1]
            outputs.append(
                SimpleNamespace(
                    text=response_text,
                    prompt_token_ids=[((ord(char) % 30) + 1) for char in prompt_text[:16]] or [1],
                    response_token_ids=response_token_ids,
                    response_token_logprobs=[-0.1 for _ in response_token_ids],
                    log_prob=float(-0.1 * len(response_token_ids)),
                    metadata={"finish_reason": "stop"},
                )
            )
        return outputs


class SpyContextManager(BaseContextManager):
    """Record observe calls while exposing the full conversation."""

    def __init__(self) -> None:
        self.observed_roles: list[list[str]] = []

    def build_messages(self, state) -> list[Message]:
        return [message.model_copy(deep=True) for message in state.conversation.messages]

    def observe(self, state, new_messages: list[Message]) -> None:
        del state
        self.observed_roles.append([message.role for message in new_messages])


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


def build_runtime(
    *,
    backend: ScriptedServingBackend | None = None,
    tools=None,
    context_manager: BaseContextManager | None = None,
    max_steps: int = 3,
) -> Agent:
    """Build one runtime directly for focused unit tests."""
    agent = Agent(
        lambda live_agent: None,
        tools=tools,
        context_manager=context_manager,
        max_steps=max_steps,
    )
    return Agent._create_runtime(
        agent,
        prompt=Prompt(text="demo"),
        serving_backend=backend or ScriptedServingBackend([["draft"]]),
    )


def test_subprocess_tool_runtime_executes_valid_tool() -> None:
    """The subprocess runtime should import and execute one tool entrypoint."""
    runtime = SubprocessToolRuntime(default_timeout_seconds=3.0, default_memory_limit_mb=64)
    result = runtime.execute(
        Tool(
            name="echo",
            description="Echo text.",
            entrypoint="tests.tool_runtime_helpers:echo_tool",
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
            entrypoint="tests.tool_runtime_helpers:failing_tool",
        ),
        arguments={},
        prompt=Prompt(text="prompt"),
    )

    assert result.error is True
    assert "Tool execution failed" in result.content
    assert result.metadata["status"] in {"tool_error", "exception"}


def test_framework_tools_is_a_compatibility_shim() -> None:
    """The temporary tools shim should point at the relocated agent tool types."""
    assert ShimTool is Tool
    assert ShimToolRuntime is SubprocessToolRuntime


def test_agent_call_delegates_to_run_batch() -> None:
    """The agent rollout adapter should delegate ``__call__`` to the explicit named method."""
    agent = Agent(lambda live_agent: None, max_steps=1)
    expected = [SimpleNamespace(value="done")]
    called: dict[str, object] = {}

    def fake_run_batch(prompts, serving_backend):
        called["prompts"] = prompts
        called["serving_backend"] = serving_backend
        return expected

    agent.run_batch = fake_run_batch  # type: ignore[method-assign]

    prompts = [Prompt(text="demo")]
    backend = object()
    assert agent(prompts, backend) == expected
    assert called == {"prompts": prompts, "serving_backend": backend}


def test_agent_build_prompt_renders_tools_footer_and_transcript() -> None:
    """The prompt helper should render tools, transcript, footer, then the assistant cue."""
    runtime = build_runtime()
    runtime.add_message("system", "system prompt")
    prompt_text = runtime.build_prompt(
        tools=[
            Tool(
                name="echo",
                description="Echo text.",
                entrypoint="tests.tool_runtime_helpers:echo_tool",
            )
        ],
        footer="Reply with Final: ...",
    )

    assert prompt_text.startswith("Available tools:\n- echo: Echo text.")
    assert "Available tools:\n- echo: Echo text." in prompt_text
    assert "Transcript:\nSystem: system prompt\nUser: demo" in prompt_text
    assert "\n\nReply with Final: ...\n\nAssistant:" in prompt_text
    assert prompt_text.endswith("Assistant:")
    assert "Instruction:" not in prompt_text


def test_agent_build_prompt_renders_no_tools_and_custom_messages_override() -> None:
    """The prompt helper should support empty tool sets and custom message views."""
    runtime = build_runtime()
    override_messages = [Message(role="assistant", content="override only")]
    prompt_text = runtime.build_prompt(
        tools=[],
        messages=override_messages,
    )

    assert "Available tools:\n- no tools available on this step" in prompt_text
    assert "Transcript:\nAssistant: override only" in prompt_text
    assert "User: demo" not in prompt_text
    assert prompt_text.endswith("Assistant:")
    assert "Instruction:" not in prompt_text


def test_agent_build_prompt_uses_system_message_for_contract_text() -> None:
    """Policy text should live in system messages, not in a separate prompt section."""
    runtime = build_runtime()
    runtime.add_message("system", "Follow the policy.")

    prompt_text = runtime.build_prompt(
        tools=[],
        footer="Answer tersely.",
    )

    assert "Instruction:" not in prompt_text
    assert "Transcript:\nSystem: Follow the policy.\nUser: demo" in prompt_text
    assert "\n\nAnswer tersely.\n\nAssistant:" in prompt_text
    assert prompt_text.endswith("Assistant:")


def test_agent_build_prompt_is_completion_ready_for_generation() -> None:
    """Whitebox prompt rendering should end at the assistant turn for completion backends."""
    backend = AssistantCueServingBackend()

    def run(agent: Agent) -> None:
        agent.add_message("system", "Follow the contract.")
        sample = agent.generate(agent.build_prompt(tools=[]))
        agent.record_generation(sample)
        agent.finish()

    output = Agent(run, max_steps=1)([Prompt(text="demo")], backend)[0]

    assert len(backend.rendered_prompts) == 1
    assert len(backend.rendered_prompts[0]) == 1
    assert backend.rendered_prompts[0][0].endswith("Assistant:")
    assert output.text == "ready answer"


def test_agent_exposes_common_state_directly() -> None:
    """Common runtime state should be available without going through ``state``."""
    runtime = build_runtime()

    assert runtime.done is False
    assert runtime.prompt.text == "demo"
    assert runtime.conversation.messages[0].role == "user"
    assert runtime.metadata == {}


def test_agent_generate_record_and_finish_records_one_turn() -> None:
    """Explicit runtime loops should record assistant turns and finalize cleanly."""
    backend = ScriptedServingBackend(responses_by_step=[["draft answer"]])

    def run(agent: Agent) -> None:
        agent.add_message("system", "system prompt")
        sample = agent.generate("System: system prompt\nUser: demo")
        agent.record_generation(sample)
        agent.finish()

    output = Agent(run, max_steps=2)([Prompt(text="demo")], backend)[0]

    assert output.text == "draft answer"
    assert output.metadata["stop_reason"] == "final"
    assert len(output.assistant_turns) == 1
    assert [message.role for message in output.conversation.messages] == ["system", "user", "assistant"]


def test_agent_run_tools_requires_explicit_record_first() -> None:
    """Tool execution should fail clearly when the latest sample is still unrecorded."""
    runtime = build_runtime(backend=ScriptedServingBackend([["working"]]))
    available_tools = [
        Tool(
            name="echo",
            description="Echo text.",
            entrypoint="tests.tool_runtime_helpers:echo_tool",
        )
    ]
    sample = runtime.generate("prompt")

    with pytest.raises(RuntimeError, match="record_generation\\(sample, .*\\) first"):
        runtime.run_tools(
            [ToolCall(name="echo", arguments={"text": "hi"}, tool_id="tool-1")],
            tools=available_tools,
        )

    with pytest.raises(RuntimeError, match="record_generation\\(sample, .*\\) first"):
        runtime.finish()

    with pytest.raises(RuntimeError, match="record_generation\\(sample, .*\\) first"):
        runtime.add_message("tool", "nope")

    runtime.record_generation(sample)


def test_agent_returning_with_unrecorded_sample_raises() -> None:
    """Returning from the loop with a pending sample should raise instead of auto-flushing."""
    backend = ScriptedServingBackend(responses_by_step=[["draft"]])

    def run(agent: Agent) -> None:
        agent.generate("one step")

    with pytest.raises(RuntimeError, match="unrecorded sample"):
        Agent(run, max_steps=2)([Prompt(text="demo")], backend)


def test_agent_run_tools_records_order_and_visible_tools() -> None:
    """Tool execution should preserve order and keep visible-tool metadata."""
    agent = Agent(
        lambda live_agent: None,
        tools=[
            Tool(
                name="slow",
                description="Sleep and return text.",
                entrypoint="tests.tool_runtime_helpers:slow_tool",
            ),
            Tool(
                name="fast",
                description="Sleep and return text.",
                entrypoint="tests.tool_runtime_helpers:slow_tool",
            ),
        ],
        max_steps=3,
    )
    backend = ScriptedServingBackend(responses_by_step=[["working"], ["done"]])

    def run(live_agent: Agent) -> None:
        live_agent.add_message("system", "Use tools.")
        available_tools = live_agent.available_tools()
        sample = live_agent.generate(live_agent.build_prompt(tools=available_tools))
        calls = [
            ToolCall(name="slow", arguments={"text": "first", "delay": 0.15}, tool_id="tool-1"),
            ToolCall(name="fast", arguments={"text": "second", "delay": 0.01}, tool_id="tool-2"),
        ]
        live_agent.record_generation(sample, tool_calls=calls)
        live_agent.run_tools(calls, tools=available_tools)
        final_sample = live_agent.generate("final step")
        live_agent.record_generation(final_sample)
        live_agent.finish("done")

    agent.run_fn = run
    output = agent([Prompt(text="demo")], backend)[0]
    tool_messages = [message for message in output.conversation.messages if message.role == "tool"]

    assert output.text == "done"
    assert len(output.assistant_turns) == 2
    assert [message.content for message in tool_messages] == ["first", "second"]
    assert output.conversation.messages[2].tool_calls[0].name == "slow"
    assert output.conversation.messages[3].metadata["available_tool_names"] == ["slow", "fast"]


def test_agent_supports_dynamic_tools_and_records_visible_tools() -> None:
    """Dynamic tool resolution should be visible in the rollout trace."""
    def dynamic_tools(state) -> list[Tool]:
        used_tools = any(message.role == "tool" for message in state.conversation.messages)
        if used_tools:
            return []
        return [
            Tool(
                name="echo",
                description="Echo text.",
                entrypoint="tests.tool_runtime_helpers:echo_tool",
            )
        ]

    backend = ScriptedServingBackend(responses_by_step=[["first"], ["second"]])

    def run(agent: Agent) -> None:
        available_tools = agent.available_tools()
        first_sample = agent.generate("first step")
        first_calls = [ToolCall(name="echo", arguments={"text": "first"}, tool_id="tool-1")]
        agent.record_generation(first_sample, tool_calls=first_calls)
        agent.run_tools(first_calls, tools=available_tools)

        available_tools = agent.available_tools()
        second_sample = agent.generate("second step")
        second_calls = [ToolCall(name="echo", arguments={"text": "second"}, tool_id="tool-2")]
        agent.record_generation(second_sample, tool_calls=second_calls)
        agent.run_tools(second_calls, tools=available_tools)
        agent.finish("done")

    output = Agent(run, tools=dynamic_tools, max_steps=3)([Prompt(text="demo")], backend)[0]
    tool_messages = [message for message in output.conversation.messages if message.role == "tool"]

    assert [message.content for message in tool_messages] == ["first", "Tool not available on this step: echo"]
    assert tool_messages[1].metadata["status"] == "tool_unavailable"
    assert output.metadata["visible_tools_by_step"] == [["echo"], []]
    assert output.conversation.metadata["visible_tools_by_step"] == [["echo"], []]


def test_agent_run_returned_finalizes_with_last_assistant_text() -> None:
    """Returning from the loop without finish should still produce a valid rollout."""
    backend = ScriptedServingBackend(responses_by_step=[["draft"]])

    def run(agent: Agent) -> None:
        sample = agent.generate("one step")
        agent.record_generation(sample)

    output = Agent(run, max_steps=2)([Prompt(text="demo")], backend)[0]

    assert output.text == "draft"
    assert output.metadata["stop_reason"] == "run_returned"


def test_agent_max_steps_stops_further_generation_cleanly() -> None:
    """Further generation should stop cleanly once max_steps is reached."""
    backend = ScriptedServingBackend(responses_by_step=[["first"]])

    def run(agent: Agent) -> None:
        sample = agent.generate("first step")
        agent.record_generation(sample)
        agent.generate("second step")

    output = Agent(run, max_steps=1)([Prompt(text="demo")], backend)[0]

    assert output.text == "first"
    assert output.metadata["stop_reason"] == "max_steps"
    assert len(output.assistant_turns) == 1


def test_context_manager_observe_hook_and_windowed_context() -> None:
    """Context managers should observe traced messages and build deterministic windows."""
    spy = SpyContextManager()
    backend = ScriptedServingBackend(responses_by_step=[["draft"]])

    def run(agent: Agent) -> None:
        agent.add_message("system", "system")
        sample = agent.generate("prompt")
        agent.record_generation(sample)
        agent.add_message("tool", "tool result", metadata={"tool_name": "echo"})
        agent.finish()

    output = Agent(run, context_manager=spy, max_steps=2)([Prompt(text="demo")], backend)[0]
    window = WindowedContextManager(max_messages=2).build_messages(
        SimpleNamespace(conversation=output.conversation)
    )

    assert spy.observed_roles == [["system"], ["assistant"], ["tool"]]
    assert [message.role for message in window] == ["system", "assistant", "tool"]


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
    assert isinstance(whitebox, Agent)
    assert not callable(whitebox.tools)
    assert whitebox.tools[0].name == "calculator"
    assert module.calculator_tool({"expression": "20 + 22"}, Prompt(text="prompt")) == "42"


def test_math_whitebox_rollout_runs_end_to_end_offline() -> None:
    """The math whitebox rollout should execute one tool step then one final step."""
    module = load_script_module(
        "flashrl_reasoning_math_train_whitebox_smoke",
        "flashrl/framework/examples/math/train.py",
    )
    backend = ScriptedServingBackend(
        responses_by_step=[
            ['Action: {"tool": "calculator", "arguments": {"expression": "20 + 22"}}'],
            ["Final: <think>Use the calculator.</think><answer>42</answer>"],
        ]
    )
    prompt = Prompt(text="Solve 20 + 22", metadata={"final_answer": "42"})
    rollout = module.build_math_whitebox_agent(training_mode="reasoning")([prompt], backend)[0]
    reward = module.math_reward_fn(rollout, training_mode="reasoning")

    assert rollout.text == "<think>Use the calculator.</think><answer>42</answer>"
    assert rollout.metadata["assistant_turn_count"] == 2
    assert reward.reward == pytest.approx(1.1)


def test_agent_tools_example_runs_and_prints_rollout_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The minimal custom-loop example should run offline and print one rollout payload."""
    module = load_script_module(
        "flashrl_agent_tools_demo",
        "flashrl/framework/examples/agent-tools/run.py",
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "The sum is 42 and the product is 42."
    tool_messages = [message for message in payload["conversation"]["messages"] if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages] == ["42", "42"]


def test_agent_react_example_runs_and_prints_rollout_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The explicit ReAct recipe example should run offline and print one rollout payload."""
    module = load_script_module(
        "flashrl_agent_react_demo",
        "flashrl/framework/examples/agent-react/run.py",
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "8 + 13 is 21 and 7 * 6 is 42."


def test_agent_dynamic_tools_example_runs_and_prints_rollout_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dynamic-tools example should run offline and print one rollout payload."""
    module = load_script_module(
        "flashrl_agent_dynamic_tools_demo",
        "flashrl/framework/examples/agent-dynamic-tools/run.py",
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["visible_tools_by_step"] == [["lookup_note"], []]
    assert payload["text"].startswith("Alpha is the stronger default choice")
