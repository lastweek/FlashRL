"""Focused tests for the generic harness extensions and agent harness examples."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from flashrl.framework.agent import (
    Agent,
    CompactionManager,
    CompactionPolicy,
    Tool,
    ToolProfile,
    ToolRegistry,
)
from flashrl.framework.data_models import Prompt, ToolCall


pytestmark = pytest.mark.unit


def load_script_module(
    module_name: str,
    relative_path: str,
    *,
    aliases: tuple[str, ...] = (),
):
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    for alias in aliases:
        sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


class GroupedProbeBackend:
    """Backend that records grouped and dynamic batches."""

    def __init__(self) -> None:
        self.grouped_calls: list[tuple[list[str], int]] = []
        self.batch_calls: list[list[str]] = []
        self.generation_defaults: dict[str, object] = {}

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        del kwargs
        self.grouped_calls.append((list(prompts), int(group_size)))
        outputs = []
        for prompt_text in prompts:
            grouped = []
            for candidate_index in range(group_size):
                text = f"grouped::{candidate_index}::{prompt_text[-12:]}"
                grouped.append(
                    SimpleNamespace(
                        text=text,
                        prompt_token_ids=[1, 2],
                        response_token_ids=[3, 4],
                        response_token_logprobs=[-0.1, -0.1],
                        log_prob=-0.2,
                        metadata={"finish_reason": "stop"},
                    )
                )
            outputs.append(grouped)
        return outputs

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        self.batch_calls.append(list(prompts))
        outputs = []
        for prompt_text in prompts:
            if "Tool[" in prompt_text:
                text = "Final: done"
            else:
                text = 'Action: {"tool": "echo", "arguments": {"text": "x"}}'
            outputs.append(
                SimpleNamespace(
                    text=text,
                    prompt_token_ids=[1, 2],
                    response_token_ids=[3, 4],
                    response_token_logprobs=[-0.1, -0.1],
                    log_prob=-0.2,
                    metadata={"finish_reason": "stop"},
                )
            )
        return outputs


def test_tool_registry_resolves_profiles() -> None:
    registry = ToolRegistry.from_tools(
        [
            Tool(name="read", description="Read", entrypoint="tests.tool_runtime_helpers:echo_tool"),
        ],
        profiles=[ToolProfile.READONLY.value],
    )
    registry.register(
        Tool(name="write", description="Write", entrypoint="tests.tool_runtime_helpers:echo_tool"),
        profiles=[ToolProfile.DEFAULT.value],
    )
    runtime = Agent._create_runtime(
        Agent(lambda agent: None, tools=registry, max_steps=1),
        prompt=Prompt(text="demo"),
        serving_backend=GroupedProbeBackend(),
    )

    runtime.session.tool_profile = ToolProfile.READONLY.value
    assert [tool.name for tool in runtime.available_tools()] == ["read"]

    runtime.session.tool_profile = ToolProfile.DEFAULT.value
    assert [tool.name for tool in runtime.available_tools()] == ["write"]


def test_compaction_manager_records_trace_event() -> None:
    manager = CompactionManager(
        policy=CompactionPolicy(trigger_message_count=4, preserve_recent_messages=2, max_summary_chars=256)
    )
    runtime = Agent._create_runtime(
        Agent(lambda agent: None, context_manager=manager, max_steps=1),
        prompt=Prompt(text="demo"),
        serving_backend=GroupedProbeBackend(),
    )
    runtime.add_message("system", "system")
    runtime.add_message("assistant", "first")
    runtime.add_message("tool", "tool one", metadata={"tool_name": "echo"})
    runtime.add_message("assistant", "second")
    runtime.add_message("tool", "tool two", metadata={"tool_name": "echo"})

    assert runtime.session.rolling_summary
    assert any(event.event_type == "compaction" for event in runtime.agent_trace.events)


def test_agent_run_grouped_uses_grouped_step0_and_preserves_indices() -> None:
    backend = GroupedProbeBackend()

    def run(agent: Agent) -> None:
        available_tools = agent.available_tools()
        sample = agent.generate(agent.build_prompt(tools=available_tools))
        if sample.text.startswith("grouped::"):
            calls = [ToolCall(name="echo", arguments={"text": "x"}, tool_id="tool-1")]
            agent.record_generation(sample, tool_calls=calls)
            agent.run_tools(calls, tools=available_tools)
            followup = agent.generate(agent.build_prompt(tools=[]))
            agent.record_generation(followup)
            agent.finish("done")
            return
        agent.record_generation(sample)
        agent.finish("done")

    agent = Agent(
        run_fn=run,
        tools=[
            Tool(name="echo", description="Echo text.", entrypoint="tests.tool_runtime_helpers:echo_tool"),
        ],
        max_steps=3,
    )
    prompts = [Prompt(text="alpha"), Prompt(text="beta")]
    flat_prompts, rollouts, prompt_indices, candidate_indices = agent.run_grouped(
        prompts,
        backend,
        group_size=2,
    )

    assert len(backend.grouped_calls) == 1
    assert backend.grouped_calls[0][1] == 2
    assert prompt_indices == [0, 0, 1, 1]
    assert candidate_indices == [0, 1, 0, 1]
    assert len(flat_prompts) == 4
    assert all(rollout.metadata["candidate_index"] in {0, 1} for rollout in rollouts)
    assert all(any(event.event_type == "scheduler_batch" for event in rollout.agent_trace.events) for rollout in rollouts)


def test_agent_harness_example_runs_offline_with_special_tools() -> None:
    module = importlib.import_module("flashrl.examples.agent_harness.harness")

    class ScriptedBackend:
        def __init__(self) -> None:
            self.responses = [
                'Action: [{"tool": "load_skill", "arguments": {"name": "repo_triage"}}, '
                '{"tool": "list_repo_files", "arguments": {}}]',
                "Final: v3",
            ]
            self.index = 0

        def generate_batch(self, prompts: list[str], **kwargs):
            del kwargs
            response_text = self.responses[min(self.index, len(self.responses) - 1)]
            self.index += 1
            return [
                SimpleNamespace(
                    text=response_text,
                    prompt_token_ids=[1, 2],
                    response_token_ids=[3, 4],
                    response_token_logprobs=[-0.1, -0.1],
                    log_prob=-0.2,
                    metadata={"finish_reason": "stop"},
                )
                for _ in prompts
            ]

    agent = module.build_coding_agent(module.CodingHarnessConfig())
    prompt = module.build_coding_train_dataset(limit=1)[0]
    rollout = agent.run_batch([prompt], ScriptedBackend())[0]

    assert rollout.text == "v3"
    assert any(event.event_type == "skill_load" for event in rollout.agent_trace.events)
    assert any(event.event_type == "tool_batch" for event in rollout.agent_trace.events)
    assert any(message.role == "tool" for message in rollout.conversation.messages)


def test_agent_harness_ablation_matrix_and_summary() -> None:
    train_module = importlib.import_module("flashrl.examples.agent_harness_ablation.train")
    eval_module = importlib.import_module("flashrl.examples.agent_harness_ablation.eval")
    variants = train_module.expand_matrix(
        {
            "base_harness": {
                "enable_skills": True,
                "enable_compaction": True,
                "enable_subagents": True,
            },
            "matrix": [
                {"name": "tools_only", "harness": {"enable_skills": False, "enable_compaction": False, "enable_subagents": False}},
                {"name": "full_harness", "harness": {}},
            ],
        }
    )

    assert [name for name, _ in variants] == ["tools_only", "full_harness"]
    assert variants[0][1].enable_skills is False
    summary = eval_module.summarize_manifest(
        {
            "study_name": "demo",
            "runs": [
                {
                    "variant": "tools_only",
                    "seed": 1,
                    "evaluation": {
                        "eval_accuracy": 0.5,
                        "mean_total_model_tokens": 12.0,
                        "mean_rollout_seconds": 1.2,
                    },
                },
                {
                    "variant": "full_harness",
                    "seed": 1,
                    "evaluation": {
                        "eval_accuracy": 1.0,
                        "mean_total_model_tokens": 20.0,
                        "mean_rollout_seconds": 1.8,
                    },
                },
            ],
        }
    )

    assert summary["variant_count"] == 2
    assert summary["variants"][0]["variant"] == "full_harness"
    assert isinstance(summary["variants"][0]["pareto_optimal"], bool)
