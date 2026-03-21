"""Evaluation helpers for the reference agent harness."""

from __future__ import annotations

from typing import Any, Sequence

from flashrl.examples.agent_harness.dataset import build_eval_dataset, reward_fn
from flashrl.framework import FlashRL
from flashrl.framework.agent import Agent
from flashrl.framework.data_models import RewardOutput, RolloutOutput


def evaluate_model(
    flashrl: FlashRL,
    rollout_agent: Agent,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    dataset = build_eval_dataset(limit=limit)
    rollouts = rollout_agent.run_batch(dataset, flashrl.serving_backend)
    rewards = [reward_fn(rollout) for rollout in rollouts]
    return {
        "dataset_size": len(dataset),
        **summarize_rollouts(rollouts, rewards),
    }


def summarize_rollouts(
    rollouts: Sequence[RolloutOutput],
    rewards: Sequence[RewardOutput],
) -> dict[str, Any]:
    count = len(rollouts)
    accuracy = (
        float(sum(1.0 for reward in rewards if reward.metadata.get("accuracy_pass")) / count)
        if count
        else 0.0
    )
    token_counts = [
        int(rollout.metadata.get("response_token_count", len(rollout.response_token_ids)))
        for rollout in rollouts
    ]
    rollout_seconds = [
        float(rollout.metadata.get("generation_seconds", 0.0))
        for rollout in rollouts
    ]
    tool_calls = [
        sum(len(message.tool_calls) for message in rollout.conversation.messages if message.role == "assistant")
        for rollout in rollouts
    ]
    tool_errors = [
        sum(1 for message in rollout.conversation.messages if message.role == "tool" and message.metadata.get("error"))
        for rollout in rollouts
    ]
    skill_loads = [
        sum(1 for event in rollout.agent_trace.events if event.event_type == "skill_load")
        for rollout in rollouts
    ]
    compactions = [
        sum(1 for event in rollout.agent_trace.events if event.event_type == "compaction")
        for rollout in rollouts
    ]
    subagents = [len(rollout.agent_trace.subagents) for rollout in rollouts]
    assistant_turns = [len(rollout.assistant_turns) for rollout in rollouts]
    return {
        "eval_accuracy": accuracy,
        "mean_total_model_tokens": _mean(token_counts),
        "mean_rollout_seconds": _mean(rollout_seconds),
        "mean_assistant_turns": _mean(assistant_turns),
        "mean_tool_calls": _mean(tool_calls),
        "mean_tool_error_count": _mean(tool_errors),
        "mean_skill_loads": _mean(skill_loads),
        "mean_compactions": _mean(compactions),
        "mean_subagent_calls": _mean(subagents),
    }


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))
