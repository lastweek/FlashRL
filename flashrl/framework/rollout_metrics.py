"""Utility functions for computing metrics from rollout outputs."""

from typing import Any


def count_llm_call_rounds(rollouts: list[Any]) -> int:
    """Count total LLM API call rounds across all rollouts.

    For single-turn conversations, this equals the number of rollouts.
    For multi-turn conversations, counts the number of assistant messages.

    Args:
        rollouts: List of RolloutOutput objects

    Returns:
        Total count of LLM call rounds
    """
    total = 0
    for rollout in rollouts:
        conversation = getattr(rollout, 'conversation', None)
        if conversation is not None:
            messages = getattr(conversation, 'messages', [])
            # Count assistant messages as LLM call rounds
            total += sum(1 for msg in messages if getattr(msg, 'role', None) == 'assistant')
        else:
            # Fallback: count as single LLM call per rollout
            total += 1
    return total


def count_tool_calls(rollouts: list[Any]) -> int:
    """Count total tool calls across all rollout conversations.

    Iterates through all messages in all conversations and sums up
    the tool_calls field for each message.

    Args:
        rollouts: List of RolloutOutput objects

    Returns:
        Total count of tool calls
    """
    total = 0
    for rollout in rollouts:
        conversation = getattr(rollout, 'conversation', None)
        if conversation is not None:
            messages = getattr(conversation, 'messages', [])
            for msg in messages:
                tool_calls = getattr(msg, 'tool_calls', [])
                total += len(tool_calls) if tool_calls else 0
    return total
