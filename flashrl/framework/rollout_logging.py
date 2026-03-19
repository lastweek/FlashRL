"""Helpers for serializing rollout artifacts."""

from __future__ import annotations

import math
from typing import Any, Callable


ROLLOUT_SCHEMA_VERSION = 3

PROMOTED_PROMPT_METADATA_KEYS = (
    "task_id",
    "source",
    "split",
    "language",
    "rating",
    "verifier",
)
PROMOTED_REWARD_METADATA_KEYS = (
    "pass_rate",
    "passed_tests",
    "total_tests",
    "accuracy_pass",
    "format_pass",
    "truncated",
    "execution_seconds",
    "failure_reason",
    "checker_used",
    "execution_status",
    "code_preview",
)
PROMOTED_OUTPUT_METADATA_KEYS = (
    "finish_reason",
    "stop_reason",
    "ttft_seconds",
    "tpot_seconds",
    "generation_seconds",
    "response_token_count",
)


def clone_json_mapping(value: Any) -> dict[str, Any]:
    """Return a shallow dict copy for JSON-like mappings."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def compact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields while preserving zeros and booleans."""
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            item = compact_mapping(item)
        elif isinstance(item, list):
            item = [compact_mapping(entry) if isinstance(entry, dict) else entry for entry in item]

        if item is None:
            continue
        if item == {}:
            continue
        if item == []:
            continue
        compact[key] = item
    return compact


def clone_json_messages(messages: Any) -> list[dict[str, Any]]:
    """Return a list of shallow-copied JSON-like messages."""
    if not isinstance(messages, list):
        return []
    return [
        compact_mapping(dict(message))
        for message in messages
        if isinstance(message, dict)
    ]


def promote_metadata_fields(
    metadata: dict[str, Any],
    promoted_keys: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a metadata mapping into promoted and leftover fields."""
    promoted: dict[str, Any] = {}
    leftovers: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in promoted_keys:
            promoted[key] = value
            continue
        leftovers[key] = value
    return promoted, leftovers


def factor_shared_messages(
    message_groups: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Extract the longest shared message prefix across candidate transcripts."""
    if not message_groups:
        return [], []

    prefix_length = min(len(messages) for messages in message_groups)
    shared_length = 0
    while shared_length < prefix_length:
        candidate = message_groups[0][shared_length]
        if any(messages[shared_length] != candidate for messages in message_groups[1:]):
            break
        shared_length += 1

    if shared_length == 0:
        return [], [clone_json_messages(messages) for messages in message_groups]

    return (
        clone_json_messages(message_groups[0][:shared_length]),
        [clone_json_messages(messages[shared_length:]) for messages in message_groups],
    )


def derive_log_prob_stats(
    *,
    log_prob: float,
    response_token_count: int,
    prompt_token_count: int,
) -> dict[str, float | None]:
    """Compute normalized rollout confidence and length ratios."""
    avg_log_prob_per_token: float | None = None
    avg_token_prob: float | None = None
    if response_token_count > 0:
        avg_log_prob_per_token = float(log_prob / response_token_count)
        avg_token_prob = float(math.exp(avg_log_prob_per_token))

    output_to_prompt_token_ratio: float | None = None
    if prompt_token_count > 0:
        output_to_prompt_token_ratio = float(response_token_count / prompt_token_count)

    return {
        "avg_log_prob_per_token": avg_log_prob_per_token,
        "avg_token_prob": avg_token_prob,
        "output_to_prompt_token_ratio": output_to_prompt_token_ratio,
    }


def candidate_is_solved(reward: dict[str, Any]) -> bool:
    """Infer whether one candidate solved the task from promoted reward fields."""
    accuracy_pass = reward.get("accuracy_pass")
    if accuracy_pass is not None:
        return bool(accuracy_pass)

    passed_tests = reward.get("passed_tests")
    total_tests = reward.get("total_tests")
    if isinstance(passed_tests, int) and isinstance(total_tests, int) and total_tests > 0:
        return passed_tests == total_tests

    pass_rate = reward.get("pass_rate")
    if isinstance(pass_rate, (int, float)):
        return float(pass_rate) >= 1.0

    return False


def derive_pass_rate(reward: dict[str, Any]) -> float | None:
    """Derive a normalized pass rate from the promoted reward fields."""
    pass_rate = reward.get("pass_rate")
    if isinstance(pass_rate, (int, float)):
        return float(pass_rate)

    passed_tests = reward.get("passed_tests")
    total_tests = reward.get("total_tests")
    if isinstance(passed_tests, int) and isinstance(total_tests, int) and total_tests > 0:
        return float(passed_tests / total_tests)

    accuracy_pass = reward.get("accuracy_pass")
    if accuracy_pass is not None:
        return 1.0 if bool(accuracy_pass) else 0.0

    return None


def derive_rollout_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute prompt-group summary stats from serialized candidates."""
    if not candidates:
        return {
            "reward_mean": 0.0,
            "reward_max": 0.0,
            "reward_min": 0.0,
            "reward_span": 0.0,
            "best_candidate_index": None,
            "fastest_candidate_index": None,
            "longest_candidate_index": None,
            "format_pass_count": 0,
            "accuracy_pass_count": 0,
            "truncated_count": 0,
            "solved_count": 0,
            "avg_pass_rate": None,
            "avg_generation_seconds": None,
            "avg_log_prob_per_token": None,
        }

    reward_values = [float(candidate["reward"]["value"]) for candidate in candidates]
    generation_pairs = [
        (float(candidate["output"]["generation_seconds"]), int(candidate["candidate_index"]))
        for candidate in candidates
        if isinstance(candidate["output"].get("generation_seconds"), (int, float))
    ]
    longest_pairs = [
        (int(candidate["output"]["response_token_count"]), int(candidate["candidate_index"]))
        for candidate in candidates
    ]
    avg_pass_rates = [
        float(candidate["reward"]["pass_rate"])
        for candidate in candidates
        if isinstance(candidate["reward"].get("pass_rate"), (int, float))
    ]
    avg_generation_seconds = [
        float(candidate["output"]["generation_seconds"])
        for candidate in candidates
        if isinstance(candidate["output"].get("generation_seconds"), (int, float))
    ]
    avg_log_probs = [
        float(candidate["output"]["avg_log_prob_per_token"])
        for candidate in candidates
        if isinstance(candidate["output"].get("avg_log_prob_per_token"), (int, float))
    ]

    best_candidate = max(
        (
            (float(candidate["reward"]["value"]), -int(candidate["candidate_index"]))
            for candidate in candidates
        ),
        default=(0.0, 0),
    )
    fastest_candidate = min(generation_pairs, default=(0.0, None))
    longest_candidate = max(longest_pairs, default=(0, None))

    return {
        "reward_mean": float(sum(reward_values) / len(reward_values)),
        "reward_max": max(reward_values),
        "reward_min": min(reward_values),
        "reward_span": max(reward_values) - min(reward_values),
        "best_candidate_index": int(-best_candidate[1]),
        "fastest_candidate_index": fastest_candidate[1],
        "longest_candidate_index": longest_candidate[1],
        "format_pass_count": sum(int(bool(candidate["reward"].get("format_pass"))) for candidate in candidates),
        "accuracy_pass_count": sum(
            int(bool(candidate["reward"].get("accuracy_pass"))) for candidate in candidates
        ),
        "truncated_count": sum(int(bool(candidate["reward"].get("truncated"))) for candidate in candidates),
        "solved_count": sum(int(candidate_is_solved(candidate["reward"])) for candidate in candidates),
        "avg_pass_rate": (
            float(sum(avg_pass_rates) / len(avg_pass_rates)) if avg_pass_rates else None
        ),
        "avg_generation_seconds": (
            float(sum(avg_generation_seconds) / len(avg_generation_seconds))
            if avg_generation_seconds
            else None
        ),
        "avg_log_prob_per_token": (
            float(sum(avg_log_probs) / len(avg_log_probs)) if avg_log_probs else None
        ),
    }


def build_rollout_record(
    *,
    run_id: str,
    run_index: int,
    step: int,
    epoch: int,
    batch_index: int,
    batches_in_epoch: int,
    prompt_index: int,
    prompt_count: int,
    group_size: int,
    batch_candidate_count: int,
    prompt: Any,
    candidates: list[dict[str, Any]],
    serialize_for_json: Callable[[Any], Any],
    truncate_text: Callable[[str], str],
) -> dict[str, Any]:
    """Serialize one prompt-group rollout record in schema v3."""
    prompt_text = str(getattr(prompt, "text", ""))
    prompt_metadata = clone_json_mapping(serialize_for_json(getattr(prompt, "metadata", {})))

    message_groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        rollout = candidate["rollout"]
        conversation = clone_json_mapping(
            serialize_for_json(getattr(rollout, "conversation", {}))
        )
        messages = clone_json_messages(conversation.get("messages"))
        if not messages:
            synthetic_messages = []
            if prompt_text:
                synthetic_messages.append(
                    {
                        "role": "user",
                        "content": prompt_text,
                        "tool_calls": [],
                        "metadata": {},
                    }
                )
            synthetic_messages.append(
                {
                    "role": "assistant",
                    "content": str(getattr(rollout, "text", "")),
                    "tool_calls": [],
                    "metadata": {},
                }
            )
            messages = synthetic_messages
        message_groups.append(messages)

    shared_messages, completion_message_groups = factor_shared_messages(message_groups)

    serialized_candidates: list[dict[str, Any]] = []
    merged_prompt_metadata = dict(prompt_metadata)
    serving_weight_version: dict[str, Any] = {}
    for candidate, completion_messages in zip(
        candidates,
        completion_message_groups,
        strict=True,
    ):
        rollout = candidate["rollout"]
        reward = candidate["reward"]
        response_text = str(getattr(rollout, "text", ""))
        prompt_token_count = len(getattr(rollout, "prompt_token_ids", []))
        response_token_count = len(getattr(rollout, "response_token_ids", []))

        rollout_metadata = clone_json_mapping(
            serialize_for_json(getattr(rollout, "metadata", {}))
        )
        weight_version = clone_json_mapping(rollout_metadata.pop("weight_version", {}))
        if weight_version and not serving_weight_version:
            serving_weight_version = dict(weight_version)
        raw_prompt_metadata = clone_json_mapping(rollout_metadata.pop("prompt_metadata", {}))
        merged_prompt_metadata.update(
            {
                key: value
                for key, value in raw_prompt_metadata.items()
                if key not in merged_prompt_metadata
            }
        )

        promoted_output, remaining_output_metadata = promote_metadata_fields(
            rollout_metadata,
            PROMOTED_OUTPUT_METADATA_KEYS,
        )
        output_stats = derive_log_prob_stats(
            log_prob=float(getattr(rollout, "log_prob", 0.0)),
            response_token_count=response_token_count,
            prompt_token_count=prompt_token_count,
        )
        generation_seconds = promoted_output.get("generation_seconds")
        tokens_per_second = 0.0
        if isinstance(generation_seconds, (int, float)) and float(generation_seconds) > 0.0:
            tokens_per_second = float(response_token_count / float(generation_seconds))

        reward_metadata = clone_json_mapping(
            serialize_for_json(getattr(reward, "metadata", {}))
        )
        promoted_reward, remaining_reward_metadata = promote_metadata_fields(
            reward_metadata,
            PROMOTED_REWARD_METADATA_KEYS,
        )

        derived_pass_rate = derive_pass_rate(promoted_reward)
        serialized_candidates.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "completion_messages": completion_messages,
                "output": compact_mapping(
                    {
                        "preview": truncate_text(response_text),
                        "response_token_count": response_token_count,
                        "response_char_count": len(response_text),
                        "finish_reason": promoted_output.get("finish_reason"),
                    "stop_reason": promoted_output.get("stop_reason"),
                    "ttft_seconds": promoted_output.get("ttft_seconds"),
                    "tpot_seconds": promoted_output.get("tpot_seconds"),
                        "generation_seconds": generation_seconds,
                        "tokens_per_second": tokens_per_second,
                        "log_prob": float(getattr(rollout, "log_prob", 0.0)),
                        "weight_version": weight_version,
                        "metadata": remaining_output_metadata,
                        **output_stats,
                    }
                ),
                "reward": compact_mapping(
                    {
                        "value": float(getattr(reward, "reward", 0.0)),
                        "pass_rate": derived_pass_rate,
                        "passed_tests": promoted_reward.get("passed_tests"),
                        "total_tests": promoted_reward.get("total_tests"),
                    "accuracy_pass": promoted_reward.get("accuracy_pass"),
                    "format_pass": promoted_reward.get("format_pass"),
                    "truncated": promoted_reward.get("truncated"),
                    "execution_seconds": promoted_reward.get("execution_seconds"),
                        "failure_reason": promoted_reward.get("failure_reason"),
                        "checker_used": promoted_reward.get("checker_used"),
                        "execution_status": promoted_reward.get("execution_status"),
                        "code_preview": promoted_reward.get("code_preview"),
                        "metadata": remaining_reward_metadata,
                    }
                ),
                }
            )

    promoted_prompt_metadata, remaining_prompt_metadata = promote_metadata_fields(
        merged_prompt_metadata,
        PROMOTED_PROMPT_METADATA_KEYS,
    )
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_index": run_index,
        "step": step,
        "epoch": epoch,
        "batch_index": batch_index,
        "batches_in_epoch": batches_in_epoch,
        "prompt_index": prompt_index,
        "prompt_count": prompt_count,
        "group_size": group_size,
        "candidate_count": len(serialized_candidates),
        "batch_candidate_count": batch_candidate_count,
        "input": compact_mapping({
            "shared_messages": shared_messages,
            "prompt_preview": truncate_text(prompt_text),
            "prompt_token_count": (
                len(getattr(candidates[0]["rollout"], "prompt_token_ids", [])) if candidates else 0
            ),
            "prompt_char_count": len(prompt_text),
            "metadata": remaining_prompt_metadata,
            **promoted_prompt_metadata,
        }),
        "serving": compact_mapping({
            "weight_version": serving_weight_version,
        }),
        "summary": compact_mapping(derive_rollout_summary(serialized_candidates)),
        "candidates": serialized_candidates,
    }
