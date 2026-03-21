"""Strict R1-Zero math training example with explicit dataset selection."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Literal
from uuid import uuid4
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_CONFIG_PATH = EXAMPLE_DIR / "config.yaml"
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from flashrl.framework import (
    FlashRL,
)
from flashrl.framework.agent import (
    Agent,
    SubprocessToolRuntime,
    Tool,
)
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    ToolCall,
)
DEFAULT_MATH_DATASET = "gsm8k"
SUPPORTED_MATH_DATASETS = ("gsm8k", "aime25")
# Some environments expose GSM8K under `openai/gsm8k`, while others expose the
# equivalent `gsm8k` alias. Try both so the example stays one-command friendly.
DEFAULT_GSM8K_HF_LOAD_CANDIDATES = (
    ("openai/gsm8k", "main"),
    ("gsm8k", "main"),
)
DEFAULT_GSM8K_TRAIN_SPLIT = "train"
DEFAULT_GSM8K_EVAL_SPLIT = "test"
DEFAULT_AIME25_HF_DATASET = "math-ai/aime25"
DEFAULT_AIME25_TRAIN_SPLIT = "test"
DEFAULT_AIME25_EVAL_SPLIT = "test"
DEFAULT_REASONING_CHECKPOINT_PATH = "/tmp/flashrl_reasoning_checkpoint.pt"
DEFAULT_REASONING_EVAL_BATCH_SIZE = 8
DEFAULT_MATH_TRAINING_MODE = "math"
DEFAULT_MATH_ROLLOUT_MODE = "blackbox"
SUPPORTED_MATH_ROLLOUT_MODES = ("blackbox", "whitebox")
FINAL_ANSWER_PATTERN = re.compile(r"####\s*(.+)$", re.MULTILINE)
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
STRICT_RESPONSE_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _MathDecision:
    kind: Literal["action", "final"]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    parse_error: str | None = None


def build_math_system_prompt(training_mode: str = "math") -> str:
    """Return the explicit system prompt used by the whitebox math rollout."""
    if training_mode == "reasoning":
        return (
            "You are a careful math assistant. Use the calculator tool when arithmetic "
            "would help.\n"
            "When you need tools, respond with exactly one line that starts with `Action:` "
            "followed by either one JSON object or a JSON array of objects.\n"
            "Each object must have the keys `tool` and `arguments`.\n"
            "When you are ready to answer, respond with `Final:` followed by the final answer content.\n"
            "The content after `Final:` must contain exactly one <think>...</think> block "
            "followed immediately by one <answer>...</answer> block."
        )
    return (
        "You are a careful math assistant. Use the calculator tool when arithmetic "
        "would help.\n"
        "When you need tools, respond with exactly one line that starts with `Action:` "
        "followed by either one JSON object or a JSON array of objects.\n"
        "Each object must have the keys `tool` and `arguments`.\n"
        "When you are ready to answer, respond with `Final:` followed by the final answer content.\n"
        "The content after `Final:` should contain only the final answer."
    )


def _parse_math_whitebox_response(raw_text: str) -> _MathDecision:
    """Parse one whitebox math assistant response into actions or a final answer."""
    stripped = raw_text.strip()
    if stripped.startswith("Action:"):
        payload_text = stripped[len("Action:"):].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return _MathDecision(
                kind="action",
                parse_error=f"Invalid Action payload: {exc}",
            )
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return _MathDecision(
                kind="action",
                parse_error="Action payload must be a JSON object or array.",
            )
        tool_calls: list[ToolCall] = []
        for entry in payload:
            if not isinstance(entry, dict):
                return _MathDecision(
                    kind="action",
                    parse_error="Each Action entry must be a JSON object.",
                )
            tool_name = entry.get("tool")
            arguments = entry.get("arguments", {})
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                return _MathDecision(
                    kind="action",
                    parse_error="Each Action entry must include string `tool` and object `arguments`.",
                )
            tool_calls.append(
                ToolCall(name=tool_name, arguments=dict(arguments), tool_id=uuid4().hex)
            )
        return _MathDecision(kind="action", tool_calls=tool_calls)
    if stripped.startswith("Final:"):
        return _MathDecision(
            kind="final",
            final_text=stripped[len("Final:"):].strip(),
        )
    return _MathDecision(kind="final", final_text=raw_text)
# Prompt contract
def render_math_prompt(problem: str, training_mode: str = "reasoning") -> str:
    """Render one strict user prompt with no system role.
    Args:
        problem: The math problem statement
        training_mode: "math" for answer-only, "reasoning" for reasoning + answer
    Returns:
        Formatted prompt string
    """
    if training_mode == "reasoning":
        # Reasoning mode: Ask for thinking blocks AND answer
        return (
            "Solve the following math problem.\n"
            "Respond with exactly one <think>...</think> block followed immediately by "
            "exactly one <answer>...</answer> block.\n"
            "Do not output any text before <think> or after </answer>.\n\n"
            f"Problem: {problem.strip()}"
        )
    else:
        # Math mode: Just ask for the problem
        return (
            "Solve the following math problem.\n\n"
            f"Problem: {problem.strip()}"
        )


def _render_system_prefixed_prompt(prompt_text: str, system_prompt: str | None) -> str:
    """Render one plain-text prompt for the blackbox path."""
    if not system_prompt:
        return prompt_text
    return f"System: {system_prompt}\n\nUser: {prompt_text}"


def _resolve_math_training_mode(
    *,
    explicit_training_mode: str | None = None,
    prompt_metadata: dict[str, Any] | None = None,
    prompts: list[Prompt] | None = None,
) -> str:
    """Resolve math vs reasoning mode from one explicit override or prompt metadata."""
    if explicit_training_mode:
        return str(explicit_training_mode)
    if prompt_metadata is not None:
        prompt_mode = str(prompt_metadata.get("training_mode") or "").strip()
        if prompt_mode:
            return prompt_mode
    if prompts is not None:
        for prompt in prompts:
            prompt_mode = str(prompt.metadata.get("training_mode") or "").strip()
            if prompt_mode:
                return prompt_mode
    return DEFAULT_MATH_TRAINING_MODE


def _safe_decimal_expression(text: str) -> Decimal:
    """Evaluate one arithmetic expression with a small AST allowlist."""
    allowed_binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
        ast.Mod: lambda left, right: left % right,
    }
    allowed_unary = {
        ast.UAdd: lambda value: value,
        ast.USub: lambda value: -value,
    }

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
            left = evaluate(node.left)
            right = evaluate(node.right)
            return allowed_binary[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](evaluate(node.operand))
        raise ValueError("Only arithmetic expressions are supported.")

    parsed = ast.parse(text, mode="eval")
    return evaluate(parsed)


def calculator_tool(arguments: dict[str, Any], prompt: Prompt) -> str:
    """Safely evaluate one arithmetic expression for the whitebox math example."""
    del prompt
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ValueError("calculator_tool requires a non-empty `expression` argument.")
    try:
        value = _safe_decimal_expression(expression)
    except (ValueError, SyntaxError, ArithmeticError, InvalidOperation) as exc:
        raise ValueError(f"calculator_tool could not evaluate the expression: {exc}") from exc
    normalized = format(value.normalize(), "f") if value == value.to_integral() else format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".") or "0"
    return normalized


def build_math_whitebox_agent(
    *,
    training_mode: str,
    system_prompt: str | None = None,
) -> Agent:
    """Construct one explicit custom-loop whitebox rollout for the math example."""
    resolved_system_prompt = system_prompt or build_math_system_prompt(training_mode)

    def run(agent: Agent) -> None:
        agent.add_message("system", resolved_system_prompt)
        while not agent.done:
            available_tools = agent.available_tools()
            sample = agent.generate(agent.build_prompt(tools=available_tools))
            decision = _parse_math_whitebox_response(sample.text)
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
                name="calculator",
                description="Evaluate one arithmetic expression and return the numeric result.",
                entrypoint="flashrl.examples.math.train:calculator_tool",
                timeout_seconds=3.0,
                memory_limit_mb=64,
            )
        ],
        max_steps=4,
        runtime=SubprocessToolRuntime(default_timeout_seconds=3.0, default_memory_limit_mb=64),
        max_parallel_calls=4,
    )

# Dataset loading
def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    """Validate one positive integer value."""
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1 (got {parsed}).")
    return parsed
def _resolve_math_limit(*, split_kind: str, explicit_limit: int | None) -> int | None:
    """Validate explicit CLI limits for the current dataset split."""
    if explicit_limit is not None:
        return _coerce_positive_int(explicit_limit, field_name=f"{split_kind}_limit")
    return None
def _resolve_math_dataset(dataset: str) -> str:
    """Validate one built-in math dataset choice."""
    if dataset not in SUPPORTED_MATH_DATASETS:
        choices = ", ".join(SUPPORTED_MATH_DATASETS)
        raise ValueError(f"dataset must be one of {{{choices}}} (got {dataset!r}).")
    return dataset
def _load_dataset_module():
    """Import `datasets.load_dataset` lazily so the example can fail clearly."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in live example usage
        raise RuntimeError(
            "flashrl.examples.math requires the `datasets` package to load math datasets. "
            "Install project dependencies or `pip install datasets`."
        ) from exc
    return load_dataset
def _print_dataset_summary(
    *,
    dataset_name: str,
    source: str,
    split: str,
    available: int,
    selected: int,
    problem_field: str,
    answer_field: str,
    target_format: str,
) -> None:
    """Print one compact dataset summary before the run starts."""
    print(
        f"dataset  name={dataset_name}  source={source}  split={split}  "
        f"available={available}  selected={selected}",
        flush=True,
    )
    print(
        f"format   problem_field={problem_field}  answer_field={answer_field}  "
        f"target={target_format}",
        flush=True,
    )
def _load_gsm8k_split(
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Load and normalize one GSM8K split."""
    load_dataset = _load_dataset_module()
    dataset_error: Exception | None = None
    raw_dataset = None
    loaded_source = None
    for dataset_name, config_name in DEFAULT_GSM8K_HF_LOAD_CANDIDATES:
        try:
            raw_dataset = load_dataset(dataset_name, config_name, split=split)
            loaded_source = dataset_name
            break
        except Exception as exc:  # pragma: no cover - depends on external dataset availability
            dataset_error = exc
    if raw_dataset is None:
        raise RuntimeError("Unable to load GSM8K from Hugging Face.") from dataset_error
    available_count = len(raw_dataset)
    if limit is not None and hasattr(raw_dataset, "select"):
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))
    rows: list[dict[str, str]] = []
    for index, row in enumerate(raw_dataset):
        task_id = str(row.get("id") or f"gsm8k-{split}-{index:06d}")
        problem = str(row["question"]).strip()
        final_answer = _extract_math_target_answer(str(row["answer"]))
        rows.append(
            {
                "task_id": task_id,
                "source": str(loaded_source),
                "split": split,
                "problem": problem,
                "final_answer": final_answer,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    _print_dataset_summary(
        dataset_name="gsm8k",
        source=str(loaded_source),
        split=split,
        available=available_count,
        selected=len(rows),
        problem_field="question",
        answer_field="answer",
        target_format="parse #### + numeric normalize",
    )
    return rows
def _load_aime25_split(
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Load and normalize the `math-ai/aime25` split."""
    load_dataset = _load_dataset_module()
    try:
        raw_dataset = load_dataset(DEFAULT_AIME25_HF_DATASET, split=split)
    except Exception as exc:  # pragma: no cover - depends on external dataset availability
        raise RuntimeError("Unable to load AIME25 from Hugging Face.") from exc
    available_count = len(raw_dataset)
    if limit is not None and hasattr(raw_dataset, "select"):
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))
    rows: list[dict[str, str]] = []
    for index, row in enumerate(raw_dataset):
        task_id = str(row.get("id") or f"aime25-{split}-{index:06d}")
        problem = str(row["problem"]).strip()
        final_answer = _normalize_math_answer(str(row["answer"]))
        if not final_answer:
            raise ValueError(f"AIME25 answer normalized to empty text: {task_id}")
        rows.append(
            {
                "task_id": task_id,
                "source": DEFAULT_AIME25_HF_DATASET,
                "split": split,
                "problem": problem,
                "final_answer": final_answer,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    _print_dataset_summary(
        dataset_name="aime25",
        source=DEFAULT_AIME25_HF_DATASET,
        split=split,
        available=available_count,
        selected=len(rows),
        problem_field="problem",
        answer_field="answer",
        target_format="direct numeric normalize",
    )
    return rows
def build_math_train_dataset(
    dataset: str = DEFAULT_MATH_DATASET,
    limit: int | None = None,
    training_mode: str = DEFAULT_MATH_TRAINING_MODE,
) -> list[Prompt]:
    """Build the math training dataset for both YAML hooks and the example CLI."""
    resolved_dataset = _resolve_math_dataset(dataset)
    resolved_limit = _resolve_math_limit(
        split_kind="train",
        explicit_limit=limit,
    )
    if resolved_dataset == "gsm8k":
        rows = _load_gsm8k_split(
            DEFAULT_GSM8K_TRAIN_SPLIT,
            limit=resolved_limit,
        )
    else:
        rows = _load_aime25_split(
            DEFAULT_AIME25_TRAIN_SPLIT,
            limit=resolved_limit,
        )
    return [
        Prompt(
            text=render_math_prompt(row["problem"], training_mode),  # NEW: pass training_mode
            metadata={
                "task_id": row["task_id"],
                "source": row["source"],
                "split": row["split"],
                "problem": row["problem"],
                "final_answer": row["final_answer"],
                "verifier": "numeric_exact",
                "training_mode": training_mode,
            },
        )
        for row in rows
    ]
def build_math_eval_dataset(
    dataset: str = DEFAULT_MATH_DATASET,
    limit: int | None = None,
    training_mode: str = DEFAULT_MATH_TRAINING_MODE,
) -> list[Prompt]:
    """Build the held-out math evaluation dataset."""
    resolved_dataset = _resolve_math_dataset(dataset)
    resolved_limit = _resolve_math_limit(
        split_kind="eval",
        explicit_limit=limit,
    )
    if resolved_dataset == "gsm8k":
        rows = _load_gsm8k_split(
            DEFAULT_GSM8K_EVAL_SPLIT,
            limit=resolved_limit,
        )
    else:
        rows = _load_aime25_split(
            DEFAULT_AIME25_EVAL_SPLIT,
            limit=resolved_limit,
        )
    return [
        Prompt(
            text=render_math_prompt(row["problem"], training_mode),  # NEW: pass training_mode
            metadata={
                "task_id": row["task_id"],
                "source": row["source"],
                "split": row["split"],
                "problem": row["problem"],
                "final_answer": row["final_answer"],
                "verifier": "numeric_exact",
                "training_mode": training_mode,
            },
        )
        for row in rows
    ]
# Answer parsing and normalization
# GSM8K mixes commas, currency markers, fractions, and decimal spellings in the
# final `####` target. Normalizing them here keeps the reward exact-match based
# without making the reward function itself hard to read.
def _normalize_math_answer(text: str) -> str:
    """Normalize one answer string for exact-match comparison."""
    value = str(text).strip()
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", "", value)
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.rstrip(".")
    if not value:
        return ""
    if re.fullmatch(r"-?\d+/\d+", value):
        fraction = Fraction(value)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    try:
        decimal_value = Decimal(value)
    except InvalidOperation:
        return value
    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized
def _extract_math_target_answer(raw_answer: str) -> str:
    """Extract and normalize the final GSM8K answer after the `####` marker."""
    match = FINAL_ANSWER_PATTERN.search(raw_answer)
    if match is None:
        raise ValueError(f"GSM8K answer is missing a final '####' marker: {raw_answer[:80]!r}")
    normalized = _normalize_math_answer(match.group(1))
    if not normalized:
        raise ValueError(f"GSM8K answer normalized to empty text: {raw_answer[:80]!r}")
    return normalized
def _extract_answer_block(response_text: str) -> str | None:
    """Extract the single parsed answer block when one exists."""
    matches = ANSWER_BLOCK_PATTERN.findall(response_text)
    if len(matches) == 1:
        answer_text = matches[0].strip()
        if answer_text:
            return answer_text
    return None


def _extract_last_number(response_text: str) -> str | None:
    """Extract the last numeric answer from free-form text."""
    number_pattern = re.compile(r'-?\d+(?:\.\d+)?|\d+/\d+')
    numbers = number_pattern.findall(response_text)
    if numbers:
        return numbers[-1]
    return None
# Reward logic
def _prompt_metadata_from_rollout(rollout: RolloutOutput) -> dict[str, Any]:
    """Recover the original prompt metadata attached by the rollout hook."""
    prompt_metadata = rollout.metadata.get("prompt_metadata")
    if isinstance(prompt_metadata, dict):
        return prompt_metadata
    return {}
def reasoning_reward_fn(rollout: RolloutOutput, debug_reward: bool = False) -> RewardOutput:
    """Compute reasoning reward from <thinking> tags in response.
    Evaluates reasoning quality based on:
    1. Presence of <thinking> tags
    2. Content quality within tags
    3. Logical structure of reasoning (numbered lists, bullets, length)
    Returns score in [0, 1] range.
    """
    text = rollout.text
    # Extract thinking content
    thinking_pattern = r'<think>(.*?)</think>'
    matches = re.findall(thinking_pattern, text, re.IGNORECASE | re.DOTALL)
    if not matches:
        # No thinking tags found
        if debug_reward:
            print(f"reward:reasoning  tags_found=False  score=0.0", flush=True)
        return RewardOutput(
            reward=0.0,
            metadata={"reasoning_score": 0.0, "thinking_tags_found": False}
        )
    thinking_content = matches[0].strip()
    if not thinking_content:
        # Empty thinking tags
        if debug_reward:
            print(f"reward:reasoning  tags_found=True  empty=True  score=0.0", flush=True)
        return RewardOutput(
            reward=0.0,
            metadata={"reasoning_score": 0.0, "thinking_tags_found": True, "empty": True}
        )
    # Score based on content quality
    # Look for structural indicators: numbered steps, bullet points, clear logic
    score = 0.5  # Base score for having tags
    # Bonus for structured reasoning (numbered lists, bullets)
    if re.search(r'\d+\.', thinking_content) or re.search(r'[-*]', thinking_content):
        score += 0.2
    # Bonus for length (more reasoning is generally better)
    if len(thinking_content) > 100:
        score += 0.2
    elif len(thinking_content) > 50:
        score += 0.1
    # Cap at 1.0
    score = min(score, 1.0)
    if debug_reward:
        has_structure = bool(re.search(r'\d+\.', thinking_content) or re.search(r'[-*]', thinking_content))
        print(
            f"reward:reasoning  tags_found=True  empty=False  "
            f"length={len(thinking_content)}  has_structure={has_structure}  "
            f"score={score:.3f}",
            flush=True
        )
    return RewardOutput(
        reward=score,
        metadata={
            "reasoning_score": score,
            "thinking_tags_found": True,
            "thinking_length": len(thinking_content)
        },
    )
def _compute_math_score(rollout: RolloutOutput, training_mode: str = "math", debug_reward: bool = False) -> RewardOutput:
    """Compute pure math score (existing logic).
    Reward matrix for math mode (no format restrictions):
    - `1.0`: correct answer
    - `0.0`: wrong answer
    Reward matrix for reasoning mode (with format):
    - `1.1`: strict correct (format + accuracy)
    - `1.0`: correct but malformed
    - `0.1`: strict wrong (format only)
    - `0.0`: invalid or both checks fail
    """
    text = rollout.text
    prompt_metadata = _prompt_metadata_from_rollout(rollout)
    expected_answer = _normalize_math_answer(str(prompt_metadata.get("final_answer", "")))
    finish_reason = rollout.metadata.get("finish_reason")
    truncated = finish_reason == "length"
    think_blocks = THINK_BLOCK_PATTERN.findall(text)
    answer_blocks = ANSWER_BLOCK_PATTERN.findall(text)
    strict_match = STRICT_RESPONSE_PATTERN.fullmatch(text)
    has_single_think_block = len(think_blocks) == 1
    has_single_answer_block = len(answer_blocks) == 1
    think_content = strict_match.group("think").strip() if strict_match is not None else ""
    answer_content = strict_match.group("answer").strip() if strict_match is not None else ""
    has_non_empty_strict_blocks = bool(think_content and answer_content)
    # In math mode: only check answer accuracy, no format restrictions
    if training_mode == "math":
        # Parse answer from various formats, not just <answer> blocks
        parsed_answer = _extract_answer_block(text) or _extract_last_number(text)
        answer_parse_pass = parsed_answer is not None
        normalized_answer = _normalize_math_answer(parsed_answer or "")
        accuracy_pass = bool(answer_parse_pass and normalized_answer == expected_answer)
        result = RewardOutput(
            reward=1.0 if accuracy_pass else 0.0,  # Only accuracy in math mode
            metadata={
                "expected_answer": expected_answer,
                "parsed_answer": parsed_answer,
                "normalized_answer": normalized_answer,
                "answer_parse_pass": answer_parse_pass,
                "accuracy_pass": accuracy_pass,
                "truncated": truncated,
                "finish_reason": finish_reason,
            },
        )
        if debug_reward:
            print(
                f"reward:math  mode=math  "
                f"expected={expected_answer!r}  "
                f"parsed={repr(parsed_answer) if parsed_answer else 'None'}  "
                f"normalized={repr(normalized_answer) if normalized_answer else 'None'}  "
                f"parse_pass={answer_parse_pass}  "
                f"accuracy_pass={accuracy_pass}  "
                f"truncated={truncated}  "
                f"finish_reason={finish_reason or 'None'}  "
                f"reward={result.reward:.1f}",
                flush=True
            )
        return result
    # Format stays strict on purpose:
    # - `...<answer>42</answer>` can earn the `+0.1` bonus
    # - `...<answer>42</answer> extra` cannot
    # - duplicate tags, missing close tags, or `finish_reason == "length"` cannot
    format_pass = bool(
        strict_match is not None
        and has_single_think_block
        and has_single_answer_block
        and has_non_empty_strict_blocks
        and not truncated
    )
    parsed_answer = _extract_answer_block(text)
    answer_parse_pass = parsed_answer is not None
    normalized_answer = _normalize_math_answer(parsed_answer or "")
    # Accuracy is intentionally independent from strict format:
    # - `<answer>$42.00.</answer>` still matches `42` after normalization
    # - `...<answer>42</answer> extra` still earns `1.0`
    accuracy_pass = bool(answer_parse_pass and normalized_answer == expected_answer)
    # Reward examples:
    # - strict correct -> `1.1`
    # - strict wrong -> `0.1`
    # - correct but malformed -> `1.0`
    # - duplicate or missing answer block -> `0.0`
    accuracy_reward = 1.0 if accuracy_pass else 0.0
    format_reward = 0.1 if format_pass else 0.0
    reward = accuracy_reward + format_reward
    result = RewardOutput(
        reward=reward,
        metadata={
            "expected_answer": expected_answer,
            "parsed_answer": parsed_answer,
            "normalized_answer": normalized_answer,
            "answer_parse_pass": answer_parse_pass,
            "accuracy_pass": accuracy_pass,
            "format_pass": format_pass,
            "truncated": truncated,
            "finish_reason": finish_reason,
            "think_block_count": len(think_blocks),
            "answer_block_count": len(answer_blocks),
            "think_char_count": len(think_content),
            "answer_char_count": len(answer_content),
        },
    )
    if debug_reward:
        print(
            f"reward:math  mode=reasoning  "
            f"expected={expected_answer!r}  "
            f"parsed={repr(parsed_answer) if parsed_answer else 'None'}  "
            f"normalized={repr(normalized_answer) if normalized_answer else 'None'}  "
            f"parse_pass={answer_parse_pass}  "
            f"accuracy_pass={accuracy_pass}  "
            f"format_pass={format_pass}  "
            f"truncated={truncated}  "
            f"finish_reason={finish_reason or 'None'}  "
            f"think_blocks={len(think_blocks)}  "
            f"answer_blocks={len(answer_blocks)}  "
            f"reward={result.reward:.1f}",
            flush=True
        )
    return result
def math_reward_fn(
    rollout: RolloutOutput,
    training_mode: str | None = None,
    debug_reward: bool = False,
) -> RewardOutput:
    """Compute math reward from rollout metadata.
    Args:
        rollout: Rollout output with completion and metadata
        training_mode: "math" for answer-only, "reasoning" for reasoning + answer
        debug_reward: Enable detailed logging of reward computation
    Returns:
        RewardOutput with combined score (0-1.0 range for math mode, 0-1.1 for reasoning mode)
    In "math" mode: Only checks answer correctness (0-1.0 range)
    In "reasoning" mode: Combines 70% reasoning score + 30% math score
    """
    resolved_training_mode = _resolve_math_training_mode(
        explicit_training_mode=training_mode,
        prompt_metadata=_prompt_metadata_from_rollout(rollout),
    )
    return _compute_math_score(
        rollout,
        training_mode=resolved_training_mode,
        debug_reward=debug_reward,
    )


def build_math_reward_fn(
    *,
    training_mode: str | None = None,
    debug_reward: bool = False,
):
    """Build a reward callable for config-driven launches."""

    def reward_fn(rollout: RolloutOutput) -> RewardOutput:
        return math_reward_fn(
            rollout,
            training_mode=training_mode,
            debug_reward=debug_reward,
        )

    return reward_fn


# Future code reasoning can live below as explicit `code_*` helpers later.
# Keep this example concrete until we have real code-task data and verifiers.
# Rollout hook
def reasoning_rollout_fn(
    prompts: list[Prompt],
    serving_backend,
    *,
    system_prompt: str | None = None,
) -> list[RolloutOutput]:
    """Generate one rollout per prompt with prompt metadata attached."""
    rendered_prompts = [
        _render_system_prefixed_prompt(prompt.text, system_prompt)
        for prompt in prompts
    ]
    samples = serving_backend.generate_batch(rendered_prompts)
    rollouts: list[RolloutOutput] = []
    for prompt, sample in zip(prompts, samples, strict=True):
        # The reward only sees RolloutOutput, so we copy the prompt metadata here
        # instead of reparsing the original prompt text later.
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.extend(
            [
                Message(role="user", content=prompt.text),
                Message(role="assistant", content=sample.text),
            ]
        )
        rollouts.append(
            RolloutOutput(
                text=sample.text,
                log_prob=sample.log_prob,
                prompt_token_ids=sample.prompt_token_ids,
                response_token_ids=sample.response_token_ids,
                response_token_logprobs=sample.response_token_logprobs,
                metadata={
                    **dict(sample.metadata),
                    "prompt_metadata": dict(prompt.metadata),
                },
                conversation=Conversation(messages=messages),
            )
        )
    return rollouts


def build_math_rollout(
    *,
    rollout_mode: str,
    training_mode: str | None = None,
    system_prompt: str | None = None,
):
    """Return either the example blackbox rollout or the traced whitebox rollout."""
    if rollout_mode == "whitebox" and training_mode is not None:
        resolved_system_prompt = system_prompt or build_math_system_prompt(training_mode)
        return build_math_whitebox_agent(
            training_mode=training_mode,
            system_prompt=resolved_system_prompt,
        )

    def rollout_fn(prompts: list[Prompt], serving_backend):
        resolved_training_mode = _resolve_math_training_mode(
            explicit_training_mode=training_mode,
            prompts=prompts,
        )
        resolved_system_prompt = system_prompt or build_math_system_prompt(resolved_training_mode)
        if rollout_mode == "whitebox":
            agent = build_math_whitebox_agent(
                training_mode=resolved_training_mode,
                system_prompt=resolved_system_prompt,
            )
            return agent.run_batch(prompts, serving_backend)
        return reasoning_rollout_fn(
            prompts,
            serving_backend,
            system_prompt=resolved_system_prompt,
        )

    return rollout_fn
# Runtime and CLI helpers
def find_default_vllm_python() -> str | None:
    """Return a prepared default vLLM runtime when one is available."""
    candidates: list[Path] = []
    if sys.platform == "darwin" and os.uname().machine == "arm64":
        candidates.append(Path.home() / ".venv-vllm-metal" / "bin" / "python")
    candidates.append(Path.home() / ".venv-vllm" / "bin" / "python")
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    current_python = Path(sys.executable)
    sibling_vllm = shutil.which("vllm", path=str(current_python.parent))
    if sibling_vllm is None:
        return None
    try:
        __import__("vllm")
    except Exception:
        return None
    return str(current_python)
def _config_uses_vllm(config_path: str | Path) -> bool:
    """Return whether one example config selects the vLLM serving backend."""
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return False
    framework = payload.get("framework")
    if not isinstance(framework, dict):
        return False
    serving = framework.get("serving")
    if not isinstance(serving, dict):
        return False
    return serving.get("backend") == "vllm"


def prepare_reasoning_environment(config_path: str | Path) -> None:
    """Populate example-only env defaults before YAML config loading."""
    if os.environ.get("FLASHRL_VLLM_PYTHON"):
        return
    if not _config_uses_vllm(config_path):
        return
    # This keeps the example one-command friendly when the repo's dedicated vLLM
    # runtime exists, without forcing users to hardcode runtime_python by hand.
    runtime_python = find_default_vllm_python()
    if runtime_python is not None:
        os.environ["FLASHRL_VLLM_PYTHON"] = runtime_python
def build_argument_parser() -> argparse.ArgumentParser:
    """Build the math example CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the FlashRL math example."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the FlashRL config.yaml file.",
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_MATH_DATASETS,
        default=DEFAULT_MATH_DATASET,
        help="Math dataset to use for training.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional number of training questions to load.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["math", "reasoning"],
        default="math",
        help="Training mode: 'math' for pure math capability, 'reasoning' for reasoning quality + math",
    )
    parser.add_argument(
        "--rollout-mode",
        choices=SUPPORTED_MATH_ROLLOUT_MODES,
        default=DEFAULT_MATH_ROLLOUT_MODE,
        help="Rollout implementation: 'blackbox' uses the example rollout hook, "
        "'whitebox' uses FlashRL's traced agent building blocks.",
    )
    parser.add_argument(
        "--debug-reward",
        action="store_true",
        help="Enable detailed reward computation logging for debugging",
    )
    return parser
def main(argv: list[str] | None = None) -> int:
    """Run the math example from the selected FlashRL config file."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_reasoning_environment(args.config)
    flashrl: FlashRL | None = None
    try:
        dataset = build_math_train_dataset(
            dataset=args.dataset,
            limit=args.train_limit,
            training_mode=args.training_mode,  # NEW
        )
        system_prompt = build_math_system_prompt(args.training_mode)
        rollout_impl = build_math_rollout(
            rollout_mode=args.rollout_mode,
            training_mode=args.training_mode,
            system_prompt=system_prompt,
        )
        # Create reward function wrapper that captures training_mode
        def reward_fn_with_mode(rollout: RolloutOutput) -> RewardOutput:
            return math_reward_fn(rollout, training_mode=args.training_mode, debug_reward=args.debug_reward)
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=rollout_impl,
            reward_fn=reward_fn_with_mode,
        )
        flashrl.train(dataset)
    except Exception as exc:
        print(f"\nFlashRL math example failed: {exc}", file=sys.stderr)
        print(
            "\nNote: This example loads a base Qwen checkpoint and a Hugging Face math dataset.",
            file=sys.stderr,
        )
        print(
            "If you're offline or have network issues, use a local model in the selected YAML config "
            "and make sure Hugging Face dataset access is available.",
            file=sys.stderr,
        )
        print(
            "Use --dataset to switch between gsm8k and aime25, --train-limit to cap the "
            "selected training set, and edit the YAML `checkpointing:` section when you "
            "want to change final-save or resume behavior.",
            file=sys.stderr,
        )
        print(
            "If you're using `serving.backend: vllm`, either set FLASHRL_VLLM_PYTHON to a "
            "prepared vLLM runtime or install FlashRL with the optional `vllm` extra in the "
            "current environment.",
            file=sys.stderr,
        )
        return 1
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
