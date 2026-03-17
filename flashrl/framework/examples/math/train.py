"""Strict R1-Zero math training example with explicit dataset selection."""
from __future__ import annotations
import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)
from flashrl.framework import FlashRL
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
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
FINAL_ANSWER_PATTERN = re.compile(r"####\s*(.+)$", re.MULTILINE)
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
STRICT_RESPONSE_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Prompt contract
def render_math_prompt(problem: str, training_mode: str = "math") -> str:
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
            "Show your work in a ```thinking``` block, then provide your final answer "
            "in an <answer>...</answer> block.\n\n"
            f"Problem: {problem.strip()}"
        )
    else:
        # Math mode: Just ask for the problem
        return (
            "Solve the following math problem.\n\n"
            f"Problem: {problem.strip()}"
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
            "flashrl/framework/examples/math requires the `datasets` package to load math datasets. "
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
    training_mode: str = "math",  # NEW
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
            },
        )
        for row in rows
    ]
def build_math_eval_dataset(
    dataset: str = DEFAULT_MATH_DATASET,
    limit: int | None = None,
    training_mode: str = "math",  # NEW
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
    """Extract the single parsed answer block when one exists.
    Falls back to extracting the last number from free-form text when no tags found."""
    # First try structured <answer> blocks (for reasoning mode compatibility)
    matches = ANSWER_BLOCK_PATTERN.findall(response_text)
    if len(matches) == 1:
        answer_text = matches[0].strip()
        if answer_text:
            return answer_text
    # Fallback: Extract last number from free-form text (for math mode)
    # Handles patterns like "The answer is 42." or "Answer: 13.25"
    number_pattern = re.compile(r'-?\d+(?:\.\d+)?|\d+/\d+')
    numbers = number_pattern.findall(response_text)
    if numbers:
        return numbers[-1]  # Return the last number found in the text
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
    thinking_pattern = r'<thinking>(.*?)</thinking>'
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
        parsed_answer = _extract_answer_block(text)
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
def math_reward_fn(rollout: RolloutOutput, training_mode: str = "math", debug_reward: bool = False) -> RewardOutput:
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
    if training_mode == "reasoning":
        # Reasoning mode: Combine reasoning + math scores
        reasoning_result = reasoning_reward_fn(rollout, debug_reward=debug_reward)
        math_result = _compute_math_score(rollout, training_mode="reasoning", debug_reward=debug_reward)  # Pass mode
        # Combine: 70% reasoning + 30% math
        combined_score = (
            0.7 * reasoning_result.reward +
            0.3 * math_result.reward
        )
        # Merge metadata
        metadata = {
            **reasoning_result.metadata,
            **math_result.metadata,
            "training_mode": "reasoning",
            "combined_score": combined_score
        }
        if debug_reward:
            print(
                f"reward:combined  mode=reasoning  "
                f"reasoning_score={reasoning_result.reward:.3f} (70%)  "
                f"math_score={math_result.reward:.1f} (30%)  "
                f"combined_score={combined_score:.3f}",
                flush=True
            )
        return RewardOutput(
            reward=combined_score,
            metadata=metadata
        )
    else:
        # Math mode: Only check answer accuracy
        return _compute_math_score(rollout, training_mode="math", debug_reward=debug_reward)  # Pass mode
# Future code reasoning can live below as explicit `code_*` helpers later.
# Keep this example concrete until we have real code-task data and verifiers.
# Rollout hook
def reasoning_rollout_fn(
    prompts: list[Prompt],
    serving_backend,
) -> list[RolloutOutput]:
    """Generate one rollout per prompt with prompt metadata attached."""
    samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
    rollouts: list[RolloutOutput] = []
    for prompt, sample in zip(prompts, samples, strict=True):
        # The reward only sees RolloutOutput, so we copy the prompt metadata here
        # instead of reparsing the original prompt text later.
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
                conversation=Conversation(
                    messages=[
                        Message(role="user", content=prompt.text),
                        Message(role="assistant", content=sample.text),
                    ]
                ),
            )
        )
    return rollouts
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
def prepare_reasoning_environment(config_path: str) -> None:
    """Populate example-only env defaults before YAML config loading."""
    if os.environ.get("FLASHRL_VLLM_PYTHON"):
        return
    if Path(config_path).name != "config_vllm.yaml":
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
        default=str(Path(__file__).with_name("config_vllm.yaml")),
        help="Path to the FlashRL runtime/training profile.",
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
        "--debug-reward",
        action="store_true",
        help="Enable detailed reward computation logging for debugging",
    )
    return parser
def main(argv: list[str] | None = None) -> int:
    """Run the math example from the selected FlashRL profile."""
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
        # Create reward function wrapper that captures training_mode
        def reward_fn_with_mode(rollout: RolloutOutput) -> RewardOutput:
            return math_reward_fn(rollout, training_mode=args.training_mode, debug_reward=args.debug_reward)
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=reasoning_rollout_fn,
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
            "If you're offline or have network issues, use a local model in the YAML profile "
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
