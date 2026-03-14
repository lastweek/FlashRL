"""Strict R1-Zero style math training example backed by GSM8K."""

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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flashrl.framework import FlashRL
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
)


GSM8K_DATASET_CANDIDATES = (
    ("openai/gsm8k", "main"),
    ("gsm8k", "main"),
)
FINAL_ANSWER_PATTERN = re.compile(r"####\s*(.+)$", re.MULTILINE)
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
STRICT_RESPONSE_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def render_reasoning_prompt(problem: str) -> str:
    """Render one strict user prompt with no system role."""
    return (
        "Solve the following math problem.\n"
        "Respond with exactly one <think>...</think> block followed immediately by "
        "exactly one <answer>...</answer> block.\n"
        "Put only the final answer inside <answer>.\n"
        "Do not output any text before <think> or after </answer>.\n\n"
        f"Problem: {problem.strip()}"
    )


def _normalize_answer_text(text: str) -> str:
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


def _extract_ground_truth_answer(raw_answer: str) -> str:
    """Extract and normalize the final GSM8K answer after the `####` marker."""
    match = FINAL_ANSWER_PATTERN.search(raw_answer)
    if match is None:
        raise ValueError(f"GSM8K answer is missing a final '####' marker: {raw_answer[:80]!r}")
    normalized = _normalize_answer_text(match.group(1))
    if not normalized:
        raise ValueError(f"GSM8K answer normalized to empty text: {raw_answer[:80]!r}")
    return normalized


def _dataset_limit_from_env(split: str) -> int | None:
    """Read an optional dataset limit from environment variables."""
    specific_key = f"FLASHRL_REASONING_{split.upper()}_LIMIT"
    raw_value = os.environ.get(specific_key) or os.environ.get("FLASHRL_REASONING_DATASET_LIMIT")
    if raw_value is None:
        return None
    limit = int(raw_value)
    if limit < 1:
        raise ValueError(f"{specific_key} must be >= 1 when set (got {limit}).")
    return limit


def _load_gsm8k_split(
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Load and normalize one GSM8K split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in live example usage
        raise RuntimeError(
            "examples.reasoning requires the `datasets` package to load GSM8K. "
            "Install project dependencies or `pip install datasets`."
        ) from exc

    dataset_error: Exception | None = None
    raw_dataset = None
    for dataset_name, config_name in GSM8K_DATASET_CANDIDATES:
        try:
            raw_dataset = load_dataset(dataset_name, config_name, split=split)
            break
        except Exception as exc:  # pragma: no cover - depends on external dataset availability
            dataset_error = exc
    if raw_dataset is None:
        raise RuntimeError("Unable to load GSM8K from Hugging Face.") from dataset_error

    if limit is not None and hasattr(raw_dataset, "select"):
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))

    rows: list[dict[str, str]] = []
    for index, row in enumerate(raw_dataset):
        task_id = str(row.get("id") or f"gsm8k-{split}-{index:06d}")
        problem = str(row["question"]).strip()
        final_answer = _extract_ground_truth_answer(str(row["answer"]))
        rows.append(
            {
                "task_id": task_id,
                "source": "gsm8k",
                "split": split,
                "problem": problem,
                "final_answer": final_answer,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _build_prompt_dataset(split: str, limit: int | None = None) -> list[Prompt]:
    """Build one prompt dataset split with stable metadata."""
    resolved_limit = _dataset_limit_from_env(split) if limit is None else limit
    rows = _load_gsm8k_split(split, limit=resolved_limit)
    prompts: list[Prompt] = []
    for row in rows:
        prompts.append(
            Prompt(
                text=render_reasoning_prompt(row["problem"]),
                metadata={
                    "task_id": row["task_id"],
                    "source": row["source"],
                    "split": row["split"],
                    "problem": row["problem"],
                    "final_answer": row["final_answer"],
                    "verifier": "numeric_exact",
                },
            )
        )
    return prompts


def build_dataset() -> list[Prompt]:
    """Build the training split used by the YAML example."""
    return _build_prompt_dataset("train")


def build_eval_dataset(limit: int | None = None) -> list[Prompt]:
    """Build the held-out evaluation split."""
    return _build_prompt_dataset("test", limit=limit)


def reasoning_rollout_fn(
    prompts: list[Prompt],
    serving_backend,
) -> list[RolloutOutput]:
    """Generate one reasoning rollout per prompt with the serving backend."""
    samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
    rollouts: list[RolloutOutput] = []
    for prompt, sample in zip(prompts, samples, strict=True):
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


def _prompt_metadata_from_rollout(rollout: RolloutOutput) -> dict[str, Any]:
    """Recover the original prompt metadata attached by the rollout hook."""
    prompt_metadata = rollout.metadata.get("prompt_metadata")
    if isinstance(prompt_metadata, dict):
        return prompt_metadata
    return {}


def _extract_answer_block(response_text: str) -> str | None:
    """Extract the single parsed answer block when one exists."""
    matches = ANSWER_BLOCK_PATTERN.findall(response_text)
    if len(matches) != 1:
        return None
    answer_text = matches[0].strip()
    if not answer_text:
        return None
    return answer_text


def reasoning_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Compute strict format and exact-answer rewards from rollout metadata."""
    text = rollout.text
    prompt_metadata = _prompt_metadata_from_rollout(rollout)
    expected_answer = _normalize_answer_text(str(prompt_metadata.get("final_answer", "")))
    finish_reason = rollout.metadata.get("finish_reason")
    truncated = finish_reason == "length"

    think_blocks = THINK_BLOCK_PATTERN.findall(text)
    answer_blocks = ANSWER_BLOCK_PATTERN.findall(text)
    strict_match = STRICT_RESPONSE_PATTERN.fullmatch(text)
    parsed_answer = _extract_answer_block(text)
    normalized_answer = _normalize_answer_text(parsed_answer or "")

    has_single_think_block = len(think_blocks) == 1
    has_single_answer_block = len(answer_blocks) == 1
    think_content = strict_match.group("think").strip() if strict_match is not None else ""
    answer_content = strict_match.group("answer").strip() if strict_match is not None else ""
    format_pass = bool(
        strict_match is not None
        and has_single_think_block
        and has_single_answer_block
        and think_content
        and answer_content
        and not truncated
    )
    answer_parse_pass = has_single_answer_block and parsed_answer is not None
    accuracy_pass = bool(answer_parse_pass and normalized_answer == expected_answer)

    reward = 0.0
    if accuracy_pass:
        reward += 1.0
    if format_pass:
        reward += 0.1

    return RewardOutput(
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the example CLI parser."""
    parser = argparse.ArgumentParser(description="Run the FlashRL reasoning example.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_vllm.yaml")),
        help="Path to the example YAML config file.",
    )
    return parser


def _default_vllm_python() -> str | None:
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


def _prepare_example_environment(config_path: str) -> None:
    """Populate example-only env defaults before YAML config loading."""
    if os.environ.get("FLASHRL_VLLM_PYTHON"):
        return
    if Path(config_path).name != "config_vllm.yaml":
        return

    runtime_python = _default_vllm_python()
    if runtime_python is not None:
        os.environ["FLASHRL_VLLM_PYTHON"] = runtime_python


def main(argv: list[str] | None = None) -> int:
    """Run the reasoning example from YAML."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    _prepare_example_environment(args.config)

    flashrl: FlashRL | None = None
    try:
        flashrl = FlashRL.from_yaml(args.config)
        flashrl.train()
        flashrl.save_checkpoint("/tmp/flashrl_reasoning_checkpoint.pt")
    except Exception as exc:
        print(f"\nFlashRL reasoning example failed: {exc}", file=sys.stderr)
        print(
            "\nNote: This example loads a base Qwen checkpoint and the GSM8K dataset.",
            file=sys.stderr,
        )
        print(
            "If you're offline or have network issues, use a local model in the YAML config "
            "and make sure Hugging Face dataset access is available.",
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
