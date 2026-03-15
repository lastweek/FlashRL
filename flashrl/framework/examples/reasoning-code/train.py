"""Strict R1-style Codeforces training example with local execution reward."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable

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

import executor

DEFAULT_CODEFORCES_HF_DATASET = "open-r1/codeforces"
DEFAULT_CODEFORCES_HF_CONFIG = "verifiable-prompts"
DEFAULT_CODEFORCES_TRAIN_SPLIT = "train"
DEFAULT_CODEFORCES_EVAL_SPLIT = "test"
DEFAULT_CODE_LANGUAGE = "python"
DEFAULT_CODE_RATING_MAX = 1600
DEFAULT_REASONING_CODE_CHECKPOINT_PATH = "/tmp/flashrl_reasoning_code_checkpoint.pt"
DEFAULT_REASONING_CODE_EVAL_BATCH_SIZE = 4
DEFAULT_RUN_TIMEOUT_SECONDS = 4.0
DEFAULT_MEMORY_LIMIT_MB = 512
CODE_PREVIEW_MAX_LINES = 6
CODE_PREVIEW_MAX_CHARS = 240

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
STRICT_RESPONSE_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
CODE_BLOCK_PATTERN = re.compile(
    r"^\s*```(?P<lang>python|py)\s*\n(?P<code>.*?)\n```\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Keep per-problem tests and checkers out of prompt metadata so rollout logs stay readable.
CODEFORCES_EXECUTION_PAYLOADS: dict[str, dict[str, Any]] = {}


# Prompt contract


def render_code_prompt(problem_prompt: str) -> str:
    """Render one strict user prompt around the dataset-provided problem text."""
    return (
        "Solve the following Codeforces problem in Python.\n"
        "Respond with exactly one <think>...</think> block followed immediately by "
        "exactly one <answer>...</answer> block.\n"
        "Inside <answer>, output exactly one fenced Python code block.\n"
        "Do not output any text before <think> or after </answer>.\n\n"
        f"{problem_prompt.strip()}"
    )


# Dataset loading


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    """Validate one positive integer CLI override."""
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1 (got {parsed}).")
    return parsed


def _coerce_optional_int(value: Any | None) -> int | None:
    """Best-effort integer parsing for optional dataset fields."""
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any | None) -> float | None:
    """Best-effort float parsing for optional dataset fields."""
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_time_limit_seconds(value: Any | None) -> float | None:
    """Normalize time limits that may be expressed in seconds or milliseconds."""
    parsed = _coerce_optional_float(value)
    if parsed is None or parsed <= 0:
        return None
    if parsed > 60.0:
        return parsed / 1000.0
    return parsed


def _normalize_memory_limit_mb(value: Any | None) -> int | None:
    """Normalize memory limits that may be expressed in MB or raw bytes."""
    parsed = _coerce_optional_int(value)
    if parsed is None or parsed <= 0:
        return None
    if parsed > 4096:
        return max(1, parsed // (1024 * 1024))
    return parsed


def _resolve_limit(*, split_kind: str, explicit_limit: int | None) -> int | None:
    """Validate one optional split-specific limit."""
    if explicit_limit is None:
        return None
    return _coerce_positive_int(explicit_limit, field_name=f"{split_kind}_limit")


def _resolve_optional_rating(value: int | None, *, field_name: str) -> int | None:
    """Validate one optional rating filter."""
    if value is None:
        return None
    return _coerce_positive_int(value, field_name=field_name)


def _resolve_max_tests(max_tests_per_problem: int | None) -> int | None:
    """Validate the optional per-problem official test cap."""
    if max_tests_per_problem is None:
        return None
    return _coerce_positive_int(max_tests_per_problem, field_name="max_tests_per_problem")


def _load_dataset_module():
    """Import `datasets.load_dataset` lazily so failures stay example-local."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in live example usage
        raise RuntimeError(
            "flashrl/framework/examples/reasoning-code requires the `datasets` package. "
            "Install project dependencies or `pip install datasets`."
        ) from exc
    return load_dataset


def _print_dataset_summary(
    *,
    split: str,
    available: int,
    selected: int,
    rating_min: int | None,
    rating_max: int | None,
    max_tests_per_problem: int | None,
) -> None:
    """Print one compact dataset summary before training or evaluation starts."""
    if rating_min is None and rating_max is None:
        rating_text = "all"
    elif rating_min is None:
        rating_text = f"<= {rating_max}"
    elif rating_max is None:
        rating_text = f">= {rating_min}"
    else:
        rating_text = f"{rating_min}-{rating_max}"
    test_cap_text = str(max_tests_per_problem) if max_tests_per_problem is not None else "all"
    print(
        "dataset  "
        f"name=codeforces  source={DEFAULT_CODEFORCES_HF_DATASET}  "
        f"config={DEFAULT_CODEFORCES_HF_CONFIG}  split={split}  "
        f"available={available}  selected={selected}",
        flush=True,
    )
    print(
        "filters  "
        f"language={DEFAULT_CODE_LANGUAGE}  rating={rating_text}  "
        "input_mode=stdio  tests=official  checker=when-provided  "
        f"max_tests_per_problem={test_cap_text}",
        flush=True,
    )


def _normalize_official_tests(
    raw_tests: Any,
    *,
    max_tests_per_problem: int | None,
) -> list[dict[str, str]]:
    """Normalize the dataset's official tests into simple stdin/stdout pairs."""
    tests: list[dict[str, str]] = []
    if not isinstance(raw_tests, list):
        return tests
    for raw_test in raw_tests:
        if not isinstance(raw_test, dict):
            continue
        if "input" not in raw_test or "output" not in raw_test:
            continue
        tests.append(
            {
                "input": str(raw_test["input"]),
                "output": str(raw_test["output"]),
            }
        )
        if max_tests_per_problem is not None and len(tests) >= max_tests_per_problem:
            break
    return tests


def _build_problem_prompt(row: dict[str, Any]) -> str:
    """Prefer the dataset prompt and fall back to a minimal statement assembly."""
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt

    title = str(row.get("title") or row.get("name") or "").strip()
    statement = str(row.get("description") or row.get("problem_statement") or "").strip()
    input_format = str(row.get("input_format") or "").strip()
    output_format = str(row.get("output_format") or "").strip()
    examples = row.get("examples") or []

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if statement:
        parts.append(statement)
    if input_format:
        parts.append(f"Input\n{input_format}")
    if output_format:
        parts.append(f"Output\n{output_format}")
    if isinstance(examples, list) and examples:
        rendered_examples: list[str] = []
        for index, example in enumerate(examples[:3], start=1):
            if not isinstance(example, dict):
                continue
            example_input = str(example.get("input", "")).strip()
            example_output = str(example.get("output", "")).strip()
            rendered_examples.append(
                f"Example {index}\nInput:\n{example_input}\nOutput:\n{example_output}"
            )
        if rendered_examples:
            parts.append("\n\n".join(rendered_examples))
    return "\n\n".join(part for part in parts if part).strip()


def _build_task_id(row: dict[str, Any], *, split: str, index: int) -> str:
    """Build one stable task id for the selected Codeforces row."""
    for key in ("problem_id", "id"):
        value = row.get(key)
        if value not in {None, ""}:
            return f"codeforces-{split}-{value}"
    contest_id = row.get("contest_id")
    problem_index = row.get("problem_index") or row.get("index")
    if contest_id not in {None, ""} and problem_index not in {None, ""}:
        return f"codeforces-{split}-{contest_id}-{problem_index}"
    return f"codeforces-{split}-{index:06d}"


def _load_codeforces_split(
    split: str,
    *,
    limit: int | None = None,
    rating_min: int | None = None,
    rating_max: int | None = DEFAULT_CODE_RATING_MAX,
    max_tests_per_problem: int | None = None,
) -> list[dict[str, Any]]:
    """Load and normalize one filtered Codeforces split."""
    load_dataset = _load_dataset_module()
    try:
        raw_dataset = load_dataset(
            DEFAULT_CODEFORCES_HF_DATASET,
            DEFAULT_CODEFORCES_HF_CONFIG,
            split=split,
        )
    except Exception as exc:  # pragma: no cover - depends on external dataset availability
        raise RuntimeError("Unable to load the Codeforces dataset from Hugging Face.") from exc

    available_count = len(raw_dataset)
    selected_rows: list[dict[str, Any]] = []
    CODEFORCES_EXECUTION_PAYLOADS.clear()

    for index, row in enumerate(raw_dataset):
        language = str(row.get("language") or "").strip().lower()
        if language != DEFAULT_CODE_LANGUAGE:
            continue
        if bool(row.get("interactive") or row.get("is_interactive")):
            continue
        if not bool(row.get("official_tests_complete", False)):
            continue
        if str(row.get("input_mode") or "stdio").strip().lower() != "stdio":
            continue

        rating = _coerce_optional_int(row.get("rating")) or 0
        if rating_min is not None and rating < rating_min:
            continue
        if rating_max is not None and rating > rating_max:
            continue

        official_tests = _normalize_official_tests(
            row.get("official_tests"),
            max_tests_per_problem=max_tests_per_problem,
        )
        if not official_tests:
            continue

        problem_prompt = _build_problem_prompt(row)
        if not problem_prompt:
            continue

        task_id = _build_task_id(row, split=split, index=index)
        checker_code = str(row.get("generated_checker") or "").strip() or None
        CODEFORCES_EXECUTION_PAYLOADS[task_id] = {
            "official_tests": official_tests,
            "checker_code": checker_code,
            "time_limit_seconds": _normalize_time_limit_seconds(
                row.get("time_limit_seconds") or row.get("time_limit")
            ),
            "memory_limit_mb": _normalize_memory_limit_mb(
                row.get("memory_limit_mb") or row.get("memory_limit")
            ),
        }
        selected_rows.append(
            {
                "task_id": task_id,
                "source": DEFAULT_CODEFORCES_HF_DATASET,
                "config": DEFAULT_CODEFORCES_HF_CONFIG,
                "split": split,
                "prompt": problem_prompt,
                "language": DEFAULT_CODE_LANGUAGE,
                "rating": rating,
            }
        )
        if limit is not None and len(selected_rows) >= limit:
            break

    _print_dataset_summary(
        split=split,
        available=available_count,
        selected=len(selected_rows),
        rating_min=rating_min,
        rating_max=rating_max,
        max_tests_per_problem=max_tests_per_problem,
    )
    return selected_rows


def build_code_train_dataset(
    *,
    limit: int | None = None,
    rating_min: int | None = None,
    rating_max: int | None = DEFAULT_CODE_RATING_MAX,
    max_tests_per_problem: int | None = None,
) -> list[Prompt]:
    """Build the Codeforces training dataset used by the example CLI."""
    rows = _load_codeforces_split(
        DEFAULT_CODEFORCES_TRAIN_SPLIT,
        limit=_resolve_limit(split_kind="train", explicit_limit=limit),
        rating_min=_resolve_optional_rating(rating_min, field_name="rating_min"),
        rating_max=_resolve_optional_rating(rating_max, field_name="rating_max"),
        max_tests_per_problem=_resolve_max_tests(max_tests_per_problem),
    )
    return [
        Prompt(
            text=render_code_prompt(row["prompt"]),
            metadata={
                "task_id": row["task_id"],
                "source": row["source"],
                "config": row["config"],
                "split": row["split"],
                "language": row["language"],
                "rating": row["rating"],
                "verifier": "python_tests",
            },
        )
        for row in rows
    ]


def build_code_eval_dataset(
    *,
    limit: int | None = None,
    rating_min: int | None = None,
    rating_max: int | None = DEFAULT_CODE_RATING_MAX,
    max_tests_per_problem: int | None = None,
) -> list[Prompt]:
    """Build the held-out Codeforces evaluation dataset."""
    rows = _load_codeforces_split(
        DEFAULT_CODEFORCES_EVAL_SPLIT,
        limit=_resolve_limit(split_kind="eval", explicit_limit=limit),
        rating_min=_resolve_optional_rating(rating_min, field_name="rating_min"),
        rating_max=_resolve_optional_rating(rating_max, field_name="rating_max"),
        max_tests_per_problem=_resolve_max_tests(max_tests_per_problem),
    )
    return [
        Prompt(
            text=render_code_prompt(row["prompt"]),
            metadata={
                "task_id": row["task_id"],
                "source": row["source"],
                "config": row["config"],
                "split": row["split"],
                "language": row["language"],
                "rating": row["rating"],
                "verifier": "python_tests",
            },
        )
        for row in rows
    ]


# Response parsing and reward


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


def _extract_python_code(answer_text: str) -> str | None:
    """Extract the single fenced Python block from the answer text."""
    match = CODE_BLOCK_PATTERN.fullmatch(answer_text)
    if match is None:
        return None
    code = match.group("code").strip()
    if not code:
        return None
    return code


def _build_code_preview(text: str | None) -> str:
    """Keep a short readable preview of the generated answer or code."""
    if text is None:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""

    preview_lines = stripped.splitlines()[:CODE_PREVIEW_MAX_LINES]
    preview = "\n".join(line.rstrip() for line in preview_lines).strip()
    if len(stripped.splitlines()) > CODE_PREVIEW_MAX_LINES:
        preview += "\n..."
    if len(preview) > CODE_PREVIEW_MAX_CHARS:
        preview = preview[:CODE_PREVIEW_MAX_CHARS].rstrip() + "..."
    return preview


def _print_code_reward_summary(
    *,
    task_id: str,
    reward: float,
    format_pass: bool,
    truncated: bool,
    execution_result: executor.ExecutionResult,
    code_preview: str,
) -> None:
    """Print a compact execution summary for the current rollout."""
    status = "passed" if execution_result.failure_reason is None else execution_result.failure_reason
    line = (
        "code     "
        f"task={task_id or 'unknown'}  "
        f"tests={execution_result.passed_tests}/{execution_result.total_tests}  "
        f"pass_rate={execution_result.pass_rate:.2f}  "
        f"format={'ok' if format_pass else 'bad'}  "
        f"reward={reward:.2f}  "
        f"exec={execution_result.execution_seconds:.3f}s  "
        f"status={status}"
    )
    if truncated:
        line += "  truncated=yes"
    if execution_result.checker_used:
        line += "  checker=yes"
    print(line, flush=True)
    if code_preview:
        print(f"preview  {code_preview.replace(chr(10), ' | ')}", flush=True)


def score_code_rollout(
    rollout: RolloutOutput,
    *,
    run_timeout_seconds: float,
    memory_limit_mb: int | None,
) -> RewardOutput:
    """Score one generated Codeforces rollout with local test execution."""
    text = rollout.text
    prompt_metadata = _prompt_metadata_from_rollout(rollout)
    finish_reason = rollout.metadata.get("finish_reason")
    truncated = finish_reason == "length"
    task_id = str(prompt_metadata.get("task_id", ""))
    execution_payload = CODEFORCES_EXECUTION_PAYLOADS.get(task_id, {})
    official_tests = execution_payload.get("official_tests") or []
    checker_code = execution_payload.get("checker_code")

    think_blocks = THINK_BLOCK_PATTERN.findall(text)
    answer_blocks = ANSWER_BLOCK_PATTERN.findall(text)
    strict_match = STRICT_RESPONSE_PATTERN.fullmatch(text)
    strict_answer = strict_match.group("answer").strip() if strict_match is not None else ""
    strict_code = _extract_python_code(strict_answer) if strict_answer else None
    format_pass = bool(
        strict_match is not None
        and len(think_blocks) == 1
        and len(answer_blocks) == 1
        and strict_match.group("think").strip()
        and strict_code is not None
        and not truncated
    )

    answer_text = _extract_answer_block(text)
    extracted_code = _extract_python_code(answer_text or "") if answer_text else None
    execution_result: executor.ExecutionResult
    if extracted_code is None or not official_tests:
        execution_result = executor.ExecutionResult(
            passed_tests=0,
            total_tests=len(official_tests),
            pass_rate=0.0,
            execution_seconds=0.0,
            failure_reason="missing_code" if answer_text is None else "invalid_code_fence",
            checker_used=False,
        )
        if not official_tests:
            execution_result.failure_reason = "missing_execution_payload"
    else:
        dataset_timeout = execution_payload.get("time_limit_seconds")
        effective_timeout = max(
            run_timeout_seconds,
            float(dataset_timeout) if isinstance(dataset_timeout, (int, float)) and dataset_timeout > 0 else 0.0,
        )
        dataset_memory_limit = execution_payload.get("memory_limit_mb")
        effective_memory_limit = memory_limit_mb
        if isinstance(dataset_memory_limit, int) and dataset_memory_limit > 0:
            if effective_memory_limit is None:
                effective_memory_limit = dataset_memory_limit
            else:
                effective_memory_limit = min(effective_memory_limit, dataset_memory_limit)
        execution_result = executor.run_python_solution(
            extracted_code,
            official_tests=official_tests,
            checker_code=checker_code,
            timeout_seconds=effective_timeout,
            memory_limit_mb=effective_memory_limit,
        )

    accuracy_pass = bool(
        execution_result.total_tests > 0
        and execution_result.passed_tests == execution_result.total_tests
    )
    reward = float(execution_result.pass_rate) + (0.1 if format_pass else 0.0)
    code_preview = _build_code_preview(extracted_code or answer_text or text)
    execution_status = "passed" if accuracy_pass else execution_result.failure_reason or "failed"

    _print_code_reward_summary(
        task_id=task_id,
        reward=reward,
        format_pass=format_pass,
        truncated=truncated,
        execution_result=execution_result,
        code_preview=code_preview,
    )

    return RewardOutput(
        reward=reward,
        metadata={
            "task_id": task_id,
            "accuracy_pass": accuracy_pass,
            "format_pass": format_pass,
            "truncated": truncated,
            "finish_reason": finish_reason,
            "passed_tests": execution_result.passed_tests,
            "total_tests": execution_result.total_tests,
            "pass_rate": execution_result.pass_rate,
            "execution_seconds": execution_result.execution_seconds,
            "failure_reason": execution_result.failure_reason,
            "execution_status": execution_status,
            "checker_used": execution_result.checker_used,
            "code_preview": code_preview,
            "rating": prompt_metadata.get("rating"),
        },
    )


def make_code_reward_fn(
    *,
    run_timeout_seconds: float,
    memory_limit_mb: int | None,
) -> Callable[[RolloutOutput], RewardOutput]:
    """Build one reward function with the selected local execution limits."""

    def reward_fn(rollout: RolloutOutput) -> RewardOutput:
        return score_code_rollout(
            rollout,
            run_timeout_seconds=run_timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )

    return reward_fn


# Rollout hook


def reasoning_code_rollout_fn(
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


def prepare_reasoning_code_environment(config_path: str) -> None:
    """Populate example-only env defaults before profile loading."""
    if os.environ.get("FLASHRL_VLLM_PYTHON"):
        return
    if Path(config_path).name != "config_vllm.yaml":
        return

    runtime_python = find_default_vllm_python()
    if runtime_python is not None:
        os.environ["FLASHRL_VLLM_PYTHON"] = runtime_python


# CLI


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the reasoning-code CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the FlashRL Codeforces reasoning-code example."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_vllm.yaml")),
        help="Path to the FlashRL runtime/training profile.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional number of training problems to load after filtering.",
    )
    parser.add_argument(
        "--rating-min",
        type=int,
        default=None,
        help="Optional minimum Codeforces rating filter.",
    )
    parser.add_argument(
        "--rating-max",
        type=int,
        default=DEFAULT_CODE_RATING_MAX,
        help="Optional maximum Codeforces rating filter.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        help="Per-test execution timeout for generated Python solutions.",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=DEFAULT_MEMORY_LIMIT_MB,
        help="Best-effort local memory limit for generated Python solutions.",
    )
    parser.add_argument(
        "--max-tests-per-problem",
        type=int,
        default=None,
        help="Optional cap on official tests per problem for smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Codeforces reasoning-code example from the selected profile."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_reasoning_code_environment(args.config)

    flashrl: FlashRL | None = None
    try:
        dataset = build_code_train_dataset(
            limit=args.train_limit,
            rating_min=args.rating_min,
            rating_max=args.rating_max,
            max_tests_per_problem=args.max_tests_per_problem,
        )
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=reasoning_code_rollout_fn,
            reward_fn=make_code_reward_fn(
                run_timeout_seconds=float(args.run_timeout_seconds),
                memory_limit_mb=args.memory_limit_mb,
            ),
        )
        flashrl.train(dataset)
    except Exception as exc:
        print(f"\nFlashRL reasoning-code example failed: {exc}", file=sys.stderr)
        print(
            "\nNote: This example loads open-r1/codeforces and runs generated Python code "
            "locally against official tests.",
            file=sys.stderr,
        )
        print(
            "If you're offline, have dataset access issues, or do not want local code execution, "
            "this example will fail early.",
            file=sys.stderr,
        )
        print(
            "Use --train-limit, --rating-max, and --max-tests-per-problem to keep the first run "
            "small and readable. Edit the YAML `checkpointing:` section when you want to "
            "change final-save or resume behavior.",
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
