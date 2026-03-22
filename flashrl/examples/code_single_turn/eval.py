"""Held-out evaluation for the Codeforces reasoning-code example."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable
from uuid import uuid4

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_CONFIG_PATH = EXAMPLE_DIR / "config.yaml"
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from flashrl.examples.code_single_turn import train as code_example

from flashrl.framework import FlashRL, log_paths
from flashrl.framework.agent import Agent
from flashrl.framework.data_models import Prompt, RewardOutput, RolloutOutput

EvalRewardFn = Callable[[RolloutOutput], RewardOutput]


def evaluate_model(
    flashrl: FlashRL,
    rollout_agent: Agent,
    *,
    dataset: list[Prompt],
    batch_size: int,
    reward_fn: EvalRewardFn,
) -> dict[str, float | int]:
    """Run held-out evaluation against the current serving backend."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if flashrl._serving_backend is None:  # pragma: no cover - defensive in live usage
        raise RuntimeError("FlashRL serving backend is not initialized.")

    flashrl._serving_backend.set_generation_defaults(
        max_new_tokens=flashrl.rollout_config.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        do_sample=False,
    )
    total_reward = 0.0
    solved = 0
    format_passes = 0
    truncations = 0
    total_pass_rate = 0.0
    sample_count = 0

    for start in range(0, len(dataset), batch_size):
        prompts = dataset[start:start + batch_size]
        rollouts = rollout_agent.run_batch(prompts, flashrl._serving_backend)
        rewards = [reward_fn(rollout) for rollout in rollouts]
        for reward in rewards:
            sample_count += 1
            total_reward += float(reward.reward)
            total_pass_rate += float(reward.metadata.get("pass_rate", 0.0))
            solved += int(bool(reward.metadata.get("accuracy_pass", False)))
            format_passes += int(bool(reward.metadata.get("format_pass", False)))
            truncations += int(bool(reward.metadata.get("truncated", False)))

    if sample_count == 0:
        return {
            "sample_count": 0,
            "reward_mean": 0.0,
            "pass_rate_mean": 0.0,
            "solve_rate": 0.0,
            "format_pass_rate": 0.0,
            "truncation_rate": 0.0,
        }

    return {
        "sample_count": sample_count,
        "reward_mean": total_reward / sample_count,
        "pass_rate_mean": total_pass_rate / sample_count,
        "solve_rate": solved / sample_count,
        "format_pass_rate": format_passes / sample_count,
        "truncation_rate": truncations / sample_count,
    }


def _checkpoint_run_dir_from_metadata(checkpoint_metadata: dict[str, Any] | None) -> Path | None:
    """Return the recorded training run directory when checkpoint metadata exposes it."""
    if not isinstance(checkpoint_metadata, dict):
        return None
    run_info = checkpoint_metadata.get("run")
    if not isinstance(run_info, dict):
        return None
    run_dir = run_info.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        return None
    resolved_run_dir = Path(run_dir)
    if not resolved_run_dir.exists():
        return None
    return resolved_run_dir


def _allocate_eval_run_dir(*, log_root: str | Path, model_name: str) -> Path:
    """Create one per-invocation eval run directory under the configured log root."""
    resolved_log_root = Path(log_root).expanduser()
    run_index = log_paths.allocate_run_index(resolved_log_root)
    run_id = (
        f"{log_paths.format_run_index(run_index)}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{log_paths.sanitize_model_name(model_name)}-"
        f"eval-{uuid4().hex[:8]}"
    )
    run_dir = resolved_log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_eval_run_dir(
    flashrl: FlashRL,
    *,
    checkpoint: str | None,
) -> Path:
    """Resolve the eval artifact run directory from checkpoint metadata or a fresh log run."""
    if checkpoint:
        controller = getattr(flashrl, "_controller", None)
        read_checkpoint_metadata = getattr(controller, "read_checkpoint_metadata", None)
        if callable(read_checkpoint_metadata):
            checkpoint_run_dir = _checkpoint_run_dir_from_metadata(
                read_checkpoint_metadata(str(checkpoint))
            )
            if checkpoint_run_dir is not None:
                return checkpoint_run_dir
    return _allocate_eval_run_dir(
        log_root=flashrl.logging_config.log_dir,
        model_name=flashrl.actor_config.model_name,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for held-out Codeforces evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the FlashRL Codeforces reasoning-code example."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the FlashRL config.yaml file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to load before evaluation.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional number of held-out problems to evaluate after filtering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=code_example.DEFAULT_REASONING_CODE_EVAL_BATCH_SIZE,
        help="Optional number of prompts to evaluate per generate_batch call.",
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
        default=code_example.DEFAULT_CODE_RATING_MAX,
        help="Optional maximum Codeforces rating filter.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=code_example.DEFAULT_RUN_TIMEOUT_SECONDS,
        help="Per-test execution timeout for generated Python solutions.",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=code_example.DEFAULT_MEMORY_LIMIT_MB,
        help="Best-effort local memory limit for generated Python solutions.",
    )
    parser.add_argument(
        "--max-tests-per-problem",
        type=int,
        default=None,
        help="Optional cap on official tests per problem for smoke runs.",
    )
    parser.add_argument(
        "--training-mode",
        choices=code_example.SUPPORTED_TRAINING_MODES,
        default="code",
        help="Training mode: 'code' for pure code capability, 'reasoning-code' for reasoning + code",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save generated code and rewards. Eval defaults to '<run_dir>/generated_code/' under logs/; absolute paths stay absolute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run held-out evaluation and print compact JSON metrics."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    code_example.prepare_reasoning_code_environment(args.config)

    flashrl: FlashRL | None = None
    resolved_eval_run_dir: Path | None = None

    def resolve_eval_run_dir() -> Path | None:
        return resolved_eval_run_dir

    try:
        rollout_agent = code_example.build_code_agent()
        dataset = code_example.build_code_eval_dataset(
            limit=args.eval_limit,
            rating_min=args.rating_min,
            rating_max=args.rating_max,
            max_tests_per_problem=args.max_tests_per_problem,
            training_mode=args.training_mode,
        )
        checkpoint = args.checkpoint
        if checkpoint is None:
            default_checkpoint = Path(code_example.DEFAULT_REASONING_CODE_CHECKPOINT_PATH)
            if default_checkpoint.exists():
                checkpoint = str(default_checkpoint)

        reward_fn = code_example.make_code_reward_fn(
            run_timeout_seconds=float(args.run_timeout_seconds),
            memory_limit_mb=args.memory_limit_mb,
            training_mode=args.training_mode,
            log_dir=args.log_dir,
            run_dir_resolver=resolve_eval_run_dir,
        )
        flashrl = FlashRL(
            config_path=args.config,
            rollout_fn=rollout_agent,
            reward_fn=reward_fn,
        )
        configured_log_dir = Path(args.log_dir) if args.log_dir is not None else None
        if configured_log_dir is None or not configured_log_dir.is_absolute():
            resolved_eval_run_dir = _resolve_eval_run_dir(
                flashrl,
                checkpoint=checkpoint,
            )
        if checkpoint:
            flashrl.load_checkpoint(checkpoint)
        metrics = evaluate_model(
            flashrl,
            rollout_agent,
            dataset=dataset,
            batch_size=args.batch_size,
            reward_fn=reward_fn,
        )
        print(json.dumps(metrics, ensure_ascii=True, sort_keys=True))
    except Exception as exc:
        print(f"FlashRL reasoning-code evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
