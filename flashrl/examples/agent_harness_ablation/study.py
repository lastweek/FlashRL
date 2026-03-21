"""Reusable study helpers for the agent harness ablation example."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any

from flashrl.examples.agent_harness.common import load_yaml_mapping
from flashrl.examples.agent_harness.config import AgentHarnessConfig
from flashrl.examples.agent_harness.dataset import build_train_dataset, reward_fn
from flashrl.examples.agent_harness.evaluation import evaluate_model
from flashrl.examples.agent_harness.harness import build_agent_harness
from flashrl.framework import FlashRL
from flashrl.framework.config import RunConfig

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXAMPLE_DIR / "config.yaml"
DEFAULT_MANIFEST_GLOB = "logs/studies/*/manifest.json"
DEFAULT_STUDY_NAME = "agent-harness-ablation"


def load_study_payload(config_path: str | Path) -> dict[str, Any]:
    payload = load_yaml_mapping(config_path)
    study = payload.get("study")
    if not isinstance(study, dict):
        raise ValueError("config.study must be a mapping.")
    return study


def expand_matrix(study: dict[str, Any]) -> list[tuple[str, AgentHarnessConfig]]:
    base_harness = AgentHarnessConfig.from_mapping(study.get("base_harness"))
    rows = study.get("matrix")
    if not isinstance(rows, list) or not rows:
        raise ValueError("study.matrix must be a non-empty list.")
    variants: list[tuple[str, AgentHarnessConfig]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each study.matrix row must be a mapping.")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("Each study.matrix row must have a non-empty name.")
        harness_overrides = row.get("harness") or {}
        if not isinstance(harness_overrides, dict):
            raise ValueError(f"study.matrix[{name}].harness must be a mapping.")
        variants.append((name, base_harness.with_overrides(harness_overrides)))
    return variants


def run_study(config_path: str | Path) -> Path:
    study = load_study_payload(config_path)
    variants = expand_matrix(study)
    seeds = [int(seed) for seed in study.get("seeds", [42])]
    train_limit, eval_limit = _resolve_dataset_limits(study)
    train_dataset = build_train_dataset(limit=train_limit)
    output_dir = _create_output_dir(str(study.get("name", DEFAULT_STUDY_NAME)))
    manifest_path = output_dir / "manifest.json"
    manifest = _initialize_manifest(study, config_path=config_path, output_dir=output_dir)

    for variant_name, harness_config in variants:
        for seed in seeds:
            flashrl: FlashRL | None = None
            run_config = RunConfig.from_yaml(config_path)
            run_config.controller.seed = int(seed)
            rollout_agent = build_agent_harness(harness_config)
            try:
                flashrl = FlashRL(
                    run_config=run_config,
                    rollout_fn=rollout_agent,
                    reward_fn=reward_fn,
                )
                flashrl.train(train_dataset)
                eval_summary = evaluate_model(
                    flashrl,
                    rollout_agent,
                    limit=eval_limit,
                )
                manifest["runs"].append(
                    _build_manifest_run_entry(
                        variant_name=variant_name,
                        seed=seed,
                        harness_config=harness_config,
                        eval_summary=eval_summary,
                        run_dir=flashrl.run_dir,
                    )
                )
                _write_manifest(manifest, manifest_path)
            finally:
                if flashrl is not None:
                    flashrl.close()

    return manifest_path


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("runs", []):
        if not isinstance(entry, dict):
            continue
        variant = str(entry.get("variant", "")).strip()
        if not variant:
            continue
        grouped.setdefault(variant, []).append(entry)

    variants: list[dict[str, Any]] = []
    for variant_name, entries in sorted(grouped.items()):
        accuracies = [float(entry["evaluation"]["eval_accuracy"]) for entry in entries]
        token_costs = [float(entry["evaluation"]["mean_total_model_tokens"]) for entry in entries]
        rollout_seconds = [float(entry["evaluation"]["mean_rollout_seconds"]) for entry in entries]
        variants.append(
            {
                "variant": variant_name,
                "seed_count": len(entries),
                "eval_accuracy_mean": _mean(accuracies),
                "eval_accuracy_std": _std(accuracies),
                "mean_total_model_tokens_mean": _mean(token_costs),
                "mean_total_model_tokens_std": _std(token_costs),
                "mean_rollout_seconds_mean": _mean(rollout_seconds),
                "mean_rollout_seconds_std": _std(rollout_seconds),
                "runs": entries,
            }
        )

    frontier = _pareto_frontier(variants)
    for variant in variants:
        variant["pareto_optimal"] = variant["variant"] in frontier
    variants.sort(
        key=lambda item: (
            -float(item["eval_accuracy_mean"]),
            float(item["mean_total_model_tokens_mean"]),
            float(item["mean_rollout_seconds_mean"]),
        )
    )
    return {
        "study_name": manifest.get("study_name"),
        "variant_count": len(variants),
        "variants": variants,
    }


def write_reports(summary: dict[str, Any], *, manifest_path: str | Path) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    output_dir = manifest_path.parent
    json_path = output_dir / "summary.json"
    text_path = output_dir / "leaderboard.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# {summary.get('study_name', 'Agent Harness Study')}",
        "",
        "| Variant | Pareto | Accuracy | Mean Tokens | Mean Seconds | Seeds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in summary["variants"]:
        lines.append(
            "| {variant} | {pareto} | {accuracy:.3f} | {tokens:.1f} | {seconds:.3f} | {seeds} |".format(
                variant=variant["variant"],
                pareto="yes" if variant["pareto_optimal"] else "no",
                accuracy=float(variant["eval_accuracy_mean"]),
                tokens=float(variant["mean_total_model_tokens_mean"]),
                seconds=float(variant["mean_rollout_seconds_mean"]),
                seeds=int(variant["seed_count"]),
            )
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path


def find_latest_manifest() -> Path | None:
    candidates = sorted(Path("logs/studies").glob("*/manifest.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_dataset_limits(study: dict[str, Any]) -> tuple[int | None, int | None]:
    dataset_cfg = study.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        raise ValueError("study.dataset must be a mapping when present.")
    train_limit = int(dataset_cfg["train_limit"]) if isinstance(dataset_cfg.get("train_limit"), int) else None
    eval_limit = int(dataset_cfg["eval_limit"]) if isinstance(dataset_cfg.get("eval_limit"), int) else None
    return train_limit, eval_limit


def _create_output_dir(study_name: str) -> Path:
    output_dir = Path("logs") / "studies" / f"{study_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _initialize_manifest(
    study: dict[str, Any],
    *,
    config_path: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "study_name": str(study.get("name", DEFAULT_STUDY_NAME)),
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "runs": [],
    }


def _build_manifest_run_entry(
    *,
    variant_name: str,
    seed: int,
    harness_config: AgentHarnessConfig,
    eval_summary: dict[str, object],
    run_dir: Path | None,
) -> dict[str, Any]:
    return {
        "variant": variant_name,
        "seed": int(seed),
        "run_dir": str(run_dir.resolve()) if run_dir is not None else None,
        "harness": asdict(harness_config),
        "evaluation": eval_summary,
    }


def _write_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _pareto_frontier(variants: list[dict[str, Any]]) -> set[str]:
    frontier: set[str] = set()
    for candidate in variants:
        dominated = False
        for other in variants:
            if other is candidate:
                continue
            if (
                float(other["eval_accuracy_mean"]) >= float(candidate["eval_accuracy_mean"])
                and float(other["mean_total_model_tokens_mean"]) <= float(candidate["mean_total_model_tokens_mean"])
                and float(other["mean_rollout_seconds_mean"]) <= float(candidate["mean_rollout_seconds_mean"])
                and (
                    float(other["eval_accuracy_mean"]) > float(candidate["eval_accuracy_mean"])
                    or float(other["mean_total_model_tokens_mean"]) < float(candidate["mean_total_model_tokens_mean"])
                    or float(other["mean_rollout_seconds_mean"]) < float(candidate["mean_rollout_seconds_mean"])
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(str(candidate["variant"]))
    return frontier


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))
