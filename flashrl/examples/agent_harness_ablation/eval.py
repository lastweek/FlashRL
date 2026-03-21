"""Aggregate an agent harness ablation-study manifest into comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_MANIFEST_GLOB = "logs/studies/*/manifest.json"
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected manifest mapping at {path}.")
    return payload


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
        summary = {
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
        variants.append(summary)

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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate agent harness ablation reports.")
    parser.add_argument("--manifest", default=None, help="Path to one study manifest. Defaults to the newest manifest under logs/studies/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest) if args.manifest else _find_latest_manifest()
    if manifest_path is None:
        raise FileNotFoundError(f"No manifest found matching {DEFAULT_MANIFEST_GLOB!r}.")
    summary = summarize_manifest(load_manifest(manifest_path))
    json_path, text_path = write_reports(summary, manifest_path=manifest_path)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path.resolve()),
                "summary_path": str(json_path.resolve()),
                "leaderboard_path": str(text_path.resolve()),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _find_latest_manifest() -> Path | None:
    candidates = sorted(Path("logs/studies").glob("*/manifest.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


if __name__ == "__main__":
    raise SystemExit(main())
