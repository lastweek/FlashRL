"""Run a local-first ablation study over agent harness variants."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_CONFIG_PATH = EXAMPLE_DIR / "config.yaml"
for candidate in (REPO_ROOT, EXAMPLE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from flashrl.examples.agent_harness.config import CodingHarnessConfig
from flashrl.examples.agent_harness.eval import evaluate_model
from flashrl.examples.agent_harness.harness import (
    build_coding_agent,
    build_coding_reward_fn,
    build_coding_train_dataset,
)
from flashrl.examples.agent_harness.train import prepare_environment
from flashrl.framework import FlashRL
from flashrl.framework.config import RunConfig


def load_study_payload(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {config_path}.")
    study = payload.get("study")
    if not isinstance(study, dict):
        raise ValueError("config.study must be a mapping.")
    return study


def expand_matrix(study: dict[str, Any]) -> list[tuple[str, CodingHarnessConfig]]:
    base_harness = CodingHarnessConfig.from_mapping(study.get("base_harness"))
    rows = study.get("matrix")
    if not isinstance(rows, list) or not rows:
        raise ValueError("study.matrix must be a non-empty list.")
    variants: list[tuple[str, CodingHarnessConfig]] = []
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the agent harness ablation study.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the study config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    prepare_environment(args.config)
    study = load_study_payload(args.config)
    variants = expand_matrix(study)
    seeds = [int(seed) for seed in study.get("seeds", [42])]
    dataset_cfg = study.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        raise ValueError("study.dataset must be a mapping when present.")
    train_limit = int(dataset_cfg["train_limit"]) if isinstance(dataset_cfg.get("train_limit"), int) else None
    eval_limit = int(dataset_cfg["eval_limit"]) if isinstance(dataset_cfg.get("eval_limit"), int) else None
    train_dataset = build_coding_train_dataset(limit=train_limit)
    output_dir = Path("logs") / "studies" / f"{study.get('name', 'agent-harness-study')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "study_name": str(study.get("name", "agent-harness-study")),
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "runs": [],
    }

    for variant_name, harness_config in variants:
        for seed in seeds:
            flashrl: FlashRL | None = None
            run_config = RunConfig.from_yaml(args.config)
            run_config.controller.seed = int(seed)
            rollout_agent = build_coding_agent(harness_config)
            try:
                flashrl = FlashRL(
                    run_config=run_config,
                    rollout_fn=rollout_agent,
                    reward_fn=build_coding_reward_fn(),
                )
                flashrl.train(train_dataset)
                eval_summary = evaluate_model(
                    flashrl,
                    rollout_agent,
                    limit=eval_limit,
                )
                run_dir = (
                    str(flashrl._run_logger.run_dir.resolve())
                    if flashrl._run_logger is not None
                    else None
                )
                manifest["runs"].append(
                    {
                        "variant": variant_name,
                        "seed": int(seed),
                        "run_dir": run_dir,
                        "harness": harness_config.__dict__,
                        "evaluation": eval_summary,
                    }
                )
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            finally:
                if flashrl is not None:
                    flashrl.close()

    print(json.dumps({"manifest_path": str(manifest_path.resolve())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
