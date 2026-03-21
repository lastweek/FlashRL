"""Aggregate an agent harness ablation-study manifest into comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashrl.examples.agent_harness_ablation.study import (
    DEFAULT_MANIFEST_GLOB,
    find_latest_manifest,
    summarize_manifest,
    write_reports,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate agent harness ablation reports.")
    parser.add_argument("--manifest", default=None, help="Path to one study manifest. Defaults to the newest manifest under logs/studies/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest) if args.manifest else find_latest_manifest()
    if manifest_path is None:
        raise FileNotFoundError(f"No manifest found matching {DEFAULT_MANIFEST_GLOB!r}.")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest mapping at {manifest_path}.")
    summary = summarize_manifest(manifest)
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


if __name__ == "__main__":
    raise SystemExit(main())
