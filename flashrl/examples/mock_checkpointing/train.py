"""Offline managed-checkpointing example."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_run_dir(log_dir: Path, checkpoint_dir: Path, resume_from: str | None) -> tuple[Path, dict[str, Any] | None]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.json"
    restored_state: dict[str, Any] | None = None
    if resume_from == "latest" and latest_path.exists():
        restored_state = json.loads(latest_path.read_text(encoding="utf-8"))
        run_dir = Path(restored_state["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, restored_state

    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-mock_checkpointing"
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, restored_state


def _append_event(events_path: Path, event: str, payload: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": event,
                    "timestamp": _utc_now_iso(),
                    "payload": payload,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )


def _write_latest_manifest(checkpoint_dir: Path, *, checkpoint_path: Path, run_dir: Path) -> None:
    (checkpoint_dir / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "run_dir": str(run_dir),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _save_checkpoint(
    checkpoint_dir: Path,
    run_dir: Path,
    step: int,
    *,
    trigger: str,
) -> Path:
    checkpoint_path = checkpoint_dir / f"step-{step:08d}.pt"
    torch.save({"step": step, "saved_at": _utc_now_iso()}, checkpoint_path)
    _write_latest_manifest(checkpoint_dir, checkpoint_path=checkpoint_path, run_dir=run_dir)
    _append_event(
        run_dir / "events.jsonl",
        "checkpoint",
        {
            "action": "save",
            "trigger": trigger,
            "step": step,
            "path": str(checkpoint_path),
        },
    )
    return checkpoint_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the mock_checkpointing CLI parser."""
    parser = argparse.ArgumentParser(description="Run the offline mock checkpointing example.")
    parser.add_argument("--log-dir", required=True, help="Directory for run logs.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory for managed checkpoints.")
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=2,
        help="Interval checkpoint frequency in steps.",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=4,
        help="Number of fake prompts / steps to execute.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume source. Use 'latest' to continue from checkpoint-dir/latest.json.",
    )
    parser.add_argument(
        "--save-on-run-end",
        action="store_true",
        help="Save a final checkpoint as final.pt after the last step.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a tiny offline loop that exercises managed checkpoint files."""
    args = build_argument_parser().parse_args(argv)
    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_dir, restored_state = _ensure_run_dir(log_dir, checkpoint_dir, args.resume_from)
    events_path = run_dir / "events.jsonl"

    start_step = 0
    if restored_state is not None:
        checkpoint_path = Path(restored_state["checkpoint_path"])
        state = torch.load(checkpoint_path, weights_only=False)
        start_step = int(state.get("step", 0))
        _append_event(
            events_path,
            "run_resumed",
            {"checkpoint_path": str(checkpoint_path), "step": start_step},
        )
        _append_event(
            events_path,
            "checkpoint",
            {
                "action": "load",
                "trigger": "resume",
                "step": start_step,
                "path": str(checkpoint_path),
            },
        )

    for step in range(start_step + 1, start_step + int(args.prompt_count) + 1):
        _append_event(events_path, "step_done", {"step": step})
        if args.save_every_steps and step % int(args.save_every_steps) == 0:
            _save_checkpoint(checkpoint_dir, run_dir, step, trigger="interval")

    if args.save_on_run_end:
        final_step = start_step + int(args.prompt_count)
        final_path = checkpoint_dir / "final.pt"
        torch.save({"step": final_step, "saved_at": _utc_now_iso()}, final_path)
        _write_latest_manifest(checkpoint_dir, checkpoint_path=final_path, run_dir=run_dir)
        _append_event(
            events_path,
            "checkpoint",
            {
                "action": "save",
                "trigger": "final",
                "step": final_step,
                "path": str(final_path),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

