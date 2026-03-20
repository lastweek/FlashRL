"""Tests for the offline mock checkpointing example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def load_script_module(module_name: str, relative_path: str):
    """Load one hyphen-folder script as a normal Python module for tests."""
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mock_checkpointing = load_script_module(
    "flashrl_mock_checkpointing_train",
    "flashrl/examples/mock_checkpointing/train.py",
)


def test_mock_checkpointing_cli_help_exposes_managed_checkpoint_flags() -> None:
    """The mock example should show the intended managed checkpoint controls."""
    help_text = mock_checkpointing.build_argument_parser().format_help()
    assert "--log-dir" in help_text
    assert "--checkpoint-dir" in help_text
    assert "--save-every-steps" in help_text
    assert "--resume-from" in help_text
    assert "--save-on-run-end" in help_text


def test_mock_checkpointing_example_creates_interval_and_final_checkpoints(
    tmp_path: Path,
) -> None:
    """The offline example should create managed interval and final checkpoints."""
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    exit_code = mock_checkpointing.main(
        [
            "--log-dir",
            str(log_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--save-every-steps",
            "2",
            "--prompt-count",
            "4",
            "--save-on-run-end",
        ]
    )

    assert exit_code == 0
    assert (checkpoint_dir / "step-00000002.pt").exists()
    assert (checkpoint_dir / "step-00000004.pt").exists()
    assert (checkpoint_dir / "final.pt").exists()
    latest_manifest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_manifest["checkpoint_path"] == str(checkpoint_dir / "final.pt")


def test_mock_checkpointing_example_resumes_from_latest_in_same_run_dir(
    tmp_path: Path,
) -> None:
    """Resuming from latest should append to the same run directory."""
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    first_exit = mock_checkpointing.main(
        [
            "--log-dir",
            str(log_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--save-every-steps",
            "2",
            "--prompt-count",
            "4",
        ]
    )
    assert first_exit == 0

    second_exit = mock_checkpointing.main(
        [
            "--log-dir",
            str(log_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--save-every-steps",
            "2",
            "--prompt-count",
            "4",
            "--resume-from",
            "latest",
        ]
    )
    assert second_exit == 0

    run_dirs = [path for path in log_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    events = [
        json.loads(line)
        for line in (run_dirs[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["event"] == "run_resumed" for event in events)
    assert any(
        event["event"] == "checkpoint"
        and event["payload"]["action"] == "load"
        and event["payload"]["trigger"] == "resume"
        for event in events
    )

    latest_manifest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_manifest["checkpoint_path"] == str(checkpoint_dir / "step-00000008.pt")
