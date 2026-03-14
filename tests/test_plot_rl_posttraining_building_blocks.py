"""Smoke tests for the RL post-training visualization deck."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_rl_posttraining_building_blocks.py"


def load_script_module():
    """Load the plotting script as a module for direct test access."""
    spec = importlib.util.spec_from_file_location("plot_rl_posttraining_building_blocks", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_figures_returns_six_titled_pages() -> None:
    """The deck should expose six figures with non-empty page titles."""
    module = load_script_module()
    figures = module.build_figures(points=240)

    try:
        assert len(figures) == 6
        for figure in figures:
            assert figure._suptitle is not None
            assert figure._suptitle.get_text().strip()
    finally:
        for figure in figures:
            plt.close(figure)


def test_cli_generates_nonempty_pdf(tmp_path: Path) -> None:
    """Running the script via CLI should create a non-empty PDF artifact."""
    output_path = tmp_path / "rl_posttraining_building_blocks.pdf"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--points", "240", "--output", str(output_path)],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
