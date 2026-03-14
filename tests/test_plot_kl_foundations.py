"""Smoke tests for the merged KL foundations plotting scripts."""

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
FOUNDATIONS_SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_kl_foundations.py"
ESTIMATORS_SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_kl_estimators.py"
FORWARD_REVERSE_SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_forward_reverse_kl.py"


def load_script_module(script_path: Path, module_name: str):
    """Load a plotting script as a module."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(script_path: Path, output_path: Path) -> None:
    """Run a plotting CLI under a non-interactive backend."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, str(script_path), "--points", "240", "--output", str(output_path)],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )


def test_build_figures_returns_two_titled_pages() -> None:
    """The merged deck should expose two titled figures in order."""
    module = load_script_module(FOUNDATIONS_SCRIPT_PATH, "plot_kl_foundations")
    figures = module.build_figures(points=240)

    try:
        assert len(figures) == 2
        for figure in figures:
            assert figure._suptitle is not None
            assert figure._suptitle.get_text().strip()
    finally:
        for figure in figures:
            plt.close(figure)


def test_foundations_cli_generates_nonempty_pdf(tmp_path: Path) -> None:
    """The merged deck CLI should create a non-empty multi-page PDF."""
    output_path = tmp_path / "kl_foundations.pdf"
    run_script(FOUNDATIONS_SCRIPT_PATH, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_legacy_estimators_wrapper_generates_nonempty_pdf(tmp_path: Path) -> None:
    """The estimator wrapper should still produce a single-page PDF."""
    output_path = tmp_path / "kl_estimators.pdf"
    run_script(ESTIMATORS_SCRIPT_PATH, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_legacy_forward_reverse_wrapper_generates_nonempty_pdf(tmp_path: Path) -> None:
    """The forward-vs-reverse wrapper should still produce a single-page PDF."""
    output_path = tmp_path / "forward_reverse_kl.pdf"
    run_script(FORWARD_REVERSE_SCRIPT_PATH, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
