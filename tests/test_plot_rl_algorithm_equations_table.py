"""Tests for the paper-snippet RL equation table PDF."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_rl_algorithm_equations_table.py"


def load_script_module():
    """Load the plotting script as a module for direct test access."""
    spec = importlib.util.spec_from_file_location("plot_rl_algorithm_equations_table", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_dummy_snippets(tmp_path: Path, module):
    """Create deterministic local snippet images for offline composition tests."""
    snippets = []
    for index, spec in enumerate(module.PAPER_SPECS, start=1):
        image_path = tmp_path / f"snippet_{index}.png"
        image = Image.new("RGB", (900, 240), color="white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 890, 230), outline="black", width=3)
        draw.text((35, 92), f"{spec.model} raw equation snippet", fill="black")
        image.save(image_path)
        snippets.append(
            module.RenderedSnippet(
                model=spec.model,
                caption=spec.equation_label,
                image_path=image_path,
            )
        )
    return snippets


def test_build_figure_returns_one_titled_page(tmp_path: Path) -> None:
    """The table generator should expose a single titled figure."""
    module = load_script_module()
    figure = module.build_figure(make_dummy_snippets(tmp_path, module))

    try:
        assert figure._suptitle is not None
        assert figure._suptitle.get_text().strip()
    finally:
        plt.close(figure)


def test_paper_specs_keep_expected_model_order_and_metadata() -> None:
    """The table should preserve model order and paper source metadata."""
    module = load_script_module()
    models = [spec.model for spec in module.PAPER_SPECS]

    assert models == [
        "DeepSeek-V3.2",
        "GLM-5",
        "MiMo-V2-Flash",
        "Kimi K2.5",
    ]
    for spec in module.PAPER_SPECS:
        assert spec.arxiv_id
        assert spec.pdf_url.startswith("https://arxiv.org/pdf/")
        assert spec.page_index >= 0
        assert spec.search_tokens
        assert spec.crop_rect is not None


def test_main_generates_nonempty_pdf_with_mocked_snippets(tmp_path: Path, monkeypatch) -> None:
    """CLI execution should generate a non-empty PDF without network access."""
    module = load_script_module()
    output_path = tmp_path / "rl_algorithm_equations_table.pdf"
    dummy_snippets = make_dummy_snippets(tmp_path, module)
    monkeypatch.setattr(module, "prepare_rendered_snippets", lambda: dummy_snippets)

    exit_code = module.main(["--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.integration
def test_live_cli_generates_nonempty_pdf(tmp_path: Path) -> None:
    """Opt-in end-to-end smoke test for live paper fetching and cropping."""
    if shutil.which("osascript") is None:
        pytest.skip("osascript is required for the live paper-cropping path.")

    module = load_script_module()
    output_path = tmp_path / "rl_algorithm_equations_table_live.pdf"
    exit_code = module.main(["--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0
