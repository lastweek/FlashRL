#!/usr/bin/env python3
"""Generate a single-page PDF table using raw equation snippets from paper PDFs."""

import argparse
from dataclasses import dataclass
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from PIL import Image, ImageStat
except ImportError as exc:  # pragma: no cover - exercised by direct script usage.
    print(
        "matplotlib and Pillow are required for scripts/plot_rl_algorithm_equations_table.py. "
        "Install them with `python3 -m pip install matplotlib pillow`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_OUTPUT_PATH: Final[Path] = Path("docs/rl_algorithm_equations_table.pdf")
PAPER_CACHE_DIR: Final[Path] = Path("tmp/paper_pdfs")
PAGE_RENDER_DIR: Final[Path] = Path("tmp/pdfs/paper_pages")
SNIPPET_DIR: Final[Path] = Path("tmp/pdfs/equation_snippets")
PAGE_SIZE: Final[tuple[float, float]] = (17.0, 11.0)
TITLE: Final[str] = "RL Post-Training Equation Table"
SUBTITLE: Final[str] = "Primary displayed RL objectives cropped from the original paper PDFs."
HEADER_BACKGROUND: Final[str] = "#16324f"
HEADER_TEXT: Final[str] = "#ffffff"
TABLE_BORDER: Final[str] = "#7f8c99"
ROW_BACKGROUND_A: Final[str] = "#fbfcfe"
ROW_BACKGROUND_B: Final[str] = "#f3f6fa"
TEXT_COLOR: Final[str] = "#17212b"
MUTED_TEXT: Final[str] = "#4a5563"
MODEL_TEXT_COLOR: Final[str] = "#102030"
MODEL_COLUMN_RATIO: Final[float] = 0.18
HEADER_HEIGHT: Final[float] = 0.062
RENDER_LONG_SIDE: Final[int] = 2600


@dataclass(frozen=True)
class CropPadding:
    """Extra padding around the searched equation anchors in PDF points."""

    left: float
    right: float
    bottom: float
    top: float


@dataclass(frozen=True)
class PaperEquationSpec:
    """Metadata for one model's primary paper equation."""

    model: str
    paper_title: str
    arxiv_id: str
    pdf_url: str
    page_index: int
    equation_label: str
    search_tokens: tuple[str, ...]
    padding: CropPadding
    crop_rect: tuple[float, float, float, float] | None = None

    @property
    def pdf_path(self) -> Path:
        """Return the local cache path for the paper PDF."""
        return PAPER_CACHE_DIR / f"{self.arxiv_id}.pdf"


@dataclass(frozen=True)
class RenderedSnippet:
    """A rendered snippet ready for PDF composition."""

    model: str
    caption: str
    image_path: Path


PAPER_SPECS: Final[tuple[PaperEquationSpec, ...]] = (
    PaperEquationSpec(
        model="DeepSeek-V3.2",
        paper_title="DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models",
        arxiv_id="2512.02556",
        pdf_url="https://arxiv.org/pdf/2512.02556.pdf",
        page_index=6,
        equation_label="Published GRPO objective (Eq. 5)",
        search_tokens=("JGRPO", "clip", "DKL"),
        padding=CropPadding(left=20.0, right=160.0, bottom=20.0, top=22.0),
        crop_rect=(78.0, 610.0, 545.0, 690.0),
    ),
    PaperEquationSpec(
        model="GLM-5",
        paper_title="GLM-5: from Vibe Coding to Agentic Engineering",
        arxiv_id="2602.15763",
        pdf_url="https://arxiv.org/pdf/2602.15763.pdf",
        page_index=10,
        equation_label="Reasoning RL backbone (Eq. 1)",
        search_tokens=("L(θ)", "pop", "ri,t", "clip"),
        padding=CropPadding(left=24.0, right=132.0, bottom=18.0, top=20.0),
        crop_rect=(118.0, 176.0, 560.0, 298.0),
    ),
    PaperEquationSpec(
        model="MiMo-V2-Flash",
        paper_title="MiMo-V2-Flash Technical Report",
        arxiv_id="2601.02780",
        pdf_url="https://arxiv.org/pdf/2601.02780.pdf",
        page_index=17,
        equation_label="MOPD surrogate loss (Eq. 7)",
        search_tokens=("LMOPD",),
        padding=CropPadding(left=24.0, right=330.0, bottom=22.0, top=26.0),
        crop_rect=(120.0, 440.0, 520.0, 502.0),
    ),
    PaperEquationSpec(
        model="Kimi K2.5",
        paper_title="Kimi K2.5: Visual Agentic Intelligence",
        arxiv_id="2602.02276",
        pdf_url="https://arxiv.org/pdf/2602.02276.pdf",
        page_index=7,
        equation_label="Core RL objective (Eq. 1)",
        search_tokens=("LRL", "Clip", "r(x, yj)", "τ"),
        padding=CropPadding(left=24.0, right=136.0, bottom=20.0, top=24.0),
        crop_rect=(94.0, 396.0, 544.0, 440.0),
    ),
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single-page PDF table of primary RL equations cropped from the source papers. "
            f"Saved by default to {DEFAULT_OUTPUT_PATH}."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional PDF output path. If omitted, the artifact is saved to "
            f"{DEFAULT_OUTPUT_PATH}."
        ),
    )
    return parser


def resolve_output_path(output: Path | None) -> Path:
    """Resolve and normalize the final output path."""
    if output is None:
        return DEFAULT_OUTPUT_PATH
    if output.suffix == "":
        return output.with_suffix(".pdf")
    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must end with .pdf or omit the suffix for automatic .pdf.")
    return output


def configure_style() -> None:
    """Apply a stable visual style for PDF output."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ensure_platform_support() -> None:
    """Fail early if the native macOS PDF stack is unavailable."""
    if sys.platform != "darwin":
        raise RuntimeError("This script currently requires macOS because it uses PDFKit via osascript.")
    if shutil.which("osascript") is None:
        raise RuntimeError("Could not find `osascript`, which is required for native PDF rendering.")


def run_jxa(script: str) -> str:
    """Execute a JXA script and return stdout."""
    result = subprocess.run(
        ["osascript", "-l", "JavaScript"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(stderr or stdout or "JXA execution failed.")
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return stdout or stderr


def download_pdf(spec: PaperEquationSpec) -> Path:
    """Fetch the paper PDF if it is not already cached locally."""
    PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = spec.pdf_path
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    script = f"""
ObjC.import('Foundation')
var outPath = {json.dumps(str(pdf_path.resolve()))}
var data = $.NSData.dataWithContentsOfURL($.NSURL.URLWithString({json.dumps(spec.pdf_url)}))
if (!data) {{
  throw new Error('Failed to fetch {spec.arxiv_id}')
}}
data.writeToFileAtomically(outPath, true)
console.log('ok')
"""
    run_jxa(script)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(
            f"Could not fetch {spec.arxiv_id}. Expected cached PDF at {pdf_path}."
        )
    return pdf_path


def render_pdf_page(pdf_path: Path, page_index: int, output_path: Path) -> tuple[float, float]:
    """Render one PDF page to a PNG using the native macOS PDF stack."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
ObjC.import('Foundation')
ObjC.import('Quartz')
ObjC.import('AppKit')
var pdfPath = {json.dumps(str(pdf_path.resolve()))}
var outPath = {json.dumps(str(output_path.resolve()))}
var doc = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(pdfPath))
if (!doc) {{
  throw new Error('Failed to open ' + pdfPath)
}}
var page = doc.pageAtIndex({page_index})
if (!page) {{
  throw new Error('Failed to open page {page_index} for ' + pdfPath)
}}
var bounds = page.boundsForBox($.kPDFDisplayBoxMediaBox)
var width = Number(bounds.size.width)
var height = Number(bounds.size.height)
var scale = {RENDER_LONG_SIDE} / Math.max(width, height)
var image = page.thumbnailOfSizeForBox($.NSMakeSize(width * scale, height * scale), $.kPDFDisplayBoxMediaBox)
var tiff = image.TIFFRepresentation
var rep = $.NSBitmapImageRep.imageRepWithData(tiff)
var pngData = rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $({{}}))
pngData.writeToFileAtomically(outPath, true)
console.log(JSON.stringify({{width: width, height: height}}))
"""
    rendered = json.loads(run_jxa(script).splitlines()[-1])
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to render page {page_index} from {pdf_path}.")
    return float(rendered["width"]), float(rendered["height"])


def find_token_rects(spec: PaperEquationSpec) -> list[tuple[float, float, float, float]]:
    """Locate searchable token bounds on the target PDF page."""
    script = f"""
ObjC.import('Foundation')
ObjC.import('Quartz')
var doc = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath({json.dumps(str(spec.pdf_path.resolve()))}))
if (!doc) {{
  throw new Error('Failed to open ' + {json.dumps(str(spec.pdf_path.resolve()))})
}}
var expectedPage = {spec.page_index}
var tokens = {json.dumps(list(spec.search_tokens))}
var rects = []
for (var i = 0; i < tokens.length; i++) {{
  var matches = doc.findStringWithOptions(tokens[i], 0)
  for (var j = 0; j < matches.count; j++) {{
    var sel = matches.objectAtIndex(j)
    var page = sel.pages.objectAtIndex(0)
    var pageIndex = Number(doc.indexForPage(page))
    if (pageIndex === expectedPage) {{
      var bounds = sel.boundsForPage(page)
      rects.push({{
        x0: Number(bounds.origin.x),
        y0: Number(bounds.origin.y),
        x1: Number(bounds.origin.x + bounds.size.width),
        y1: Number(bounds.origin.y + bounds.size.height)
      }})
    }}
  }}
}}
console.log(JSON.stringify(rects))
"""
    rects = json.loads(run_jxa(script).splitlines()[-1])
    if not rects:
        raise RuntimeError(
            f"Could not locate equation anchors for {spec.model} on page {spec.page_index + 1} "
            f"of {spec.pdf_path}."
        )
    return [(float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])) for r in rects]


def compute_crop_rect(
    rects: list[tuple[float, float, float, float]],
    *,
    page_width: float,
    page_height: float,
    padding: CropPadding,
) -> tuple[float, float, float, float]:
    """Expand the anchor bounds into the final crop rectangle."""
    x0 = min(rect[0] for rect in rects) - padding.left
    y0 = min(rect[1] for rect in rects) - padding.bottom
    x1 = max(rect[2] for rect in rects) + padding.right
    y1 = max(rect[3] for rect in rects) + padding.top
    return (
        max(0.0, x0),
        max(0.0, y0),
        min(page_width, x1),
        min(page_height, y1),
    )


def crop_page_image(
    page_image_path: Path,
    crop_rect: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    output_path: Path,
) -> Path:
    """Crop the rendered page image using PDF-point coordinates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(page_image_path).convert("RGBA")
    x0, y0, x1, y1 = crop_rect
    left = int(round(x0 * image.width / page_width))
    right = int(round(x1 * image.width / page_width))
    top = int(round(image.height - (y1 * image.height / page_height)))
    bottom = int(round(image.height - (y0 * image.height / page_height)))
    if left >= right or top >= bottom:
        raise RuntimeError(f"Computed an invalid crop rectangle for {page_image_path}.")
    snippet = image.crop((left, top, right, bottom))
    snippet.save(output_path)
    validate_snippet(output_path)
    return output_path


def validate_snippet(snippet_path: Path) -> None:
    """Fail fast if the cropped snippet is obviously empty."""
    image = Image.open(snippet_path).convert("L")
    if image.width < 60 or image.height < 30:
        raise RuntimeError(f"Cropped snippet is unexpectedly small: {snippet_path}.")
    stat = ImageStat.Stat(image)
    if stat.mean[0] > 250:
        raise RuntimeError(f"Cropped snippet appears blank: {snippet_path}.")


def prepare_rendered_snippets() -> list[RenderedSnippet]:
    """Fetch, render, crop, and return all paper equation snippets."""
    ensure_platform_support()
    rendered: list[RenderedSnippet] = []
    for spec in PAPER_SPECS:
        pdf_path = download_pdf(spec)
        page_png = PAGE_RENDER_DIR / f"{spec.arxiv_id}_page_{spec.page_index + 1}.png"
        page_width, page_height = render_pdf_page(pdf_path, spec.page_index, page_png)
        if spec.crop_rect is None:
            token_rects = find_token_rects(spec)
            crop_rect = compute_crop_rect(
                token_rects,
                page_width=page_width,
                page_height=page_height,
                padding=spec.padding,
            )
        else:
            crop_rect = spec.crop_rect
        snippet_path = SNIPPET_DIR / f"{spec.arxiv_id}_eq.png"
        crop_page_image(
            page_png,
            crop_rect,
            page_width=page_width,
            page_height=page_height,
            output_path=snippet_path,
        )
        rendered.append(
            RenderedSnippet(
                model=spec.model,
                caption=f"{spec.equation_label} | arXiv:{spec.arxiv_id}",
                image_path=snippet_path,
            )
        )
    return rendered


def add_page_chrome(fig: Figure) -> None:
    """Add title and subtitle."""
    fig.suptitle(TITLE, fontsize=22, fontweight="bold", y=0.972, color=MODEL_TEXT_COLOR)
    fig.text(
        0.5,
        0.934,
        SUBTITLE,
        ha="center",
        va="center",
        fontsize=10.8,
        color=MUTED_TEXT,
    )


def add_snippet_image(
    fig: Figure,
    *,
    image_path: Path,
    x_left: float,
    y_bottom: float,
    width: float,
    height: float,
) -> None:
    """Place one snippet image into a fitted axes box."""
    image = Image.open(image_path).convert("RGBA")
    aspect = image.width / image.height
    box_aspect = width / height
    if aspect >= box_aspect:
        image_width = width
        image_height = width / aspect
    else:
        image_height = height
        image_width = height * aspect
    image_left = x_left + (width - image_width) / 2.0
    image_bottom = y_bottom + (height - image_height) / 2.0
    image_ax = fig.add_axes([image_left, image_bottom, image_width, image_height])
    image_ax.imshow(image)
    image_ax.axis("off")


def build_figure(rendered_snippets: list[RenderedSnippet]) -> Figure:
    """Build the single-page equation comparison figure."""
    if len(rendered_snippets) != len(PAPER_SPECS):
        raise ValueError("Expected one rendered snippet per paper spec.")

    configure_style()
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    add_page_chrome(fig)

    table_left = 0.04
    table_right = 0.98
    table_top = 0.875
    table_bottom = 0.065
    table_width = table_right - table_left
    table_height = table_top - table_bottom
    model_width = table_width * MODEL_COLUMN_RATIO
    equation_width = table_width - model_width

    ax.add_patch(
        Rectangle(
            (table_left, table_top - HEADER_HEIGHT),
            table_width,
            HEADER_HEIGHT,
            facecolor=HEADER_BACKGROUND,
            edgecolor=TABLE_BORDER,
            linewidth=1.1,
        )
    )
    ax.plot(
        [table_left + model_width, table_left + model_width],
        [table_top - HEADER_HEIGHT, table_top],
        color="#d7dee5",
        linewidth=1.1,
    )
    ax.text(
        table_left + model_width / 2.0,
        table_top - HEADER_HEIGHT / 2.0,
        "Model",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=HEADER_TEXT,
    )
    ax.text(
        table_left + model_width + equation_width / 2.0,
        table_top - HEADER_HEIGHT / 2.0,
        "Primary Paper Equation",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=HEADER_TEXT,
    )

    body_height = table_height - HEADER_HEIGHT
    row_height = body_height / len(rendered_snippets)
    for index, snippet in enumerate(rendered_snippets):
        row_top = table_top - HEADER_HEIGHT - (index * row_height)
        row_bottom = row_top - row_height
        equation_left = table_left + model_width
        background = ROW_BACKGROUND_A if index % 2 == 0 else ROW_BACKGROUND_B
        ax.add_patch(
            Rectangle(
                (table_left, row_bottom),
                table_width,
                row_height,
                facecolor=background,
                edgecolor=TABLE_BORDER,
                linewidth=1.0,
            )
        )
        ax.plot(
            [equation_left, equation_left],
            [row_bottom, row_top],
            color=TABLE_BORDER,
            linewidth=1.0,
        )
        ax.text(
            table_left + 0.018,
            row_top - 0.024,
            snippet.model,
            ha="left",
            va="top",
            fontsize=13.0,
            fontweight="bold",
            color=MODEL_TEXT_COLOR,
        )
        ax.text(
            equation_left + 0.018,
            row_top - 0.024,
            snippet.caption,
            ha="left",
            va="top",
            fontsize=9.2,
            color=MUTED_TEXT,
        )

        image_left = equation_left + 0.018
        image_width = equation_width - 0.036
        image_bottom = row_bottom + 0.018
        image_height = row_height - 0.072
        add_snippet_image(
            fig,
            image_path=snippet.image_path,
            x_left=image_left,
            y_bottom=image_bottom,
            width=image_width,
            height=image_height,
        )

    return fig


def build_figures(rendered_snippets: list[RenderedSnippet]) -> list[Figure]:
    """Return the composed figure inside a list."""
    return [build_figure(rendered_snippets)]


def save_pdf(output_path: Path, rendered_snippets: list[RenderedSnippet]) -> Path:
    """Render the equation table and write it to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(rendered_snippets)
    try:
        figure.savefig(output_path, format="pdf")
    finally:
        plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_argument_parser().parse_args(argv)
    output_path = resolve_output_path(args.output)
    rendered_snippets = prepare_rendered_snippets()
    saved_path = save_pdf(output_path, rendered_snippets)
    print(f"Saved PDF to {saved_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct script usage.
    raise SystemExit(main())
