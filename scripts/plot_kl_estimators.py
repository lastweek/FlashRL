#!/usr/bin/env python3
"""Plot Schulman's KL estimators as functions of the likelihood ratio.

Here p and q are distributions or densities, so r(x) = p(x) / q(x) is a
likelihood ratio for the same outcome x, not a scalar probability in [0, 1].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised by direct script usage.
    print(
        "matplotlib is required for scripts/plot_kl_estimators.py. "
        "Install it with `python3 -m pip install matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_OUTPUT_PATH = Path("kl_estimators.pdf")


def k1(ratio: np.ndarray) -> np.ndarray:
    """Compute k1(r) = -log(r)."""
    return -np.log(ratio)


def k2(ratio: np.ndarray) -> np.ndarray:
    """Compute k2(r) = 0.5 * log(r)^2."""
    log_ratio = np.log(ratio)
    return 0.5 * np.square(log_ratio)


def k3(ratio: np.ndarray) -> np.ndarray:
    """Compute k3(r) = (r - 1) - log(r)."""
    return (ratio - 1.0) - np.log(ratio)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot Schulman's KL estimators versus log r, where "
            "r(x) = p(x) / q(x) is a likelihood ratio, and write a PDF "
            f"artifact by default to {DEFAULT_OUTPUT_PATH}."
        )
    )
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of sample points per curve.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional PDF output path. If omitted, the figure is saved to "
            f"{DEFAULT_OUTPUT_PATH} and then shown interactively."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.points < 2:
        raise ValueError("--points must be at least 2.")


def resolve_output_path(output: Path | None) -> tuple[Path, bool]:
    """Resolve the final PDF path and whether to show the figure interactively."""
    if output is None:
        return DEFAULT_OUTPUT_PATH, True

    if output.suffix == "":
        return output.with_suffix(".pdf"), False

    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must end with .pdf or omit the suffix for automatic .pdf.")

    return output, False


def make_figure(points: int) -> plt.Figure:
    """Create the estimator figure."""
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    z = np.linspace(-8.0, 8.0, points)
    ratio = np.exp(z)
    ratio_k1 = k1(ratio)
    ratio_k2 = k2(ratio)
    ratio_k3 = k3(ratio)

    ax.plot(z, ratio_k1, label=r"$k_1(r) = -\log r$", linewidth=2.0)
    ax.plot(z, ratio_k2, label=r"$k_2(r) = \frac{1}{2}(\log r)^2$", linewidth=2.0)
    ax.plot(z, ratio_k3, label=r"$k_3(r) = (r - 1) - \log r$", linewidth=2.0)
    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="0.35", linewidth=1.0, linestyle="--")
    ax.set_title("Schulman's KL Estimators", pad=18)
    ax.set_xlabel(r"$\log r$")
    ax.set_ylabel("Value")
    ax.set_xlim(z[0], z[-1])
    ax.set_yscale("symlog", linthresh=0.25)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.58))

    equation_text = "\n".join(
        [
            r"$r(x) = \frac{p(x)}{q(x)}$",
            r"$k_1(r) = -\log r$",
            r"$k_2(r) = \frac{1}{2}(\log r)^2$",
            r"$k_3(r) = (r - 1) - \log r$",
            r"$\mathrm{KL}(q \parallel p) = E_{x \sim q}[k_1(r(x))] = E_{x \sim q}[k_3(r(x))]$",
            r"$E_{x \sim q}[k_2(r(x))]$ is a local KL approximation",
        ]
    )
    ax.text(
        0.02,
        0.03,
        equation_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )

    fig.suptitle("Likelihood-ratio view: r(x) = p(x) / q(x) for the same sampled outcome x", fontsize=14)
    return fig


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        resolved_output, should_show = resolve_output_path(args.output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    fig = make_figure(points=args.points)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_output, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {resolved_output}")

    if should_show and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
