#!/usr/bin/env python3
"""Generate a one-page PDF explaining logprob and probability metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised by direct script usage.
    print(
        "matplotlib is required for scripts/plot_logprob_probability.py. "
        "Install it with `python3 -m pip install matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_OUTPUT_PATH = Path("scripts/logprob_probability.pdf")
PAGE_SIZE = (16.0, 9.0)
POLICY_COLOR = "#1f77b4"
REWARD_COLOR = "#d95f02"
BASELINE_COLOR = "#2ca02c"
REFERENCE_COLOR = "#222222"
POLICY_LIGHT = "#9ecae1"
REWARD_LIGHT = "#fdd0a2"
BASELINE_LIGHT = "#c7e9c0"


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a one-page PDF explaining how logprob maps to probability "
            f"and rollout confidence metrics. Saved by default to {DEFAULT_OUTPUT_PATH}."
        )
    )
    parser.add_argument(
        "--points",
        type=int,
        default=600,
        help="Number of sample points used for smooth curves.",
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
    if args.points < 200:
        raise ValueError("--points must be at least 200 for stable curves.")


def resolve_output_path(output: Path | None) -> tuple[Path, bool]:
    """Resolve the final PDF path and whether to show the figure interactively."""
    if output is None:
        return DEFAULT_OUTPUT_PATH, True

    if output.suffix == "":
        return output.with_suffix(".pdf"), False

    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must end with .pdf or omit the suffix for automatic .pdf.")

    return output, False


def configure_style() -> None:
    """Apply shared plot styling."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_page_chrome(fig: plt.Figure, title: str, subtitle: str, takeaway: str) -> None:
    """Add shared title, subtitle, and takeaway styling."""
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="center",
        fontsize=11.25,
        color="#444444",
    )
    fig.text(
        0.5,
        0.072,
        takeaway,
        ha="center",
        va="center",
        fontsize=10.7,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f7f7f7",
            "alpha": 0.98,
            "edgecolor": "#cccccc",
        },
    )


def format_axis(ax) -> None:
    """Apply common axis formatting."""
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)


def make_figure(points: int) -> plt.Figure:
    """Create the logprob versus probability explainer figure."""
    configure_style()

    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.80,
        bottom=0.18,
        wspace=0.28,
        width_ratios=[1.05, 1.0, 1.08],
    )
    ax_mapping = fig.add_subplot(grid[0, 0])
    ax_length = fig.add_subplot(grid[0, 1])
    ax_normalized = fig.add_subplot(grid[0, 2])

    logprob_grid = np.linspace(-10.0, 0.0, points)
    probability_grid = np.exp(logprob_grid)

    ax_mapping.plot(probability_grid, logprob_grid, color=POLICY_COLOR, linewidth=2.6)
    ax_mapping.fill_betweenx(logprob_grid, 0.0, probability_grid, color=POLICY_LIGHT, alpha=0.35)
    anchor_values = np.array([0.0, -1.0, -3.0, -5.0])
    anchor_probabilities = np.exp(anchor_values)
    ax_mapping.scatter(anchor_probabilities, anchor_values, color=REWARD_COLOR, s=44, zorder=3)
    for value, probability in zip(anchor_values, anchor_probabilities, strict=True):
        ax_mapping.annotate(
            f"log p = {value:.0f}\np = {probability:.3f}",
            xy=(probability, value),
            xytext=(probability + 0.08, value + (0.55 if value < -1.0 else -0.85)),
            fontsize=9.4,
            color=REWARD_COLOR,
            arrowprops={"arrowstyle": "->", "color": REWARD_COLOR, "lw": 1.2},
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.92, "edgecolor": REWARD_COLOR},
        )
    ax_mapping.set_title("1. logprob -> probability")
    ax_mapping.set_xlabel("probability")
    ax_mapping.set_ylabel("log probability")
    ax_mapping.set_xlim(0.0, 1.05)
    ax_mapping.set_ylim(-10.0, 0.5)
    format_axis(ax_mapping)
    ax_mapping.text(
        0.04,
        0.08,
        r"$\log p = \log(p)$" "\n" r"Equal steps in log space become multiplicative changes in probability.",
        transform=ax_mapping.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.1,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    response_lengths = np.array([8, 16, 32, 64], dtype=np.int64)
    avg_log_prob = -0.45
    total_log_probs = response_lengths * avg_log_prob
    total_probabilities = np.exp(total_log_probs)

    bar_positions = np.arange(response_lengths.size)
    ax_length.bar(
        bar_positions,
        total_log_probs,
        color=REWARD_LIGHT,
        edgecolor=REWARD_COLOR,
        linewidth=1.0,
    )
    ax_length.axhline(0.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_length.set_title("2. total log_prob is length-dominated")
    ax_length.set_xlabel("Toy response length")
    ax_length.set_ylabel("Sequence log_prob")
    ax_length.set_xticks(bar_positions, [str(length) for length in response_lengths])
    format_axis(ax_length)
    for index, (log_prob, total_probability) in enumerate(zip(total_log_probs, total_probabilities, strict=True)):
        ax_length.text(
            index,
            log_prob - 0.55,
            f"log_prob={log_prob:.1f}\nseq prob={total_probability:.2e}",
            ha="center",
            va="top",
            fontsize=8.8,
            color="#333333",
        )
    ax_length.text(
        0.04,
        0.90,
        "All four responses share the same\navg_log_prob_per_token = -0.45.\nOnly the length changes.",
        transform=ax_length.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )
    ax_length.annotate(
        "Raw sequence log_prob gets more negative\njust because there are more tokens.",
        xy=(3, total_log_probs[-1]),
        xytext=(0.35, -13.5),
        fontsize=9.7,
        color=REWARD_COLOR,
        arrowprops={"arrowstyle": "->", "color": REWARD_COLOR, "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": REWARD_COLOR},
    )

    candidate_names = ["A", "B", "C", "D"]
    avg_log_prob_per_token = np.array([-0.20, -0.45, -0.90, -1.60], dtype=np.float64)
    avg_token_prob = np.exp(avg_log_prob_per_token)
    candidate_positions = np.arange(len(candidate_names))
    width = 0.38

    ax_normalized.bar(
        candidate_positions - width / 2,
        avg_log_prob_per_token,
        width=width,
        color=POLICY_LIGHT,
        edgecolor=POLICY_COLOR,
        linewidth=1.0,
        label="avg_log_prob_per_token",
    )
    ax_normalized_right = ax_normalized.twinx()
    ax_normalized_right.bar(
        candidate_positions + width / 2,
        avg_token_prob,
        width=width,
        color=BASELINE_LIGHT,
        edgecolor=BASELINE_COLOR,
        linewidth=1.0,
        label="avg_token_prob",
    )

    ax_normalized.axhline(0.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_normalized.set_title("3. avg_log_prob_per_token <-> avg_token_prob")
    ax_normalized.set_xlabel("Toy candidate")
    ax_normalized.set_ylabel("Average logprob / token", color=POLICY_COLOR)
    ax_normalized_right.set_ylabel("Average token probability", color=BASELINE_COLOR)
    ax_normalized.tick_params(axis="y", labelcolor=POLICY_COLOR)
    ax_normalized_right.tick_params(axis="y", labelcolor=BASELINE_COLOR)
    ax_normalized.set_xticks(candidate_positions, candidate_names)
    ax_normalized.set_ylim(min(avg_log_prob_per_token) - 0.25, 0.05)
    ax_normalized_right.set_ylim(0.0, 1.05)
    format_axis(ax_normalized)
    for index, (log_prob, token_prob) in enumerate(zip(avg_log_prob_per_token, avg_token_prob, strict=True)):
        ax_normalized.text(
            index - width / 2,
            log_prob - 0.07,
            f"{log_prob:.2f}",
            ha="center",
            va="top",
            fontsize=8.9,
            color=POLICY_COLOR,
        )
        ax_normalized_right.text(
            index + width / 2,
            token_prob + 0.03,
            f"{token_prob:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.9,
            color=BASELINE_COLOR,
        )
    ax_normalized.text(
        0.04,
        0.88,
        "Same ranking, different units:\n"
        "log space is additive;\n"
        "probability space is easier to read.",
        transform=ax_normalized.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )
    left_handles, left_labels = ax_normalized.get_legend_handles_labels()
    right_handles, right_labels = ax_normalized_right.get_legend_handles_labels()
    ax_normalized.legend(left_handles + right_handles, left_labels + right_labels, loc="lower left", framealpha=0.95)

    add_page_chrome(
        fig,
        "Logprob and Probability",
        "The same rollout confidence can be read in log space or probability space, but sequence totals and per-token averages answer different questions.",
        "Takeaway: use raw log_prob for within-sequence accounting, but compare candidates across lengths with avg_log_prob_per_token or its readable probability-space form avg_token_prob = exp(avg_log_prob_per_token).",
    )
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

    figure = make_figure(args.points)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {resolved_output}")

    if should_show and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
