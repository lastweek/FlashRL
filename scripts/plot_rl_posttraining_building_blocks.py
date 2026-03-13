#!/usr/bin/env python3
"""Generate a multi-page PDF deck for RL post-training building blocks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - exercised by direct script usage.
    print(
        "matplotlib is required for scripts/plot_rl_posttraining_building_blocks.py. "
        "Install it with `python3 -m pip install matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_OUTPUT_PATH = Path("scripts/rl_posttraining_building_blocks.pdf")
PAGE_SIZE = (16.0, 9.0)
EPSILON = 1e-12
CLIP_RATIO = 0.2
SEED = 7
POLICY_COLOR = "#1f77b4"
POLICY_LIGHT = "#9ecae1"
REFERENCE_COLOR = "#222222"
REWARD_COLOR = "#d95f02"
REWARD_LIGHT = "#fdd0a2"
BASELINE_COLOR = "#2ca02c"
BASELINE_LIGHT = "#c7e9c0"
ACCENT_COLOR = "#7f7f7f"
WARNING_COLOR = "#c23b22"
GUIDE_COLOR = "#bdbdbd"


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return the Gaussian density."""
    z = (x - mean) / std
    return np.exp(-0.5 * np.square(z)) / (std * np.sqrt(2.0 * np.pi))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted)


def normalize_density(x: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Normalize a non-negative density on a grid."""
    area = np.trapezoid(density, x)
    if area <= 0.0:
        raise ValueError("Density area must be positive.")
    return density / area


def kl_divergence(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Return D_KL(p || q) on the provided grid."""
    p_safe = np.clip(p, EPSILON, None)
    q_safe = np.clip(q, EPSILON, None)
    return float(np.trapezoid(p_safe * (np.log(p_safe) - np.log(q_safe)), x))


def configure_style() -> None:
    """Apply a consistent plotting style across all deck pages."""
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a six-page PDF deck covering key RL post-training building blocks, "
            f"saved by default to {DEFAULT_OUTPUT_PATH}."
        )
    )
    parser.add_argument(
        "--points",
        type=int,
        default=500,
        help="Number of sample points used for smooth curves.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional PDF output path. If omitted, the deck is saved to "
            f"{DEFAULT_OUTPUT_PATH} and then shown interactively."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.points < 200:
        raise ValueError("--points must be at least 200 for stable curves.")


def resolve_output_path(output: Path | None) -> tuple[Path, bool]:
    """Resolve the final PDF path and whether to show figures interactively."""
    if output is None:
        return DEFAULT_OUTPUT_PATH, True

    if output.suffix == "":
        return output.with_suffix(".pdf"), False

    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must end with .pdf or omit the suffix for automatic .pdf.")

    return output, False


def add_page_chrome(fig: Figure, title: str, subtitle: str, takeaway: str) -> None:
    """Add the shared page title, subtitle, and takeaway box."""
    fig.suptitle(title, fontsize=21, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="center",
        fontsize=11.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.07,
        takeaway,
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f7f7f7",
            "alpha": 0.98,
            "edgecolor": "#cccccc",
        },
    )


def format_axis(ax, *, grid: bool = True) -> None:
    """Apply common axis formatting."""
    if grid:
        ax.grid(alpha=0.22)
    ax.set_axisbelow(True)


def build_trust_region_figure(points: int) -> Figure:
    """Build page 1: trust regions and on-policy drift."""
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.83,
        bottom=0.18,
        wspace=0.28,
        width_ratios=[1.05, 1.0, 1.2],
    )
    ax_policy = fig.add_subplot(grid[0, 0])
    ax_ratio = fig.add_subplot(grid[0, 1])
    ax_clip = fig.add_subplot(grid[0, 2])

    tokens = np.array(["A", "B", "C", "D", "E", "F"])
    old_policy = np.array([0.30, 0.23, 0.18, 0.13, 0.10, 0.06], dtype=np.float64)
    new_policy = np.array([0.16, 0.17, 0.14, 0.18, 0.20, 0.15], dtype=np.float64)
    ratios = new_policy / old_policy

    bar_positions = np.arange(tokens.size)
    width = 0.36
    ax_policy.bar(
        bar_positions - width / 2,
        old_policy,
        width=width,
        color=POLICY_LIGHT,
        edgecolor=POLICY_COLOR,
        linewidth=1.0,
        label="Old policy",
    )
    ax_policy.bar(
        bar_positions + width / 2,
        new_policy,
        width=width,
        color=POLICY_COLOR,
        alpha=0.85,
        label="Updated policy",
    )
    ax_policy.set_xticks(bar_positions, tokens)
    ax_policy.set_ylabel("Probability")
    ax_policy.set_title("1. Sampled Under One Policy, Updated Under Another")
    ax_policy.set_ylim(0.0, 0.34)
    ax_policy.legend(loc="upper right", framealpha=0.95)
    format_axis(ax_policy)
    ax_policy.text(
        0.04,
        0.93,
        "Reusing a batch only works while the updated policy stays close\n"
        "to the one that produced those samples.",
        transform=ax_policy.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    ax_ratio.axhspan(1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO, color=BASELINE_LIGHT, alpha=0.55)
    ax_ratio.bar(bar_positions, ratios, color=REWARD_LIGHT, edgecolor=REWARD_COLOR, linewidth=1.0)
    ax_ratio.axhline(1.0, color=REFERENCE_COLOR, linewidth=1.2, linestyle="--")
    ax_ratio.axhline(1.0 - CLIP_RATIO, color=BASELINE_COLOR, linewidth=1.0, linestyle=":")
    ax_ratio.axhline(1.0 + CLIP_RATIO, color=BASELINE_COLOR, linewidth=1.0, linestyle=":")
    ax_ratio.set_xticks(bar_positions, tokens)
    ax_ratio.set_ylabel(r"Importance ratio $r = \pi_{new} / \pi_{old}$")
    ax_ratio.set_title("2. Importance Ratios Leave the Safe Band")
    ax_ratio.set_ylim(0.0, max(2.8, ratios.max() * 1.1))
    format_axis(ax_ratio)
    ax_ratio.annotate(
        "Trust region",
        xy=(4.6, 1.15),
        xytext=(2.9, 2.35),
        color=BASELINE_COLOR,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": BASELINE_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": BASELINE_COLOR},
    )
    ax_ratio.annotate(
        "Large ratios say the update is leaning on\nsamples that no longer match the new policy.",
        xy=(5.0, ratios[-1]),
        xytext=(0.25, 2.65),
        color=REWARD_COLOR,
        fontsize=9.8,
        arrowprops={"arrowstyle": "->", "color": REWARD_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": REWARD_COLOR},
    )

    ratio_grid = np.linspace(0.3, 1.9, points)
    unclipped_positive = ratio_grid
    clipped_positive = np.minimum(
        ratio_grid,
        np.clip(ratio_grid, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO),
    )
    unclipped_negative = -ratio_grid
    clipped_negative = np.minimum(
        -ratio_grid,
        -np.clip(ratio_grid, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO),
    )

    ax_clip.axvspan(1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO, color=BASELINE_LIGHT, alpha=0.45)
    ax_clip.plot(ratio_grid, unclipped_positive, color=POLICY_LIGHT, linewidth=2.0, linestyle="--", label="Unclipped, A > 0")
    ax_clip.plot(ratio_grid, clipped_positive, color=POLICY_COLOR, linewidth=2.5, label="Clipped, A > 0")
    ax_clip.plot(ratio_grid, unclipped_negative, color=REWARD_LIGHT, linewidth=2.0, linestyle="--", label="Unclipped, A < 0")
    ax_clip.plot(ratio_grid, clipped_negative, color=REWARD_COLOR, linewidth=2.5, label="Clipped, A < 0")
    ax_clip.axvline(1.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_clip.set_title("3. Clipping Flattens the Incentive to Drift")
    ax_clip.set_xlabel("Importance ratio r")
    ax_clip.set_ylabel("Surrogate objective")
    ax_clip.set_xlim(ratio_grid[0], ratio_grid[-1])
    ax_clip.legend(loc="lower right", framealpha=0.95)
    format_axis(ax_clip)
    ax_clip.text(
        0.04,
        0.06,
        r"$L(r, A) = \min(rA, \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)A)$",
        transform=ax_clip.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.4,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    add_page_chrome(
        fig,
        "1. Trust Regions and On-Policy Drift",
        "Once the updated policy moves too far from the policy that generated the data, importance weighting becomes unstable and clipping starts to matter.",
        "Takeaway: clipping is not cosmetic - it prevents stale samples from dominating the update when the policy drifts away from the data that produced them.",
    )
    return fig


def build_best_of_n_figure(points: int) -> Figure:
    """Build page 2: best-of-N versus policy optimization."""
    del points
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.05,
        right=0.985,
        top=0.83,
        bottom=0.18,
        hspace=0.34,
        wspace=0.24,
    )
    ax_rewards = fig.add_subplot(grid[0, 0])
    ax_best = fig.add_subplot(grid[0, 1])
    ax_relative = fig.add_subplot(grid[1, 0])
    ax_tradeoff = fig.add_subplot(grid[1, 1])

    rewards = np.array([-0.35, 0.10, 0.26, 0.61, -0.04, 0.93, 0.33, 0.57, 1.18, 0.41, 0.70, 0.28])
    candidates = np.arange(1, rewards.size + 1)
    best_index = int(np.argmax(rewards))
    centered = rewards - rewards.mean()
    relative_credit = centered / (centered.std() + 1e-8)

    ax_rewards.scatter(candidates, rewards, s=68, color=POLICY_COLOR, zorder=3)
    ax_rewards.vlines(candidates, 0.0, rewards, color=POLICY_LIGHT, linewidth=3.0, alpha=0.95)
    ax_rewards.axhline(rewards.mean(), color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_rewards.set_title("1. Candidate Rewards for One Prompt")
    ax_rewards.set_xlabel("Sample index")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.set_xlim(0.3, candidates[-1] + 0.7)
    format_axis(ax_rewards)
    ax_rewards.annotate(
        "A larger sample budget finds better completions,\n"
        "but not every method learns from all of them.",
        xy=(9, rewards[best_index]),
        xytext=(1.1, 1.05),
        color=POLICY_COLOR,
        fontsize=9.8,
        arrowprops={"arrowstyle": "->", "color": POLICY_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": POLICY_COLOR},
    )

    best_colors = [GUIDE_COLOR] * rewards.size
    best_colors[best_index] = REWARD_COLOR
    ax_best.bar(candidates, rewards, color=best_colors, edgecolor="#666666", linewidth=0.9)
    ax_best.set_title("2. Best-of-N Keeps Only the Winner")
    ax_best.set_xlabel("Sample index")
    ax_best.set_ylabel("Reward used for learning")
    ax_best.set_xlim(0.3, candidates[-1] + 0.7)
    format_axis(ax_best)
    ax_best.text(
        0.04,
        0.90,
        "Winner-take-all search uses 1 sample out of N.\n"
        "The rest of the sampled evidence is discarded.",
        transform=ax_best.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    credit_colors = [BASELINE_COLOR if value >= 0.0 else REWARD_COLOR for value in relative_credit]
    ax_relative.bar(candidates, relative_credit, color=credit_colors, edgecolor="#555555", linewidth=0.8)
    ax_relative.axhline(0.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_relative.set_title("3. Relative Credit Uses the Whole Candidate Set")
    ax_relative.set_xlabel("Sample index")
    ax_relative.set_ylabel("Centered training signal")
    ax_relative.set_xlim(0.3, candidates[-1] + 0.7)
    format_axis(ax_relative)
    ax_relative.text(
        0.04,
        0.07,
        "Positive samples pull probability up.\nNegative samples push weak candidates down.",
        transform=ax_relative.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.7,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    rng = np.random.default_rng(SEED)
    n_values = np.arange(1, 33)
    draws = rng.normal(loc=0.0, scale=1.0, size=(5000, n_values[-1]))
    expected_best = np.array([draws[:, :n].max(axis=1).mean() for n in n_values])
    learning_signal_fraction = 1.0 / n_values

    ax_tradeoff.plot(n_values, expected_best, color=REWARD_COLOR, linewidth=2.4, label="Expected best reward")
    ax_tradeoff.set_title("4. Search Quality Improves, Signal Density Falls")
    ax_tradeoff.set_xlabel("Samples per prompt (N)")
    ax_tradeoff.set_ylabel("Expected best reward", color=REWARD_COLOR)
    ax_tradeoff.tick_params(axis="y", labelcolor=REWARD_COLOR)
    ax_tradeoff.set_xlim(1, n_values[-1])
    format_axis(ax_tradeoff)

    ax_tradeoff_right = ax_tradeoff.twinx()
    ax_tradeoff_right.plot(
        n_values,
        learning_signal_fraction,
        color=POLICY_COLOR,
        linewidth=2.2,
        linestyle="--",
        label="Best-of-N learning signal used",
    )
    ax_tradeoff_right.axhline(1.0, color=BASELINE_COLOR, linewidth=1.2, linestyle=":", label="All-sample signal used")
    ax_tradeoff_right.set_ylabel("Fraction of samples used for learning", color=POLICY_COLOR)
    ax_tradeoff_right.tick_params(axis="y", labelcolor=POLICY_COLOR)
    ax_tradeoff_right.set_ylim(0.0, 1.05)
    lines_left, labels_left = ax_tradeoff.get_legend_handles_labels()
    lines_right, labels_right = ax_tradeoff_right.get_legend_handles_labels()
    ax_tradeoff.legend(lines_left + lines_right, labels_left + labels_right, loc="center right", framealpha=0.95)

    add_page_chrome(
        fig,
        "2. Best-of-N, Rejection Sampling, and Policy Optimization",
        "Sampling more candidates improves the chance of finding a good response, but winner-take-all methods throw away most of the evidence they paid to generate.",
        "Takeaway: best-of-N is a strong search primitive, but policy optimization becomes more sample-efficient when it can learn from the entire ranked candidate set instead of only the top completion.",
    )
    return fig


def anchored_policy(x: np.ndarray, reference_density: np.ndarray, reward_curve: np.ndarray, temperature: float) -> np.ndarray:
    """Return the KL-anchored optimizer p*(x) proportional to q(x) exp(r(x) / lambda)."""
    log_density = np.log(np.clip(reference_density, EPSILON, None)) + reward_curve / temperature
    shifted = log_density - np.max(log_density)
    density = np.exp(shifted)
    return normalize_density(x, density)


def build_reward_hacking_figure(points: int) -> Figure:
    """Build page 3: reward hacking and KL anchoring."""
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.05,
        right=0.985,
        top=0.83,
        bottom=0.18,
        hspace=0.34,
        wspace=0.24,
    )
    ax_reference = fig.add_subplot(grid[0, 0])
    ax_reward = fig.add_subplot(grid[0, 1])
    ax_shift = fig.add_subplot(grid[1, 0])
    ax_frontier = fig.add_subplot(grid[1, 1])

    x = np.linspace(-4.0, 4.0, points)
    reference_density = normalize_density(x, normal_pdf(x, mean=-0.1, std=1.0))
    reward_curve = (
        0.10
        + 0.45 * np.exp(-0.5 * np.square((x - 1.4) / 0.75))
        + 1.85 * np.exp(-0.5 * np.square((x - 2.75) / 0.22))
    )
    strong_anchor = anchored_policy(x, reference_density, reward_curve, temperature=1.6)
    medium_anchor = anchored_policy(x, reference_density, reward_curve, temperature=0.75)
    weak_anchor = anchored_policy(x, reference_density, reward_curve, temperature=0.24)

    ax_reference.plot(x, reference_density, color=REFERENCE_COLOR, linewidth=2.6)
    ax_reference.fill_between(x, reference_density, color="#d9d9d9", alpha=0.32)
    ax_reference.set_title("1. Reference Behavior Lives in a Broad Region")
    ax_reference.set_xlabel("Behavior axis")
    ax_reference.set_ylabel("Reference density")
    format_axis(ax_reference)
    ax_reference.text(
        0.05,
        0.90,
        "Think of this as the model's original distribution:\ncoherent, broad, and not sharply tuned to the proxy reward.",
        transform=ax_reference.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    ax_reward.plot(x, reward_curve, color=REWARD_COLOR, linewidth=2.5)
    ax_reward.fill_between(x, 0.0, reward_curve, where=(x > 2.4), color=REWARD_LIGHT, alpha=0.55)
    ax_reward.axvspan(2.45, 3.05, color=REWARD_LIGHT, alpha=0.35)
    ax_reward.set_title("2. A Proxy Reward Can Have a Narrow Exploit Peak")
    ax_reward.set_xlabel("Behavior axis")
    ax_reward.set_ylabel("Reward")
    format_axis(ax_reward)
    ax_reward.annotate(
        "Hacky region:\nhigh proxy reward,\npoorly supported by the base model",
        xy=(2.75, reward_curve[np.argmax(reward_curve)]),
        xytext=(0.85, 1.72),
        color=REWARD_COLOR,
        fontsize=9.8,
        arrowprops={"arrowstyle": "->", "color": REWARD_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": REWARD_COLOR},
    )

    ax_shift.axvspan(2.45, 3.05, color=REWARD_LIGHT, alpha=0.25)
    ax_shift.plot(x, reference_density, color=REFERENCE_COLOR, linewidth=2.2, label="Reference")
    ax_shift.plot(x, strong_anchor, color=BASELINE_COLOR, linewidth=2.1, label="Strong anchor")
    ax_shift.plot(x, medium_anchor, color=POLICY_COLOR, linewidth=2.1, label="Medium anchor")
    ax_shift.plot(x, weak_anchor, color=REWARD_COLOR, linewidth=2.3, label="Weak anchor")
    ax_shift.set_title("3. Optimizers Move Along a Reward-KL Frontier")
    ax_shift.set_xlabel("Behavior axis")
    ax_shift.set_ylabel("Policy density")
    ax_shift.legend(loc="upper left", framealpha=0.95)
    format_axis(ax_shift)

    temperatures = np.geomspace(0.18, 2.6, 36)
    expected_rewards = []
    divergences = []
    for temperature in temperatures:
        policy_density = anchored_policy(x, reference_density, reward_curve, temperature=float(temperature))
        expected_rewards.append(float(np.trapezoid(policy_density * reward_curve, x)))
        divergences.append(kl_divergence(x, policy_density, reference_density))
    expected_rewards_array = np.array(expected_rewards)
    divergences_array = np.array(divergences)

    ax_frontier.plot(divergences_array, expected_rewards_array, color=POLICY_COLOR, linewidth=2.5)
    ax_frontier.scatter(
        [
            kl_divergence(x, strong_anchor, reference_density),
            kl_divergence(x, medium_anchor, reference_density),
            kl_divergence(x, weak_anchor, reference_density),
        ],
        [
            float(np.trapezoid(strong_anchor * reward_curve, x)),
            float(np.trapezoid(medium_anchor * reward_curve, x)),
            float(np.trapezoid(weak_anchor * reward_curve, x)),
        ],
        color=[BASELINE_COLOR, POLICY_COLOR, REWARD_COLOR],
        s=58,
        zorder=3,
    )
    ax_frontier.set_title("4. Higher Reward Usually Costs More Divergence")
    ax_frontier.set_xlabel(r"$D_{KL}(p || q)$")
    ax_frontier.set_ylabel("Expected reward")
    format_axis(ax_frontier)
    ax_frontier.text(
        0.05,
        0.08,
        r"$p^*(x) \propto q(x)\exp(r(x) / \lambda)$",
        transform=ax_frontier.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    add_page_chrome(
        fig,
        "3. Reward Hacking and KL Anchoring",
        "Optimizing a misspecified reward pulls probability mass toward exploit regions; KL anchoring bends that objective back toward the model's original distribution.",
        "Takeaway: if the reward has loopholes, unconstrained optimization will find them. Anchoring does not fix the reward, but it limits how aggressively the policy can run toward proxy-only wins.",
    )
    return fig


def simulate_gradient_estimators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate three policy-gradient estimators with different variance properties."""
    rng = np.random.default_rng(SEED)
    sample_count = 6400
    policy_probability = 0.35
    actions = rng.binomial(1, policy_probability, size=sample_count)
    score = actions - policy_probability
    rewards = 0.4 + 1.6 * actions + rng.normal(loc=0.0, scale=1.15, size=sample_count)

    raw_estimator = rewards * score
    scalar_baseline = rewards.mean()
    baseline_estimator = (rewards - scalar_baseline) * score

    group_size = 4
    group_rewards = rewards.reshape(-1, group_size)
    group_means = np.repeat(group_rewards.mean(axis=1), group_size)
    relative_estimator = (rewards - group_means) * score
    return raw_estimator, baseline_estimator, relative_estimator


def build_variance_reduction_figure(points: int) -> Figure:
    """Build page 4: variance reduction with baselines."""
    del points
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.05,
        right=0.985,
        top=0.83,
        bottom=0.18,
        hspace=0.34,
        wspace=0.24,
    )
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_baseline = fig.add_subplot(grid[0, 1])
    ax_relative = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    raw_estimator, baseline_estimator, relative_estimator = simulate_gradient_estimators()
    bins = np.linspace(-3.2, 3.2, 44)

    ax_raw.hist(raw_estimator, bins=bins, color=REWARD_LIGHT, edgecolor=REWARD_COLOR, alpha=0.85)
    ax_raw.axvline(raw_estimator.mean(), color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_raw.set_title("1. Raw REINFORCE-Style Estimator")
    ax_raw.set_xlabel("Gradient estimate")
    ax_raw.set_ylabel("Count")
    format_axis(ax_raw)

    ax_baseline.hist(baseline_estimator, bins=bins, color=POLICY_LIGHT, edgecolor=POLICY_COLOR, alpha=0.85)
    ax_baseline.axvline(baseline_estimator.mean(), color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_baseline.set_title("2. Subtracting a Scalar Baseline Tightens the Spread")
    ax_baseline.set_xlabel("Gradient estimate")
    ax_baseline.set_ylabel("Count")
    format_axis(ax_baseline)

    ax_relative.hist(relative_estimator, bins=bins, color=BASELINE_LIGHT, edgecolor=BASELINE_COLOR, alpha=0.9)
    ax_relative.axvline(relative_estimator.mean(), color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_relative.set_title("3. Relative Rewards Reduce Variance Further")
    ax_relative.set_xlabel("Gradient estimate")
    ax_relative.set_ylabel("Count")
    format_axis(ax_relative)
    ax_relative.text(
        0.04,
        0.91,
        "Same signal direction, less noisy magnitude.\nThat makes gradient steps far more usable.",
        transform=ax_relative.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    names = ["Raw", "Scalar baseline", "Relative group"]
    means = np.array(
        [
            raw_estimator.mean(),
            baseline_estimator.mean(),
            relative_estimator.mean(),
        ]
    )
    stds = np.array(
        [
            raw_estimator.std(ddof=0),
            baseline_estimator.std(ddof=0),
            relative_estimator.std(ddof=0),
        ]
    )
    colors = [REWARD_COLOR, POLICY_COLOR, BASELINE_COLOR]

    x_positions = np.arange(len(names))
    ax_summary.bar(x_positions, means, color=colors, alpha=0.88)
    ax_summary.errorbar(
        x_positions,
        means,
        yerr=stds,
        fmt="none",
        ecolor=REFERENCE_COLOR,
        elinewidth=1.3,
        capsize=6,
    )
    ax_summary.axhline(0.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_summary.set_xticks(x_positions, names)
    ax_summary.set_title("4. Means Stay Similar While Variance Shrinks")
    ax_summary.set_ylabel("Estimate mean with +/- 1 std")
    format_axis(ax_summary)
    for index, std in enumerate(stds):
        ax_summary.text(index, means[index] + np.sign(means[index] + 1e-6) * 0.05, f"std={std:.2f}", ha="center", va="bottom", fontsize=9.3)

    add_page_chrome(
        fig,
        "4. Variance Reduction with Baselines",
        "Baselines and relative rewards do not change what the policy should prefer in expectation, but they make the gradient estimate much less noisy.",
        "Takeaway: variance reduction is one of the most practical tricks in policy optimization - not because it changes the objective, but because it makes the objective numerically usable.",
    )
    return fig


def build_length_bias_figure(points: int) -> Figure:
    """Build page 5: length bias in token-level RL."""
    del points
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.83,
        bottom=0.18,
        wspace=0.28,
        width_ratios=[1.1, 1.0, 1.2],
    )
    ax_timelines = fig.add_subplot(grid[0, 0])
    ax_broadcast = fig.add_subplot(grid[0, 1])
    ax_influence = fig.add_subplot(grid[0, 2])

    lengths = {"Short completion": 4, "Long completion": 14}
    y_positions = {"Short completion": 1.0, "Long completion": 0.0}
    colors = {"Short completion": POLICY_COLOR, "Long completion": REWARD_COLOR}

    for name, length in lengths.items():
        y = y_positions[name]
        ax_timelines.broken_barh([(0, length)], (y - 0.18, 0.34), facecolors=colors[name], alpha=0.85)
        for token_index in range(length):
            ax_timelines.text(token_index + 0.5, y, f"t{token_index + 1}", ha="center", va="center", fontsize=8.4, color="white")
        ax_timelines.text(length + 0.55, y, "sequence reward = 1.0", ha="left", va="center", fontsize=10.0, color="#333333")
    ax_timelines.set_xlim(0, 18.5)
    ax_timelines.set_ylim(-0.6, 1.6)
    ax_timelines.set_yticks([1.0, 0.0], ["Short", "Long"])
    ax_timelines.set_xlabel("Response token position")
    ax_timelines.set_title("1. Same Reward, Different Response Lengths")
    ax_timelines.grid(alpha=0.18, axis="x")
    ax_timelines.set_axisbelow(True)

    max_length = max(lengths.values())
    broadcast = np.full((2, max_length), np.nan)
    broadcast[0, : lengths["Short completion"]] = 1.0
    broadcast[1, : lengths["Long completion"]] = 1.0
    masked = np.ma.masked_invalid(broadcast)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")
    ax_broadcast.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax_broadcast.set_yticks([0, 1], ["Short", "Long"])
    ax_broadcast.set_xticks(np.arange(max_length), [str(index + 1) for index in range(max_length)])
    ax_broadcast.set_xlabel("Token position")
    ax_broadcast.set_title("2. Sequence Credit Gets Broadcast to Every Token")
    for row_index, length in enumerate([lengths["Short completion"], lengths["Long completion"]]):
        for token_index in range(length):
            ax_broadcast.text(token_index, row_index, "1.0", ha="center", va="center", fontsize=7.5, color="white")
    ax_broadcast.text(
        0.04,
        0.04,
        "If the loss averages over all tokens in the batch,\n"
        "the longer response contributes more total gradient weight.",
        transform=ax_broadcast.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.6,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    response_lengths = np.arange(1, 33)
    batch_token_mean = response_lengths / response_lengths.mean()
    sample_normalized = np.ones_like(response_lengths, dtype=np.float64)

    ax_influence.plot(response_lengths, batch_token_mean, color=REWARD_COLOR, linewidth=2.5, label="Batch token mean")
    ax_influence.plot(response_lengths, sample_normalized, color=BASELINE_COLOR, linewidth=2.2, linestyle="--", label="Per-sample normalization")
    ax_influence.set_title("3. Total Sample Influence Rises with Length")
    ax_influence.set_xlabel("Response length")
    ax_influence.set_ylabel("Relative gradient weight")
    ax_influence.set_xlim(1, response_lengths[-1])
    ax_influence.legend(loc="upper left", framealpha=0.95)
    format_axis(ax_influence)
    ax_influence.annotate(
        "Longer responses can dominate the update\nwithout being better in reward.",
        xy=(24, batch_token_mean[23]),
        xytext=(7.5, 1.55),
        color=REWARD_COLOR,
        fontsize=9.7,
        arrowprops={"arrowstyle": "->", "color": REWARD_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": REWARD_COLOR},
    )

    add_page_chrome(
        fig,
        "5. Length Bias in Token-Level RL",
        "When sequence-level reward is copied onto every generated token, the reduction choice determines whether long responses get more optimization weight than short ones.",
        "Takeaway: response masking says which tokens count, but it does not by itself decide how much each completion should count. The reduction over tokens still sets the length bias.",
    )
    return fig


def build_exploration_budget_figure(points: int) -> Figure:
    """Build page 6: exploration budget, temperature, N, and pass@k."""
    del points
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.83,
        bottom=0.18,
        wspace=0.28,
        width_ratios=[1.1, 1.05, 1.2],
    )
    ax_temperature = fig.add_subplot(grid[0, 0])
    ax_passk = fig.add_subplot(grid[0, 1])
    ax_outliers = fig.add_subplot(grid[0, 2])

    action_labels = np.array(["1", "2", "3", "4", "5"])
    base_logits = np.array([4.2, 3.5, 2.8, 1.9, 1.5], dtype=np.float64)
    reward_by_action = np.array([0.20, 0.28, 0.45, 1.18, 1.32], dtype=np.float64)
    good_mask = reward_by_action >= 1.0
    temperatures = [0.45, 1.0, 1.7]
    colors = [POLICY_COLOR, BASELINE_COLOR, REWARD_COLOR]
    labels = ["Low temperature", "Medium temperature", "High temperature"]

    for temperature, color, label in zip(temperatures, colors, labels):
        probabilities = softmax(base_logits / temperature)
        ax_temperature.plot(action_labels, probabilities, marker="o", linewidth=2.2, color=color, label=label)
    ax_temperature.set_title("1. Temperature Reshapes the Sampling Distribution")
    ax_temperature.set_xlabel("Action rank")
    ax_temperature.set_ylabel("Sampling probability")
    ax_temperature.legend(loc="upper right", framealpha=0.95)
    format_axis(ax_temperature)

    n_values = np.arange(1, 33)
    for temperature, color, label in zip(temperatures, colors, labels):
        probabilities = softmax(base_logits / temperature)
        good_mass = float(probabilities[good_mask].sum())
        discovery_probability = 1.0 - np.power(1.0 - good_mass, n_values)
        ax_passk.plot(n_values, discovery_probability, color=color, linewidth=2.3, label=label)
    ax_passk.set_title("2. More Samples Raise the Chance of Seeing a Great Rollout")
    ax_passk.set_xlabel("Samples per prompt (N)")
    ax_passk.set_ylabel("Probability of at least one high-reward sample")
    ax_passk.set_xlim(1, n_values[-1])
    ax_passk.set_ylim(0.0, 1.02)
    ax_passk.legend(loc="lower right", framealpha=0.95)
    format_axis(ax_passk)

    rng = np.random.default_rng(SEED)
    outlier_scales = [0.18, 0.35, 0.65]
    for scale, color, label in zip(outlier_scales, colors, labels):
        reward_samples = rng.normal(loc=0.9, scale=scale, size=(3500, n_values[-1]))
        lucky_winner_gap = np.array([reward_samples[:, :n].max(axis=1).mean() - 0.9 for n in n_values])
        ax_outliers.plot(n_values, lucky_winner_gap, color=color, linewidth=2.3, label=label)
    ax_outliers.set_title("3. Broader Exploration Also Amplifies Lucky Outliers")
    ax_outliers.set_xlabel("Samples per prompt (N)")
    ax_outliers.set_ylabel("Winner's gap above the average sample")
    ax_outliers.set_xlim(1, n_values[-1])
    ax_outliers.legend(loc="upper left", framealpha=0.95)
    format_axis(ax_outliers)
    ax_outliers.text(
        0.04,
        0.06,
        "Higher temperature and larger N improve discovery,\n"
        "but the top sample becomes a noisier estimate of true quality.",
        transform=ax_outliers.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    add_page_chrome(
        fig,
        "6. Exploration Budget: Temperature, N, and pass@k",
        "Exploration changes both search quality and training signal quality: more diverse sampling finds more good rollouts, but it also increases the odds of chasing lucky outliers.",
        "Takeaway: sampling more and sampling hotter are not free wins. They improve discovery, but they also change which rollouts dominate the update and how noisy those winners are.",
    )
    return fig


def build_figures(points: int) -> list[Figure]:
    """Build every page in the deck in order."""
    configure_style()
    return [
        build_trust_region_figure(points),
        build_best_of_n_figure(points),
        build_reward_hacking_figure(points),
        build_variance_reduction_figure(points),
        build_length_bias_figure(points),
        build_exploration_budget_figure(points),
    ]


def write_pdf(figures: list[Figure], output_path: Path) -> None:
    """Write all figures to a single PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")


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

    figures = build_figures(args.points)
    write_pdf(figures, resolved_output)
    print(f"Saved deck to {resolved_output}")

    if should_show and plt.get_backend().lower() != "agg":
        plt.show()

    for figure in figures:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
