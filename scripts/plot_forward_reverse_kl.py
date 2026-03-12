#!/usr/bin/env python3
"""Generate a one-page PDF explaining forward vs reverse KL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised by direct script usage.
    print(
        "matplotlib is required for scripts/plot_forward_reverse_kl.py. "
        "Install it with `python3 -m pip install matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_OUTPUT_PATH = Path("forward_reverse_kl.pdf")
EPSILON = 1e-12
BERNOULLI_Q = 0.2
MIXTURE_WEIGHTS = np.array([0.55, 0.45], dtype=np.float64)
MIXTURE_MEANS = np.array([-2.0, 2.0], dtype=np.float64)
MIXTURE_STDS = np.array([0.5, 0.5], dtype=np.float64)
FORWARD_COLOR = "#1f77b4"
REVERSE_COLOR = "#d95f02"
TARGET_COLOR = "#222222"


def normal_pdf(x: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    """Compute the Gaussian density."""
    z = (x - mean) / std
    return np.exp(-0.5 * np.square(z)) / (std * np.sqrt(2.0 * np.pi))


def normal_logpdf(x: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    """Compute the Gaussian log-density."""
    z = (x - mean) / std
    return -0.5 * np.square(z) - np.log(std * np.sqrt(2.0 * np.pi))


def mixture_pdf(x: np.ndarray) -> np.ndarray:
    """Evaluate the asymmetric bimodal target density."""
    components = MIXTURE_WEIGHTS[:, None] * normal_pdf(
        x[None, :],
        MIXTURE_MEANS[:, None],
        MIXTURE_STDS[:, None],
    )
    return components.sum(axis=0)


def mixture_moments() -> tuple[float, float]:
    """Return the mean and standard deviation of the Gaussian mixture."""
    mean = float(np.dot(MIXTURE_WEIGHTS, MIXTURE_MEANS))
    variance = float(
        np.dot(MIXTURE_WEIGHTS, np.square(MIXTURE_STDS) + np.square(MIXTURE_MEANS - mean))
    )
    return mean, float(np.sqrt(variance))


def forward_kl_fit() -> tuple[float, float]:
    """Return the moment-matched Gaussian minimizing D_KL(q || p_theta)."""
    return mixture_moments()


def reverse_kl_grid_search(
    x_grid: np.ndarray,
    target_density: np.ndarray,
    mu_grid: np.ndarray,
    sigma_grid: np.ndarray,
) -> tuple[float, float, float]:
    """Search for the Gaussian minimizing D_KL(p_theta || q)."""
    log_q = np.log(np.clip(target_density, EPSILON, None))
    best_mu = float(mu_grid[0])
    best_sigma = float(sigma_grid[0])
    best_value = float("inf")

    for sigma in sigma_grid:
        log_p = normal_logpdf(x_grid[None, :], mu_grid[:, None], sigma)
        density = np.exp(log_p)
        reverse_kl = np.trapezoid(density * (log_p - log_q[None, :]), x_grid, axis=1)
        best_index = int(np.argmin(reverse_kl))
        candidate_value = float(reverse_kl[best_index])
        if candidate_value < best_value:
            best_value = candidate_value
            best_mu = float(mu_grid[best_index])
            best_sigma = float(sigma)

    return best_mu, best_sigma, best_value


def reverse_kl_fit(points: int) -> tuple[float, float, float]:
    """Return a deterministic reverse-KL Gaussian fit via coarse-to-fine search."""
    x_grid = np.linspace(-6.0, 6.0, max(points, 2000))
    target_density = mixture_pdf(x_grid)

    coarse_mu = np.linspace(-3.5, 3.5, 241)
    coarse_sigma = np.linspace(0.25, 2.5, 181)
    best_mu, best_sigma, best_value = reverse_kl_grid_search(
        x_grid,
        target_density,
        coarse_mu,
        coarse_sigma,
    )

    refine_mu = np.linspace(max(-3.5, best_mu - 0.25), min(3.5, best_mu + 0.25), 161)
    refine_sigma = np.linspace(max(0.25, best_sigma - 0.15), min(2.5, best_sigma + 0.15), 161)
    return reverse_kl_grid_search(x_grid, target_density, refine_mu, refine_sigma)


def bernoulli_forward_kl(prob_q: float, prob_p: np.ndarray) -> np.ndarray:
    """Compute D_KL(q || p) for Bernoulli distributions."""
    return prob_q * np.log(prob_q / prob_p) + (1.0 - prob_q) * np.log((1.0 - prob_q) / (1.0 - prob_p))


def bernoulli_reverse_kl(prob_p: np.ndarray, prob_q: float) -> np.ndarray:
    """Compute D_KL(p || q) for Bernoulli distributions."""
    return prob_p * np.log(prob_p / prob_q) + (1.0 - prob_p) * np.log((1.0 - prob_p) / (1.0 - prob_q))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a one-page PDF comparing forward and reverse KL, "
            f"saved by default to {DEFAULT_OUTPUT_PATH}."
        )
    )
    parser.add_argument(
        "--points",
        type=int,
        default=2000,
        help="Number of sample points used for the plotted curves.",
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


def make_figure(points: int) -> plt.Figure:
    """Create the forward-vs-reverse KL explainer figure."""
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    x = np.linspace(-6.0, 6.0, points)
    target_density = mixture_pdf(x)
    mu_forward, sigma_forward = forward_kl_fit()
    mu_reverse, sigma_reverse, reverse_value = reverse_kl_fit(points)

    forward_density = normal_pdf(x, mu_forward, sigma_forward)
    reverse_density = normal_pdf(x, mu_reverse, sigma_reverse)

    target_safe = np.clip(target_density, EPSILON, None)
    forward_safe = np.clip(forward_density, EPSILON, None)
    reverse_safe = np.clip(reverse_density, EPSILON, None)

    forward_contrib = target_density * (np.log(target_safe) - np.log(reverse_safe))
    reverse_contrib = reverse_density * (np.log(reverse_safe) - np.log(target_safe))

    bernoulli_p = np.linspace(1e-4, 1.0 - 1e-4, points)
    bernoulli_forward = bernoulli_forward_kl(BERNOULLI_Q, bernoulli_p)
    bernoulli_reverse = bernoulli_reverse_kl(bernoulli_p, BERNOULLI_Q)

    fig = plt.figure(figsize=(17, 7.5), facecolor="white")
    grid_spec = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.78,
        bottom=0.21,
        wspace=0.28,
        width_ratios=[1.15, 1.0, 1.0],
    )
    ax_mode = fig.add_subplot(grid_spec[0, 0])
    ax_integrand = fig.add_subplot(grid_spec[0, 1])
    ax_bernoulli = fig.add_subplot(grid_spec[0, 2])

    fig.suptitle("Forward KL vs Reverse KL", fontsize=19, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.90,
        (
            r"$D_{\mathrm{KL}}(q \parallel p) = \mathbb{E}_{x \sim q}[\log q(x) - \log p(x)]"
            r"\qquad"
            r"D_{\mathrm{KL}}(p \parallel q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$"
        ),
        ha="center",
        va="center",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.855,
        "Three views: what gets fit, where the penalty lands, and what this means for token probabilities.",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

    ax_mode.set_title("1. Mode Behavior")
    ax_mode.fill_between(x, target_density, color="#d9d9d9", alpha=0.35)
    ax_mode.plot(x, target_density, color=TARGET_COLOR, linewidth=2.4, label="Target q(x)")
    ax_mode.plot(
        x,
        forward_density,
        color=FORWARD_COLOR,
        linewidth=2.4,
        label=rf"Forward KL fit  N({mu_forward:.2f}, {sigma_forward:.2f}$^2$)",
    )
    ax_mode.plot(
        x,
        reverse_density,
        color=REVERSE_COLOR,
        linewidth=2.4,
        label=rf"Reverse KL fit  N({mu_reverse:.2f}, {sigma_reverse:.2f}$^2$)",
    )
    ax_mode.set_xlabel("x")
    ax_mode.set_ylabel("Density")
    ax_mode.set_xlim(-6.0, 6.0)
    ax_mode.set_ylim(0.0, max(target_density.max(), reverse_density.max(), forward_density.max()) * 1.18)
    ax_mode.grid(alpha=0.22)
    ax_mode.legend(loc="upper right", framealpha=0.95)
    ax_mode.annotate(
        "Forward KL fit\ncovers both modes",
        xy=(0.0, normal_pdf(np.array([0.0]), mu_forward, sigma_forward)[0]),
        xytext=(-5.35, 0.31),
        color=FORWARD_COLOR,
        fontsize=10.5,
        arrowprops={"arrowstyle": "->", "color": FORWARD_COLOR, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": FORWARD_COLOR},
    )
    ax_mode.annotate(
        "Reverse KL fit\nlocks onto the\nhigher-probability mode",
        xy=(-2.0, normal_pdf(np.array([-2.0]), mu_reverse, sigma_reverse)[0]),
        xytext=(-5.35, 0.66),
        color=REVERSE_COLOR,
        fontsize=10.5,
        arrowprops={"arrowstyle": "->", "color": REVERSE_COLOR, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": REVERSE_COLOR},
    )

    ax_integrand.set_title("2. Local KL Contributions")
    ax_integrand.plot(x, forward_contrib, color=FORWARD_COLOR, linewidth=2.3, label=r"$q(x)\log \frac{q(x)}{p(x)}$")
    ax_integrand.plot(x, reverse_contrib, color=REVERSE_COLOR, linewidth=2.3, label=r"$p(x)\log \frac{p(x)}{q(x)}$")
    ax_integrand.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax_integrand.axvline(-2.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_integrand.axvline(2.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_integrand.text(-2.0, ax_integrand.get_ylim()[1] * 0.92 if ax_integrand.get_ylim()[1] else 0.0, "left mode", ha="center", va="top", fontsize=9.5, color="#666666")
    ax_integrand.text(2.0, ax_integrand.get_ylim()[1] * 0.92 if ax_integrand.get_ylim()[1] else 0.0, "right mode", ha="center", va="top", fontsize=9.5, color="#666666")
    ax_integrand.set_xlabel("x")
    ax_integrand.set_ylabel("Pointwise contribution")
    ax_integrand.set_xlim(-6.0, 6.0)
    ax_integrand.grid(alpha=0.22)
    ax_integrand.legend(loc="upper right", framealpha=0.95)
    right_mode_index = int(np.argmin(np.abs(x - 2.0)))
    peak_reverse_index = int(np.argmax(reverse_contrib))
    ax_integrand.annotate(
        "Forward KL spikes where\nq has mass but p misses it",
        xy=(2.0, forward_contrib[right_mode_index]),
        xytext=(0.35, forward_contrib.max() * 0.72),
        color=FORWARD_COLOR,
        fontsize=10.3,
        arrowprops={"arrowstyle": "->", "color": FORWARD_COLOR, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": FORWARD_COLOR},
    )
    ax_integrand.annotate(
        "Reverse KL mainly spends its\nbudget where p already puts mass",
        xy=(x[peak_reverse_index], reverse_contrib[peak_reverse_index]),
        xytext=(-5.15, reverse_contrib.max() * 0.55),
        color=REVERSE_COLOR,
        fontsize=10.3,
        arrowprops={"arrowstyle": "->", "color": REVERSE_COLOR, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": REVERSE_COLOR},
    )
    ax_integrand.text(
        0.03,
        0.04,
        rf"Reverse KL optimum: $\mu={mu_reverse:.2f}$, $\sigma={sigma_reverse:.2f}$, value={reverse_value:.3f}",
        transform=ax_integrand.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.6,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    ax_bernoulli.set_title("3. Bernoulli / Token Intuition")
    ax_bernoulli.plot(
        bernoulli_p,
        bernoulli_forward,
        color=FORWARD_COLOR,
        linewidth=2.4,
        label=rf"$D_{{KL}}(q \parallel p)$ with $q={BERNOULLI_Q:.1f}$",
    )
    ax_bernoulli.plot(
        bernoulli_p,
        bernoulli_reverse,
        color=REVERSE_COLOR,
        linewidth=2.4,
        label=rf"$D_{{KL}}(p \parallel q)$ with $q={BERNOULLI_Q:.1f}$",
    )
    ax_bernoulli.axvline(BERNOULLI_Q, color="#666666", linewidth=1.0, linestyle="--")
    ax_bernoulli.set_xlabel("Model probability p")
    ax_bernoulli.set_ylabel("KL value")
    ax_bernoulli.set_xlim(0.0, 1.0)
    ax_bernoulli.set_ylim(0.0, 7.8)
    ax_bernoulli.grid(alpha=0.22)
    ax_bernoulli.legend(loc="upper center", framealpha=0.95)
    ax_bernoulli.annotate(
        "Forward KL punishes\np -> 0 when q still\nexpects this token",
        xy=(0.02, bernoulli_forward[int(0.02 * (points - 1))]),
        xytext=(0.08, 5.3),
        color=FORWARD_COLOR,
        fontsize=10.3,
        arrowprops={"arrowstyle": "->", "color": FORWARD_COLOR, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": FORWARD_COLOR},
    )
    ax_bernoulli.annotate(
        "Reverse KL stays finite here,\nbut it rises as p over-allocates\nmass relative to q",
        xy=(0.88, bernoulli_reverse[int(0.88 * (points - 1))]),
        xytext=(0.48, 2.65),
        color=REVERSE_COLOR,
        fontsize=10.3,
        arrowprops={"arrowstyle": "->", "color": REVERSE_COLOR, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": REVERSE_COLOR},
    )
    ax_bernoulli.text(
        BERNOULLI_Q + 0.015,
        0.28,
        r"$p=q$ gives zero KL",
        fontsize=9.6,
        color="#555555",
    )

    fig.text(
        0.5,
        0.09,
        (
            "RL intuition: forward-like penalties strongly discourage collapsing reference-supported "
            "tokens to near zero. Reverse-like penalties more strongly discourage putting probability "
            "on tokens or actions the target assigns little mass to."
        ),
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7f7f7", "alpha": 0.98, "edgecolor": "#cccccc"},
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
