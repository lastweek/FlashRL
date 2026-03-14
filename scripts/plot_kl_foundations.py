#!/usr/bin/env python3
"""Generate a shared KL foundations PDF deck and legacy single-page figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - exercised by direct script usage.
    print(
        "matplotlib is required for scripts/plot_kl_foundations.py. "
        "Install it with `python3 -m pip install matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_OUTPUT_PATH = Path("scripts/kl_foundations.pdf")
DEFAULT_ESTIMATOR_OUTPUT_PATH = Path("kl_estimators.pdf")
DEFAULT_FORWARD_REVERSE_OUTPUT_PATH = Path("forward_reverse_kl.pdf")
PAGE_SIZE = (16.0, 9.0)
EPSILON = 1e-12
BERNOULLI_Q = 0.2
MIXTURE_WEIGHTS = np.array([0.55, 0.45], dtype=np.float64)
MIXTURE_MEANS = np.array([-2.0, 2.0], dtype=np.float64)
MIXTURE_STDS = np.array([0.5, 0.5], dtype=np.float64)
FORWARD_COLOR = "#1f77b4"
REVERSE_COLOR = "#d95f02"
ESTIMATOR_TWO_COLOR = "#2ca02c"
TARGET_COLOR = "#222222"


def configure_style() -> None:
    """Apply a consistent style across the KL deck pages."""
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


def build_argument_parser(
    *,
    description: str,
    default_output_path: Path,
    default_points: int,
) -> argparse.ArgumentParser:
    """Build a standard parser for KL plotting scripts."""
    parser = argparse.ArgumentParser(
        description=f"{description} Saved by default to {default_output_path}."
    )
    parser.add_argument(
        "--points",
        type=int,
        default=default_points,
        help="Number of sample points used for smooth curves.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional PDF output path. If omitted, the artifact is saved to "
            f"{default_output_path} and then shown interactively."
        ),
    )
    return parser


def validate_points(points: int, minimum_points: int) -> None:
    """Validate a point-count argument."""
    if points < minimum_points:
        raise ValueError(f"--points must be at least {minimum_points}.")


def resolve_output_path(output: Path | None, default_output_path: Path) -> tuple[Path, bool]:
    """Resolve the final PDF path and whether the figure should be shown."""
    if output is None:
        return default_output_path, True

    if output.suffix == "":
        return output.with_suffix(".pdf"), False

    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must end with .pdf or omit the suffix for automatic .pdf.")

    return output, False


def add_page_chrome(fig: Figure, title: str, subtitle: str, takeaway: str) -> None:
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


def build_kl_estimators_figure(points: int) -> Figure:
    """Create the KL estimator overview page."""
    configure_style()
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.08, right=0.97, top=0.82, bottom=0.18)
    ax = fig.add_subplot(grid[0, 0])

    z = np.linspace(-8.0, 8.0, points)
    ratio = np.exp(z)

    ax.plot(z, k1(ratio), color=FORWARD_COLOR, linewidth=2.3, label=r"$k_1(r) = -\log r$")
    ax.plot(z, k2(ratio), color=ESTIMATOR_TWO_COLOR, linewidth=2.3, label=r"$k_2(r) = \frac{1}{2}(\log r)^2$")
    ax.plot(z, k3(ratio), color=REVERSE_COLOR, linewidth=2.3, label=r"$k_3(r) = (r - 1) - \log r$")
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_title("Likelihood-Ratio Geometry of Three KL Estimators")
    ax.set_xlabel(r"$\log r$")
    ax.set_ylabel("Estimator value")
    ax.set_xlim(z[0], z[-1])
    ax.set_yscale("symlog", linthresh=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.80), framealpha=0.95)
    format_axis(ax)

    ax.text(
        0.02,
        0.48,
        (
            "All three curves agree near $r = 1$, but they behave very\n"
            "differently once the policy drifts into the tails."
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.1,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

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
        fontsize=10.8,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )

    add_page_chrome(
        fig,
        "1. Schulman's KL Estimators",
        "Three estimators over the same likelihood ratio reveal the difference between exact KL identities and a local quadratic approximation.",
        "Takeaway: k1 and k3 recover the same KL in expectation under q, while k2 is useful as a near-reference approximation rather than an exact identity.",
    )
    return fig


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


def build_forward_reverse_figure(points: int) -> Figure:
    """Create the forward-vs-reverse KL explainer page."""
    configure_style()

    x = np.linspace(-6.0, 6.0, points)
    target_density = mixture_pdf(x)
    mu_forward, sigma_forward = forward_kl_fit()
    mu_reverse, sigma_reverse, reverse_value = reverse_kl_fit(points)

    forward_density = normal_pdf(x, mu_forward, sigma_forward)
    reverse_density = normal_pdf(x, mu_reverse, sigma_reverse)

    target_safe = np.clip(target_density, EPSILON, None)
    forward_safe = np.clip(forward_density, EPSILON, None)
    reverse_safe = np.clip(reverse_density, EPSILON, None)

    forward_contrib = target_density * (np.log(target_safe) - np.log(forward_safe))
    reverse_contrib = reverse_density * (np.log(reverse_safe) - np.log(target_safe))

    bernoulli_p = np.linspace(1e-4, 1.0 - 1e-4, points)
    bernoulli_forward = bernoulli_forward_kl(BERNOULLI_Q, bernoulli_p)
    bernoulli_reverse = bernoulli_reverse_kl(bernoulli_p, BERNOULLI_Q)

    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid_spec = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.80,
        bottom=0.18,
        wspace=0.28,
        width_ratios=[1.15, 1.0, 1.0],
    )
    ax_mode = fig.add_subplot(grid_spec[0, 0])
    ax_integrand = fig.add_subplot(grid_spec[0, 1])
    ax_bernoulli = fig.add_subplot(grid_spec[0, 2])

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
    format_axis(ax_mode)
    ax_mode.legend(loc="upper right", framealpha=0.95)
    ax_mode.annotate(
        "Forward KL fit\ncovers both modes",
        xy=(0.0, normal_pdf(np.array([0.0]), mu_forward, sigma_forward)[0]),
        xytext=(-5.35, 0.31),
        color=FORWARD_COLOR,
        fontsize=10.2,
        arrowprops={"arrowstyle": "->", "color": FORWARD_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": FORWARD_COLOR},
    )
    ax_mode.annotate(
        "Reverse KL fit\nlocks onto the\nhigher-probability mode",
        xy=(-2.0, normal_pdf(np.array([-2.0]), mu_reverse, sigma_reverse)[0]),
        xytext=(-5.35, 0.66),
        color=REVERSE_COLOR,
        fontsize=10.2,
        arrowprops={"arrowstyle": "->", "color": REVERSE_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": REVERSE_COLOR},
    )

    ax_integrand.set_title("2. Local KL Contributions")
    ax_integrand.plot(x, forward_contrib, color=FORWARD_COLOR, linewidth=2.3, label=r"$q(x)\log \frac{q(x)}{p(x)}$")
    ax_integrand.plot(x, reverse_contrib, color=REVERSE_COLOR, linewidth=2.3, label=r"$p(x)\log \frac{p(x)}{q(x)}$")
    ax_integrand.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax_integrand.axvline(-2.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_integrand.axvline(2.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_integrand.set_xlabel("x")
    ax_integrand.set_ylabel("Pointwise contribution")
    ax_integrand.set_xlim(-6.0, 6.0)
    format_axis(ax_integrand)
    ax_integrand.legend(loc="upper right", framealpha=0.95)
    right_mode_index = int(np.argmin(np.abs(x - 2.0)))
    peak_reverse_index = int(np.argmax(reverse_contrib))
    integrand_ymax = max(float(np.max(forward_contrib)), float(np.max(reverse_contrib)))
    ax_integrand.text(-2.0, integrand_ymax * 0.95, "left mode", ha="center", va="top", fontsize=9.2, color="#666666")
    ax_integrand.text(2.0, integrand_ymax * 0.95, "right mode", ha="center", va="top", fontsize=9.2, color="#666666")
    ax_integrand.annotate(
        "Forward KL spikes where\nq has mass but p misses it",
        xy=(2.0, forward_contrib[right_mode_index]),
        xytext=(0.35, float(np.max(forward_contrib)) * 0.72),
        color=FORWARD_COLOR,
        fontsize=10.0,
        arrowprops={"arrowstyle": "->", "color": FORWARD_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": FORWARD_COLOR},
    )
    ax_integrand.annotate(
        "Reverse KL mainly spends its\nbudget where p already puts mass",
        xy=(x[peak_reverse_index], reverse_contrib[peak_reverse_index]),
        xytext=(-5.15, float(np.max(reverse_contrib)) * 0.55),
        color=REVERSE_COLOR,
        fontsize=10.0,
        arrowprops={"arrowstyle": "->", "color": REVERSE_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": REVERSE_COLOR},
    )
    ax_integrand.text(
        0.03,
        0.04,
        rf"Reverse KL optimum: $\mu={mu_reverse:.2f}$, $\sigma={sigma_reverse:.2f}$, value={reverse_value:.3f}",
        transform=ax_integrand.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
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
    format_axis(ax_bernoulli)
    ax_bernoulli.legend(loc="upper center", framealpha=0.95)
    ax_bernoulli.annotate(
        "Forward KL punishes\np -> 0 when q still\nexpects this token",
        xy=(0.02, bernoulli_forward[int(0.02 * (points - 1))]),
        xytext=(0.08, 5.3),
        color=FORWARD_COLOR,
        fontsize=10.0,
        arrowprops={"arrowstyle": "->", "color": FORWARD_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": FORWARD_COLOR},
    )
    ax_bernoulli.annotate(
        "Reverse KL stays finite here,\nbut it rises as p over-allocates\nmass relative to q",
        xy=(0.88, bernoulli_reverse[int(0.88 * (points - 1))]),
        xytext=(0.48, 2.65),
        color=REVERSE_COLOR,
        fontsize=10.0,
        arrowprops={"arrowstyle": "->", "color": REVERSE_COLOR, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": REVERSE_COLOR},
    )
    ax_bernoulli.text(
        BERNOULLI_Q + 0.015,
        0.28,
        r"$p=q$ gives zero KL",
        fontsize=9.4,
        color="#555555",
    )

    add_page_chrome(
        fig,
        "2. Forward KL vs Reverse KL",
        "Two KL directions can prefer very different fitted distributions, local penalties, and token-probability tradeoffs.",
        "Takeaway: forward-like penalties strongly punish missing reference-supported mass, while reverse-like penalties more strongly punish placing probability where the target assigns little mass.",
    )
    return fig


def build_figures(points: int) -> list[Figure]:
    """Build both KL foundation pages in order."""
    return [
        build_kl_estimators_figure(points),
        build_forward_reverse_figure(points),
    ]


def save_figure(figure: Figure, output_path: Path) -> None:
    """Write one figure to a PDF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight")


def write_pdf(figures: list[Figure], output_path: Path) -> None:
    """Write a list of figures to a multi-page PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")


def _show_if_interactive(should_show: bool) -> None:
    """Show figures only when using an interactive backend."""
    if should_show and plt.get_backend().lower() != "agg":
        plt.show()


def _close_figures(figures: list[Figure]) -> None:
    """Close a list of figures."""
    for figure in figures:
        plt.close(figure)


def run_single_page_cli(
    *,
    argv: list[str] | None,
    description: str,
    default_output_path: Path,
    default_points: int,
    minimum_points: int,
    page_builder: Callable[[int], Figure],
) -> int:
    """Run a one-page plotting CLI."""
    parser = build_argument_parser(
        description=description,
        default_output_path=default_output_path,
        default_points=default_points,
    )
    args = parser.parse_args(argv)

    try:
        validate_points(args.points, minimum_points)
        resolved_output, should_show = resolve_output_path(args.output, default_output_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    figure = page_builder(args.points)
    save_figure(figure, resolved_output)
    print(f"Saved figure to {resolved_output}")
    _show_if_interactive(should_show)
    _close_figures([figure])
    return 0


def legacy_kl_estimators_main(argv: list[str] | None = None) -> int:
    """Run the legacy estimator CLI."""
    return run_single_page_cli(
        argv=argv,
        description=(
            "Plot Schulman's KL estimators versus log r, where "
            "r(x) = p(x) / q(x) is a likelihood ratio."
        ),
        default_output_path=DEFAULT_ESTIMATOR_OUTPUT_PATH,
        default_points=1000,
        minimum_points=2,
        page_builder=build_kl_estimators_figure,
    )


def legacy_forward_reverse_main(argv: list[str] | None = None) -> int:
    """Run the legacy forward-vs-reverse CLI."""
    return run_single_page_cli(
        argv=argv,
        description="Generate a one-page PDF comparing forward and reverse KL.",
        default_output_path=DEFAULT_FORWARD_REVERSE_OUTPUT_PATH,
        default_points=2000,
        minimum_points=200,
        page_builder=build_forward_reverse_figure,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the combined KL foundations deck CLI."""
    parser = build_argument_parser(
        description="Generate a two-page PDF deck covering KL estimator foundations and KL direction behavior.",
        default_output_path=DEFAULT_OUTPUT_PATH,
        default_points=2000,
    )
    args = parser.parse_args(argv)

    try:
        validate_points(args.points, 200)
        resolved_output, should_show = resolve_output_path(args.output, DEFAULT_OUTPUT_PATH)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    figures = build_figures(args.points)
    write_pdf(figures, resolved_output)
    print(f"Saved deck to {resolved_output}")
    _show_if_interactive(should_show)
    _close_figures(figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
