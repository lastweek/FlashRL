#!/usr/bin/env python3
"""Compatibility wrapper for the forward-vs-reverse KL page."""

from __future__ import annotations

from plot_kl_foundations import (
    bernoulli_forward_kl,
    bernoulli_reverse_kl,
    build_forward_reverse_figure as make_figure,
    forward_kl_fit,
    legacy_forward_reverse_main,
    mixture_moments,
    mixture_pdf,
    normal_logpdf,
    normal_pdf,
    reverse_kl_fit,
)


def main(argv: list[str] | None = None) -> int:
    """Run the legacy forward-vs-reverse entrypoint."""
    return legacy_forward_reverse_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
