#!/usr/bin/env python3
"""Compatibility wrapper for the KL estimators page."""

from __future__ import annotations

from plot_kl_foundations import build_kl_estimators_figure as make_figure
from plot_kl_foundations import k1, k2, k3, legacy_kl_estimators_main


def main(argv: list[str] | None = None) -> int:
    """Run the legacy estimator entrypoint."""
    return legacy_kl_estimators_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
