#!/usr/bin/env python3
"""Run the local minikube FlashRL platform E2E."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashrl.platform.minikube_e2e import main


if __name__ == "__main__":
    raise SystemExit(main())
