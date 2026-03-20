"""Opt-in local minikube end-to-end test for the FlashRL platform path."""

from __future__ import annotations

import os

import pytest

from flashrl.platform.minikube_e2e import run_minikube_math_e2e


pytestmark = [pytest.mark.integration, pytest.mark.minikube]


def test_minikube_math_platform_e2e() -> None:
    """Run the math example over the full minikube platform path when explicitly enabled."""
    if os.environ.get("FLASHRL_RUN_MINIKUBE_E2E") != "1":
        pytest.skip("Set FLASHRL_RUN_MINIKUBE_E2E=1 to run the local minikube E2E.")

    result = run_minikube_math_e2e(
        keep_resources=os.environ.get("FLASHRL_KEEP_MINIKUBE_E2E") == "1",
        skip_build=os.environ.get("FLASHRL_SKIP_MINIKUBE_BUILD") == "1",
        timeout_seconds=int(os.environ.get("FLASHRL_MINIKUBE_E2E_TIMEOUT", "1800")),
        artifact_dir=os.environ.get("FLASHRL_MINIKUBE_E2E_ARTIFACT_DIR"),
    )

    progress = result["job"]["status"]["progress"]
    active = result["job"]["status"]["weightVersion"]["active"]
    assert int(progress["lastCompletedStep"]) >= 1
    assert int(active["version_id"]) >= 1
