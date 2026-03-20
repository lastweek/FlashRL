"""Tests for the FlashRL platform CRD and rendered child resources."""

from __future__ import annotations

from pathlib import Path

import pytest

from flashrl.platform.cli import main as platform_main
from flashrl.platform.crd import FlashRLJob, flashrljob_crd_manifest
from flashrl.platform.operator import render_child_resources


pytestmark = pytest.mark.unit


def _job_payload() -> dict[str, object]:
    return {
        "apiVersion": "platform.flashrl.dev/v1alpha1",
        "kind": "FlashRLJob",
        "metadata": {"name": "demo-job"},
        "spec": {
            "framework": {
                "actor": {"model_name": "fake/model", "backend": "huggingface"},
                "serving": {"model_name": "fake/model", "backend": "huggingface"},
                "trainer": {"batch_size": 4, "max_epochs": 1},
                "grpo": {"group_size": 2, "kl_coefficient": 0.0},
            },
            "dataset": {"uri": "s3://datasets/train.jsonl", "format": "jsonl"},
            "controller": {"image": "ghcr.io/flashrl/controller:latest"},
            "learner": {"image": "ghcr.io/flashrl/learner:latest"},
            "servingPool": {
                "image": "ghcr.io/flashrl/serving:latest",
                "replicas": {"min": 2, "max": 4},
            },
            "rollout": {
                "image": "ghcr.io/flashrl/rollout:latest",
                "replicas": {"min": 2, "max": 8},
            },
            "reward": {
                "image": "ghcr.io/flashrl/reward:latest",
                "replicas": {"min": 1, "max": 4},
            },
            "storage": {
                "checkpoints": {"uriPrefix": "s3://checkpoints/"},
                "weights": {"uriPrefix": "s3://weights/"},
            },
        },
    }


def test_flashrl_job_rejects_local_serving_runtime_python() -> None:
    """Platform mode should reject local-only serving runtime configuration."""
    payload = _job_payload()
    payload["spec"]["framework"]["serving"]["runtime_python"] = "/tmp/python"
    with pytest.raises(ValueError, match="runtime_python"):
        FlashRLJob.model_validate(payload)


def test_render_child_resources_builds_expected_workloads() -> None:
    """One FlashRLJob should render a controller, learner, service pools, and services."""
    job = FlashRLJob.model_validate(_job_payload())
    rendered = render_child_resources(job)
    kinds = [item["kind"] for item in rendered]
    assert kinds.count("Deployment") == 4
    assert kinds.count("StatefulSet") == 1
    assert kinds.count("Service") == 5
    assert kinds.count("ConfigMap") == 1


def test_flashrl_platform_render_only_cli_prints_rendered_resources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The render-only CLI path should validate the job and print child resources."""
    job_path = tmp_path / "job.yaml"
    job_path.write_text(
        """
apiVersion: platform.flashrl.dev/v1alpha1
kind: FlashRLJob
metadata:
  name: demo-job
spec:
  framework:
    actor:
      model_name: fake/model
      backend: huggingface
    serving:
      model_name: fake/model
      backend: huggingface
    trainer:
      batch_size: 4
      max_epochs: 1
    grpo:
      group_size: 2
      kl_coefficient: 0.0
  dataset:
    uri: s3://datasets/train.jsonl
    format: jsonl
  controller:
    image: ghcr.io/flashrl/controller:latest
  learner:
    image: ghcr.io/flashrl/learner:latest
  servingPool:
    image: ghcr.io/flashrl/serving:latest
  rollout:
    image: ghcr.io/flashrl/rollout:latest
  reward:
    image: ghcr.io/flashrl/reward:latest
  storage:
    checkpoints:
      uriPrefix: s3://checkpoints/
    weights:
      uriPrefix: s3://weights/
""",
        encoding="utf-8",
    )

    exit_code = platform_main(["platform", "submit", "--file", str(job_path), "--render-only"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"kind": "Deployment"' in output
    assert '"kind": "StatefulSet"' in output


def test_flashrljob_crd_manifest_exposes_expected_kind() -> None:
    """The generated CRD manifest should expose the expected resource kind."""
    manifest = flashrljob_crd_manifest()
    assert manifest["kind"] == "CustomResourceDefinition"
    assert manifest["spec"]["names"]["kind"] == "FlashRLJob"
