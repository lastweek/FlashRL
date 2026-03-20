"""Tests for the platform controller/runtime entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flashrl.platform.runtime.controller as controller_module


pytestmark = pytest.mark.unit


def _job_payload() -> dict[str, object]:
    return {
        "apiVersion": "platform.flashrl.dev/v1alpha1",
        "kind": "FlashRLJob",
        "metadata": {"name": "demo-job", "namespace": "default"},
        "spec": {
            "framework": {
                "actor": {"model_name": "fake/model", "backend": "huggingface"},
                "serving": {"model_name": "fake/model", "backend": "huggingface"},
                "trainer": {"batch_size": 4, "max_epochs": 1},
                "grpo": {"group_size": 2, "kl_coefficient": 0.0},
            },
            "dataset": {"type": "hook"},
            "images": {
                "runtime": "ghcr.io/flashrl/flashrl-runtime:latest",
                "serving": "ghcr.io/flashrl/flashrl-serving-vllm:latest",
                "training": "ghcr.io/flashrl/flashrl-training-fsdp:latest",
            },
            "userCode": {
                "dataset": {"import": "tests.platform_hooks:build_dataset"},
                "rollout": {"import": "tests.platform_hooks:build_rollout"},
                "reward": {"import": "tests.platform_hooks:build_reward"},
            },
            "storage": {
                "checkpoints": {"uriPrefix": "/tmp/flashrl/checkpoints"},
                "weights": {"uriPrefix": "/tmp/flashrl/weights"},
            },
        },
    }


def test_run_controller_constructs_http_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The platform controller should instantiate GRPOTrainer with HTTP clients."""
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def status(self):
            return type(
                "_Status",
                (),
                {"status": type("_Payload", (), {"active_weight_version": None})()},
            )()

    class _StubStatusWriter:
        def __init__(self, *, job_name: str, namespace: str) -> None:
            captured["status_writer"] = (job_name, namespace)

        def get_job(self):
            return None

        def patch_status(self, patch):
            captured.setdefault("status_patches", []).append(patch)

    class _StubTrainer:
        def __init__(self, **kwargs) -> None:
            captured["trainer_kwargs"] = kwargs
            self.current_epoch = 0
            self.total_steps = 0

        def load_checkpoint(self, path: str) -> None:
            captured["resume_path"] = path

        def save_checkpoint(self, path: str) -> None:
            captured["saved_path"] = path

        def train(self, dataset) -> None:
            captured["dataset"] = list(dataset)

    monkeypatch.setattr(controller_module, "HttpRolloutClient", _StubClient)
    monkeypatch.setattr(controller_module, "HttpRewardClient", _StubClient)
    monkeypatch.setattr(controller_module, "HttpLearnerClient", _StubClient)
    monkeypatch.setattr(controller_module, "HttpServingClient", _StubClient)
    monkeypatch.setattr(controller_module, "ControllerStatusWriter", _StubStatusWriter)
    monkeypatch.setattr(controller_module, "GRPOTrainer", _StubTrainer)
    monkeypatch.setattr(controller_module, "load_dataset", lambda job: [])

    controller_module.run_controller(job_path)

    trainer_kwargs = captured["trainer_kwargs"]
    assert trainer_kwargs["rollout_client"].url.endswith("/demo-job-rollout.default.svc.cluster.local")
    assert trainer_kwargs["reward_client"].url.endswith("/demo-job-reward.default.svc.cluster.local")
    assert trainer_kwargs["learner_client"].url.endswith("/demo-job-learner-0.demo-job-learner.default.svc.cluster.local")
    assert trainer_kwargs["serving_client"].url.endswith("/demo-job-serving.default.svc.cluster.local")
    assert trainer_kwargs["reference_configured"] is False
    assert captured["dataset"] == []


def test_run_controller_uses_static_runtime_spec_and_live_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Controller should keep mounted runtime paths while merging live CRD status."""
    payload = _job_payload()
    payload["spec"]["sharedStorage"] = {
        "enabled": True,
        "mountPath": "/var/lib/flashrl/shared",
        "checkpointsSubPath": "checkpoints",
        "weightsSubPath": "weights",
    }
    payload["spec"]["storage"] = {
        "checkpoints": {"uriPrefix": "/var/lib/flashrl/shared/checkpoints"},
        "weights": {"uriPrefix": "/var/lib/flashrl/shared/weights"},
    }
    payload["spec"]["checkpointing"] = {"resume_from": "latest", "directory": "/var/lib/flashrl/shared/checkpoints"}
    payload["status"] = {
        "progress": {"currentEpoch": 2, "currentStep": 5, "lastCompletedStep": 4},
        "checkpoint": {"latestUri": "/var/lib/flashrl/shared/checkpoints/step-00000004.pt"},
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def status(self):
            return type(
                "_Status",
                (),
                {"status": type("_Payload", (), {"active_weight_version": None})()},
            )()

    live_payload = _job_payload()
    live_payload["status"] = {
        "progress": {"currentEpoch": 9, "currentStep": 11, "lastCompletedStep": 10},
        "checkpoint": {"latestUri": "/remote/checkpoints/step-00000010.pt"},
    }

    class _StubStatusWriter:
        def __init__(self, *, job_name: str, namespace: str) -> None:
            del job_name, namespace

        def get_job(self):
            return controller_module.FlashRLJob.model_validate(live_payload)

        def patch_status(self, patch):
            captured.setdefault("status_patches", []).append(patch)

    class _StubTrainer:
        def __init__(self, **kwargs) -> None:
            self.current_epoch = 0
            self.total_steps = 0

        def load_checkpoint(self, path: str) -> None:
            captured["resume_path"] = path

        def save_checkpoint(self, path: str) -> None:
            captured["saved_path"] = path

        def train(self, dataset) -> None:
            captured["dataset"] = list(dataset)

    def _capture_dataset(job):
        captured["checkpoint_prefix"] = job.spec.storage.checkpoints.uriPrefix
        captured["weights_prefix"] = job.spec.storage.weights.uriPrefix
        captured["merged_step"] = job.status.progress.currentStep
        return []

    monkeypatch.setattr(controller_module, "HttpRolloutClient", _StubClient)
    monkeypatch.setattr(controller_module, "HttpRewardClient", _StubClient)
    monkeypatch.setattr(controller_module, "HttpLearnerClient", _StubClient)
    monkeypatch.setattr(controller_module, "HttpServingClient", _StubClient)
    monkeypatch.setattr(controller_module, "ControllerStatusWriter", _StubStatusWriter)
    monkeypatch.setattr(controller_module, "GRPOTrainer", _StubTrainer)
    monkeypatch.setattr(controller_module, "load_dataset", _capture_dataset)

    controller_module.run_controller(job_path)

    assert captured["checkpoint_prefix"] == "/var/lib/flashrl/shared/checkpoints"
    assert captured["weights_prefix"] == "/var/lib/flashrl/shared/weights"
    assert captured["merged_step"] == 11
    assert captured["resume_path"] == "/remote/checkpoints/step-00000010.pt"
