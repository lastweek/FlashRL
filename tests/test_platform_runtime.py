"""Tests for the platform controller and explicit pod-runtime entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flashrl.__main__ as root_cli_module
import flashrl.platform.runtime.cli as runtime_cli_module
import flashrl.platform.runtime.platform_shim_controller as controller_module
import flashrl.platform.runtime.platform_shim_learner as learner_runtime_module
import flashrl.platform.runtime.platform_shim_reward as reward_runtime_module
import flashrl.platform.runtime.platform_shim_rollout as rollout_runtime_module
import flashrl.platform.runtime.platform_shim_serving as serving_runtime_module


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


def test_run_controller_constructs_remote_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The controller runtime should instantiate GRPOTrainer with remote clients."""
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

    monkeypatch.setattr(controller_module, "RolloutClient", _StubClient)
    monkeypatch.setattr(controller_module, "RewardClient", _StubClient)
    monkeypatch.setattr(controller_module, "LearnerClient", _StubClient)
    monkeypatch.setattr(controller_module, "ServingClient", _StubClient)
    monkeypatch.setattr(controller_module, "FlashRLJobStatusWriter", _StubStatusWriter)
    monkeypatch.setattr(controller_module, "GRPOTrainer", _StubTrainer)
    monkeypatch.setattr(controller_module, "load_controller_dataset", lambda job: [])

    controller_module.run_controller(job_path)

    trainer_kwargs = captured["trainer_kwargs"]
    assert trainer_kwargs["rollout"].url.endswith("/demo-job-rollout.default.svc.cluster.local")
    assert trainer_kwargs["reward"].url.endswith("/demo-job-reward.default.svc.cluster.local")
    assert trainer_kwargs["learner"].url.endswith("/demo-job-learner-0.demo-job-learner.default.svc.cluster.local")
    assert trainer_kwargs["serving"].url.endswith("/demo-job-serving.default.svc.cluster.local")
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

    monkeypatch.setattr(controller_module, "RolloutClient", _StubClient)
    monkeypatch.setattr(controller_module, "RewardClient", _StubClient)
    monkeypatch.setattr(controller_module, "LearnerClient", _StubClient)
    monkeypatch.setattr(controller_module, "ServingClient", _StubClient)
    monkeypatch.setattr(controller_module, "FlashRLJobStatusWriter", _StubStatusWriter)
    monkeypatch.setattr(controller_module, "GRPOTrainer", _StubTrainer)
    monkeypatch.setattr(controller_module, "load_controller_dataset", _capture_dataset)

    controller_module.run_controller(job_path)

    assert captured["checkpoint_prefix"] == "/var/lib/flashrl/shared/checkpoints"
    assert captured["weights_prefix"] == "/var/lib/flashrl/shared/weights"
    assert captured["merged_step"] == 11
    assert captured["resume_path"] == "/remote/checkpoints/step-00000010.pt"


def test_top_level_cli_dispatches_direct_component_verbs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level CLI should route direct pod verbs into the runtime CLI."""
    captured: dict[str, object] = {}

    def _stub_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(runtime_cli_module, "main", _stub_main)

    exit_code = root_cli_module.main(["rollout", "--port", "9000"])

    assert exit_code == 0
    assert captured["argv"] == ["rollout", "--port", "9000"]


def test_top_level_cli_rejects_old_component_subtree(capsys: pytest.CaptureFixture[str]) -> None:
    """The removed `flashrl component ...` subtree should now fail clearly."""
    exit_code = root_cli_module.main(["component", "run", "rollout"])

    assert exit_code == 2
    assert "Unknown command: component" in capsys.readouterr().err


def test_runtime_cli_dispatches_to_explicit_role_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime CLI should only parse args and delegate to one PlatformShim."""
    captured: dict[str, object] = {}

    class _StubPlatformShimRollout:
        def run(self, *, host: str, port: int) -> int:
            captured["dispatch"] = ("rollout", host, port)
            return 0

    monkeypatch.setattr(runtime_cli_module, "PlatformShimRollout", _StubPlatformShimRollout)

    assert runtime_cli_module.main(["rollout", "--host", "127.0.0.1", "--port", "9000"]) == 0
    assert captured["dispatch"] == ("rollout", "127.0.0.1", 9000)


def test_platform_shim_rollout_builds_rollout_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The rollout shim should wire the rollout hook into the rollout service."""
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job_payload()), encoding="utf-8")
    monkeypatch.setenv("FLASHRL_JOB_CONFIG_PATH", str(job_path))
    monkeypatch.setenv("FLASHRL_JOB_NAME", "demo-job")
    monkeypatch.setenv("FLASHRL_NAMESPACE", "default")
    captured: dict[str, object] = {}

    class _StubServingClient:
        def __init__(self, url: str) -> None:
            captured["serving_url"] = url

    class _StubRolloutService:
        def __init__(self, generator) -> None:
            captured["generator"] = generator

    monkeypatch.setattr(rollout_runtime_module.runtime_support, "instantiate_hook", lambda binding: "rollout-impl")
    monkeypatch.setattr(rollout_runtime_module, "ServingClient", _StubServingClient)
    monkeypatch.setattr(rollout_runtime_module, "RemoteServingBackend", lambda **kwargs: {"backend": kwargs})
    monkeypatch.setattr(
        rollout_runtime_module,
        "build_rollout_generator",
        lambda **kwargs: captured.setdefault("rollout_builder", kwargs) or "generator",
    )
    monkeypatch.setattr(rollout_runtime_module, "RolloutService", _StubRolloutService)
    monkeypatch.setattr(
        rollout_runtime_module,
        "create_rollout_service_app",
        lambda service: {"app": "rollout", "service": service},
    )

    app = rollout_runtime_module.PlatformShimRollout().create_app()
    assert app["app"] == "rollout"
    assert str(captured["serving_url"]).endswith("/demo-job-serving.default.svc.cluster.local")
    assert captured["rollout_builder"]["rollout_fn"] == "rollout-impl"


def test_platform_shim_reward_builds_reward_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The reward shim should wire the reward hook into the reward service."""
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job_payload()), encoding="utf-8")
    monkeypatch.setenv("FLASHRL_JOB_CONFIG_PATH", str(job_path))
    captured: dict[str, object] = {}

    class _StubRewardService:
        def __init__(self, reward) -> None:
            captured["reward_impl"] = reward

    monkeypatch.setattr(reward_runtime_module.runtime_support, "instantiate_hook", lambda binding: "reward-hook")
    monkeypatch.setattr(reward_runtime_module, "UserDefinedReward", lambda **kwargs: {"reward": kwargs})
    monkeypatch.setattr(reward_runtime_module, "RewardService", _StubRewardService)
    monkeypatch.setattr(
        reward_runtime_module,
        "create_reward_service_app",
        lambda service: {"app": "reward", "service": service},
    )

    app = reward_runtime_module.PlatformShimReward().create_app()
    assert app["app"] == "reward"
    assert captured["reward_impl"]["reward"]["reward_fn"] == "reward-hook"


def test_platform_shim_learner_builds_learner_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The learner shim should build backends and publish weights to shared storage."""
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job_payload()), encoding="utf-8")
    monkeypatch.setenv("FLASHRL_JOB_CONFIG_PATH", str(job_path))
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        learner_runtime_module,
        "create_training_backend",
        lambda config, **kwargs: captured.setdefault(f"{kwargs['role']}_backend", config.backend) or f"{kwargs['role']}-backend",
    )
    monkeypatch.setattr(
        learner_runtime_module,
        "LearnerService",
        lambda actor_backend, reference_backend, **kwargs: captured.setdefault(
            "learner_service",
            {
                "actor_backend": actor_backend,
                "reference_backend": reference_backend,
                **kwargs,
            },
        )
        or "learner-client",
    )
    monkeypatch.setattr(
        learner_runtime_module,
        "create_learner_service_app",
        lambda service: {"app": "learner", "service": service},
    )

    app = learner_runtime_module.PlatformShimLearner().create_app()
    assert app["app"] == "learner"
    assert captured["actor_backend"] == "huggingface"
    assert str(captured["learner_service"]["publish_dir"]).endswith("/tmp/flashrl/weights")
    assert captured["learner_service"]["synchronize_serving"] is False


def test_platform_shim_serving_builds_serving_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The serving shim should build the serving backend and serving service app."""
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job_payload()), encoding="utf-8")
    monkeypatch.setenv("FLASHRL_JOB_CONFIG_PATH", str(job_path))
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        serving_runtime_module,
        "create_serving_backend",
        lambda config, **kwargs: captured.setdefault("serving_backend", {"backend": config.backend, **kwargs}) or "serving-backend",
    )
    monkeypatch.setattr(serving_runtime_module, "ServingService", lambda backend: {"backend": backend})
    monkeypatch.setattr(
        serving_runtime_module,
        "create_serving_service_app",
        lambda service: {"app": "serving", "service": service},
    )

    app = serving_runtime_module.PlatformShimServing().create_app()
    assert app["app"] == "serving"
    assert captured["serving_backend"]["backend"] == "huggingface"
    assert str(captured["serving_backend"]["log_dir"]).endswith("/tmp/flashrl/weights")


def test_platform_shim_modules_remove_old_free_function_surface() -> None:
    """The shim modules should expose classes, not the old free-function pod entrypoints."""
    assert not hasattr(controller_module, "create_controller_app")
    assert not hasattr(controller_module, "run_controller_pod")
    assert not hasattr(rollout_runtime_module, "create_rollout_pod_app")
    assert not hasattr(rollout_runtime_module, "run_rollout_pod")
    assert not hasattr(reward_runtime_module, "create_reward_pod_app")
    assert not hasattr(reward_runtime_module, "run_reward_pod")
    assert not hasattr(learner_runtime_module, "create_learner_pod_app")
    assert not hasattr(learner_runtime_module, "run_learner_pod")
    assert not hasattr(serving_runtime_module, "create_serving_pod_app")
    assert not hasattr(serving_runtime_module, "run_serving_pod")
