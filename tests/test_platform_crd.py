"""Tests for the FlashRL platform CRD, config translation, and rendered resources."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pytest

from flashrl.framework.config import FlashRLConfig, RunConfig
from flashrl.platform.cli import main as platform_main
from flashrl.platform.config import PlatformConfig, build_flashrl_job
from flashrl.platform.k8s.job import FlashRLJob, flashrljob_crd_manifest
from flashrl.platform.k8s.renderer import render_child_resources


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
            "dataset": {"type": "hook"},
            "images": {
                "runtime": "ghcr.io/flashrl/flashrl-runtime:latest",
                "serving": "ghcr.io/flashrl/flashrl-serving-vllm:latest",
                "training": "ghcr.io/flashrl/flashrl-training-fsdp:latest",
                "pullPolicy": "IfNotPresent",
            },
            "userCode": {
                "dataset": {"import": "tests.platform_hooks:build_dataset"},
                "rollout": {"import": "tests.platform_hooks:build_rollout"},
                "reward": {"import": "tests.platform_hooks:build_reward"},
            },
            "sharedStorage": {
                "enabled": True,
                "mountPath": "/var/lib/flashrl/shared",
                "checkpointsSubPath": "checkpoints",
                "weightsSubPath": "weights",
                "claim": {"size": "2Gi"},
            },
            "serving": {
                "replicas": {"min": 2, "max": 4},
            },
            "rollout": {
                "replicas": {"min": 2, "max": 8},
            },
            "reward": {
                "replicas": {"min": 1, "max": 4},
            },
            "storage": {
                "checkpoints": {"uriPrefix": "checkpoints"},
                "weights": {"uriPrefix": "weights"},
            },
        },
    }


def _write_combined_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
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
  hooks:
    rollout_fn:
      import: tests.platform_hooks:build_rollout
    reward_fn:
      import: tests.platform_hooks:build_reward
    dataset_fn:
      import: tests.platform_hooks:build_dataset
platform:
  job:
    name: demo-job
    namespace: flashrl
  images:
    runtime: ghcr.io/flashrl/flashrl-runtime:sha-deadbeef
    serving: ghcr.io/flashrl/flashrl-serving-vllm:sha-deadbeef
    training: ghcr.io/flashrl/flashrl-training-fsdp:sha-deadbeef
    pullPolicy: IfNotPresent
  sharedStorage:
    enabled: true
    mountPath: /var/lib/flashrl/shared
    checkpointsSubPath: checkpoints
    weightsSubPath: weights
    claim:
      size: 2Gi
  serving:
    replicas:
      min: 2
      max: 4
  rollout:
    replicas:
      min: 2
      max: 8
  reward:
    replicas:
      min: 1
      max: 4
  storage:
    checkpoints:
      uriPrefix: checkpoints
    weights:
      uriPrefix: weights
profiles:
  minikube:
    platform:
      job:
        name: demo-job-minikube
        namespace: flashrl-e2e
""",
        encoding="utf-8",
    )
    return config_path


def test_flashrl_job_rejects_local_serving_runtime_python() -> None:
    """Platform mode should reject local-only serving runtime configuration."""
    payload = _job_payload()
    payload["spec"]["framework"]["serving"]["runtime_python"] = "/tmp/python"
    with pytest.raises(ValueError, match="runtime_python"):
        FlashRLJob.model_validate(payload)


def test_build_flashrl_job_translates_run_and_platform_configs(tmp_path: Path) -> None:
    """Config-first platform submission should translate into one FlashRLJob."""
    config_path = _write_combined_config(tmp_path)
    job = build_flashrl_job(
        run_config=RunConfig.from_yaml(config_path),
        platform_config=PlatformConfig.from_yaml(config_path),
    )

    assert job.metadata["name"] == "demo-job"
    assert job.metadata["namespace"] == "flashrl"
    assert job.spec.images.runtime.endswith("sha-deadbeef")
    assert job.spec.images.pullPolicy == "IfNotPresent"
    assert job.spec.dataset.type == "hook"
    assert job.spec.userCode.rollout.import_path == "tests.platform_hooks:build_rollout"
    assert job.spec.serving.replicas is not None
    assert job.spec.serving.replicas.max == 4


def test_render_child_resources_builds_expected_workloads() -> None:
    """One FlashRLJob should render the shared runtime image across controller, rollout, and reward."""
    job = FlashRLJob.model_validate(_job_payload())
    rendered = render_child_resources(job)
    kinds = [item["kind"] for item in rendered]
    assert kinds.count("Deployment") == 4
    assert kinds.count("StatefulSet") == 1
    assert kinds.count("Service") == 5
    assert kinds.count("ConfigMap") == 1
    assert kinds.count("PersistentVolumeClaim") == 1

    deployments = {
        item["metadata"]["name"]: item
        for item in rendered
        if item["kind"] == "Deployment"
    }
    assert (
        deployments["demo-job-controller"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == "ghcr.io/flashrl/flashrl-runtime:latest"
    )
    assert (
        deployments["demo-job-rollout"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == "ghcr.io/flashrl/flashrl-runtime:latest"
    )
    assert (
        deployments["demo-job-reward"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == "ghcr.io/flashrl/flashrl-runtime:latest"
    )
    assert (
        deployments["demo-job-serving"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == "ghcr.io/flashrl/flashrl-serving-vllm:latest"
    )
    assert (
        deployments["demo-job-controller"]["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"]
        == "IfNotPresent"
    )


def test_platform_module_surface_imports_cleanly() -> None:
    """The structured platform modules should remain importable by responsibility."""
    assert importlib.import_module("flashrl.platform.config") is not None
    assert importlib.import_module("flashrl.platform.k8s.job") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator") is not None
    assert importlib.import_module("flashrl.platform.runtime.controller") is not None
    assert importlib.import_module("flashrl.platform.runtime.cli") is not None


def test_platform_cli_no_longer_supports_render_operator() -> None:
    """The debug-only operator render subcommand should stay removed."""
    with pytest.raises(SystemExit) as exc_info:
        platform_main(["platform", "render-operator"])
    assert exc_info.value.code == 2


def test_flashrl_job_defaults_platform_policy_knobs() -> None:
    """Platform mode should fill autoscaling and failure defaults for the operator."""
    job = FlashRLJob.model_validate(_job_payload())

    assert job.spec.controller.autoscaling is not None
    assert job.spec.controller.autoscaling.enabled is False
    assert job.spec.learner.failurePolicy is not None
    assert job.spec.learner.failurePolicy.mode == "restart-workload"
    assert job.spec.serving.autoscaling is not None
    assert job.spec.serving.autoscaling.enabled is True
    assert job.spec.rollout.failurePolicy is not None
    assert job.spec.rollout.failurePolicy.mode == "replace-pod"


def test_render_child_resources_include_rbac_pdb_and_component_commands() -> None:
    """Rendered children should include platform infrastructure and per-component commands."""
    job = FlashRLJob.model_validate(_job_payload())
    rendered = render_child_resources(job)

    kinds = [item["kind"] for item in rendered]
    assert kinds.count("PersistentVolumeClaim") == 1
    assert kinds.count("ServiceAccount") == 1
    assert kinds.count("Role") == 1
    assert kinds.count("RoleBinding") == 1
    assert kinds.count("PodDisruptionBudget") == 3
    assert "HorizontalPodAutoscaler" not in kinds

    learner_service = next(
        item
        for item in rendered
        if item["kind"] == "Service" and item["metadata"]["name"] == "demo-job-learner"
    )
    assert learner_service["spec"]["clusterIP"] == "None"
    assert learner_service["spec"]["publishNotReadyAddresses"] is True

    serving_deployment = next(
        item
        for item in rendered
        if item["kind"] == "Deployment" and item["metadata"]["name"] == "demo-job-serving"
    )
    container = serving_deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["flashrl", "component", "run", "serving-vllm"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert "lifecycle" in container
    assert any(
        mount["name"] == "shared-storage" and mount["mountPath"] == "/var/lib/flashrl/shared"
        for mount in container["volumeMounts"]
    )

    config_map = next(
        item
        for item in rendered
        if item["kind"] == "ConfigMap" and item["metadata"]["name"] == "demo-job-config"
    )
    assert '"/var/lib/flashrl/shared/checkpoints"' in config_map["data"]["job.json"]
    assert '"/var/lib/flashrl/shared/weights"' in config_map["data"]["job.json"]


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
    type: hook
  images:
    runtime: ghcr.io/flashrl/flashrl-runtime:latest
    serving: ghcr.io/flashrl/flashrl-serving-vllm:latest
    training: ghcr.io/flashrl/flashrl-training-fsdp:latest
    pullPolicy: IfNotPresent
  userCode:
    dataset:
      import: tests.platform_hooks:build_dataset
    rollout:
      import: tests.platform_hooks:build_rollout
    reward:
      import: tests.platform_hooks:build_reward
  sharedStorage:
    enabled: true
    mountPath: /var/lib/flashrl/shared
    claim:
      size: 2Gi
  serving: {}
  rollout: {}
  reward: {}
  storage:
    checkpoints:
      uriPrefix: checkpoints
    weights:
      uriPrefix: weights
""",
        encoding="utf-8",
    )

    exit_code = platform_main(["platform", "submit", "--file", str(job_path), "--render-only"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"kind": "Deployment"' in output
    assert '"kind": "StatefulSet"' in output


def test_flashrl_platform_render_cli_builds_job_from_single_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The primary config-first CLI should render one FlashRLJob YAML payload."""
    config_path = _write_combined_config(tmp_path)
    exit_code = platform_main(
        [
            "platform",
            "render",
            "--config",
            str(config_path),
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "kind: FlashRLJob" in output
    assert "runtime:" in output
    assert "userCode:" in output


def test_flashrl_platform_render_cli_supports_profiles_and_output_file(tmp_path: Path) -> None:
    """Render should accept one config plus one optional profile and file output."""
    config_path = _write_combined_config(tmp_path)
    output_path = tmp_path / "job.yaml"

    exit_code = platform_main(
        [
            "platform",
            "render",
            "--config",
            str(config_path),
            "--profile",
            "minikube",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rendered = output_path.read_text(encoding="utf-8")
    assert "name: demo-job-minikube" in rendered
    assert "namespace: flashrl-e2e" in rendered
    assert "kind: FlashRLJob" in rendered


def test_flashrl_config_profiles_merge_platform_overrides(tmp_path: Path) -> None:
    """The combined config loader should deep-merge the selected profile."""
    config_path = _write_combined_config(tmp_path)
    merged = FlashRLConfig.from_yaml(config_path, profile="minikube")

    assert merged.framework.actor.model_name == "fake/model"
    assert merged.platform is not None
    assert merged.platform["job"]["name"] == "demo-job-minikube"
    assert merged.platform["job"]["namespace"] == "flashrl-e2e"


def test_flashrl_platform_render_requires_platform_section(tmp_path: Path) -> None:
    """Cluster rendering should fail clearly when the selected config has no platform section."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
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
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top-level `platform:` section"):
        platform_main(["platform", "render", "--config", str(config_path)])


def test_flashrl_platform_render_rejects_legacy_platform_config_flag(tmp_path: Path) -> None:
    """The old two-file platform CLI should no longer be accepted."""
    config_path = _write_combined_config(tmp_path)
    with pytest.raises(SystemExit):
        platform_main(
            [
                "platform",
                "render",
                "--config",
                str(config_path),
                "--platform-config",
                "legacy.yaml",
            ]
        )


def test_flashrljob_crd_manifest_exposes_expected_kind() -> None:
    """The generated CRD manifest should expose the expected resource kind."""
    manifest = flashrljob_crd_manifest()
    assert manifest["kind"] == "CustomResourceDefinition"
    assert manifest["spec"]["names"]["kind"] == "FlashRLJob"
    schema = manifest["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    spec_properties = schema["properties"]["spec"]["properties"]
    assert "images" in spec_properties
    assert "userCode" in spec_properties
    assert "sharedStorage" in spec_properties
    assert "autoscaling" in spec_properties["serving"]["properties"]
    assert "failurePolicy" in spec_properties["rollout"]["properties"]


def test_platform_k8s_install_assets_exist() -> None:
    """The platform package should ship raw Kubernetes install manifests."""
    assert Path("flashrl/platform/k8s/namespace.yaml").exists()
    assert Path("flashrl/platform/k8s/job-crd.yaml").exists()
    assert Path("flashrl/platform/k8s/operator-rbac.yaml").exists()
    assert Path("flashrl/platform/k8s/operator.yaml").exists()


def test_checked_in_job_crd_matches_generator_output() -> None:
    """The committed CRD artifact should exactly match the generator output."""
    repo_root = Path(__file__).resolve().parents[1]
    generator = repo_root / "flashrl/platform/k8s/job-crd-gen.py"
    checked_in = (repo_root / "flashrl/platform/k8s/job-crd.yaml").read_text(encoding="utf-8")
    generated = subprocess.check_output(
        [sys.executable, "-c", (
            "import importlib.util, pathlib; "
            "path = pathlib.Path(r'%s'); "
            "spec = importlib.util.spec_from_file_location('job_crd_gen', path); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "print(module.render_job_crd_yaml(), end='')"
        ) % str(generator)],
        text=True,
        cwd=str(repo_root),
    )
    assert checked_in == generated, (
        "flashrl/platform/k8s/job-crd.yaml is out of date. "
        "Run `python3 flashrl/platform/k8s/job-crd-gen.py` and commit the regenerated file."
    )
