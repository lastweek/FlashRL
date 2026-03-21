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
from flashrl.platform.k8s.job_resources import render_job_resources


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


def _write_combined_config(
    tmp_path: Path,
    *,
    file_name: str = "config.yaml",
    job_name: str = "demo-job",
    namespace: str = "flashrl",
) -> Path:
    config_path = tmp_path / file_name
    config_path.write_text(
        f"""
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
    name: {job_name}
    namespace: {namespace}
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


def test_render_job_resources_builds_expected_workloads() -> None:
    """One FlashRLJob should render the shared runtime image across controller, rollout, and reward."""
    job = FlashRLJob.model_validate(_job_payload())
    rendered = render_job_resources(job)
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
    assert importlib.import_module("flashrl.platform.k8s.job_resources") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator.kube") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator.reconcile") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator.status") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator.scaling") is not None
    assert importlib.import_module("flashrl.platform.k8s.operator.recovery") is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.k8s.renderer")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.k8s.operator.store")
    assert importlib.import_module("flashrl.platform.runtime.cli") is not None
    runtime_pkg = importlib.import_module("flashrl.platform.runtime")
    assert runtime_pkg.PlatformShimController is not None
    assert runtime_pkg.PlatformShimRollout is not None
    assert runtime_pkg.PlatformShimReward is not None
    assert runtime_pkg.PlatformShimLearner is not None
    assert runtime_pkg.PlatformShimServing is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_base") is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_common") is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_controller") is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_rollout") is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_reward") is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_learner") is not None
    assert importlib.import_module("flashrl.platform.runtime.platform_shim_serving") is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.controller")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.rollout")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.reward")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.learner")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.serving")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.pod")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.common")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.platform.runtime.components")


def test_platform_docs_and_tests_no_longer_reference_flat_operator_files() -> None:
    """Docs and tests should point at the operator package, not the old flat files."""
    root = Path(__file__).resolve().parents[1]
    checked_paths = [
        root / "flashrl/platform/README.md",
        root / "tests/test_platform_operator.py",
    ]
    forbidden = [
        "k8s/operator.py",
        "k8s/status.py",
        "k8s/scaling.py",
        "k8s/recovery.py",
        "k8s/reconcile.py",
    ]

    for path in checked_paths:
        content = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in content, f"{path} still references {needle}"


def test_platform_architecture_doc_is_linked_from_readmes() -> None:
    """The platform architecture doc should be discoverable from the main docs."""
    root = Path(__file__).resolve().parents[1]
    architecture_doc = root / "docs/platform-architecture.md"
    root_readme = root / "README.md"
    platform_readme = root / "flashrl/platform/README.md"

    assert architecture_doc.exists()
    assert "docs/platform-architecture.md" in root_readme.read_text(encoding="utf-8")
    assert "../../docs/platform-architecture.md" in platform_readme.read_text(encoding="utf-8")


def test_platform_readme_matches_simplified_runtime_layout() -> None:
    """The platform README should not mention removed runtime shim modules."""
    platform_readme = Path(__file__).resolve().parents[1] / "flashrl/platform/README.md"
    content = platform_readme.read_text(encoding="utf-8")

    assert "runtime/common.py" not in content
    assert "runtime/components.py" not in content
    assert "k8s/renderer.py" not in content
    assert "k8s/job_resources.py" in content
    assert "runtime/controller.py" not in content
    assert "runtime/pod.py" not in content
    assert "runtime/platform_shim_controller.py" in content
    assert "runtime/platform_shim_common.py" in content
    assert "scripts/build_platform_images.sh" in content
    assert "scripts/run_platform_job.sh" in content
    assert "scripts/cleanup_platform.sh" in content
    assert "--image-env-file" in content
    assert "--profile" not in content
    assert "profiles:" not in content
    assert "IMAGE_MODE=local" in content
    assert "LOCAL_CLUSTER_TYPE=minikube" in content
    assert "LOCAL_CLUSTER_TYPE=kind" in content
    assert "LOCAL_CLUSTER_TYPE=docker-desktop" in content
    assert "flashrl/platform/dev/math-minikube.yaml" in content


def test_platform_bash_scripts_parse_cleanly() -> None:
    """The raw platform bash scripts should at least pass shell syntax checks."""
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts/build_platform_images.sh",
        root / "scripts/run_platform_job.sh",
        root / "scripts/cleanup_platform.sh",
    ]
    for script in scripts:
        assert script.exists()
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_platform_run_and_cleanup_scripts_log_commands_readably() -> None:
    """The run and cleanup scripts should keep curated command logging helpers."""
    root = Path(__file__).resolve().parents[1]
    run_script = (root / "scripts/run_platform_job.sh").read_text(encoding="utf-8")
    cleanup_script = (root / "scripts/cleanup_platform.sh").read_text(encoding="utf-8")

    for content in (run_script, cleanup_script):
        assert "log_info() {" in content
        assert "log_cmd() {" in content
        assert "run_cmd() {" in content
        assert "[cmd]" in content
        assert "[info]" in content

    assert "capture_cmd() {" in run_script
    assert "run_labeled_cmd() {" in run_script
    assert "run_cmd_stdout_to_file() {" in run_script
    assert "run_cmd_all_to_file() {" in run_script


def test_platform_architecture_doc_covers_per_pod_workflows() -> None:
    """The architecture doc should include per-pod init and execution workflows."""
    architecture_doc = Path(__file__).resolve().parents[1] / "docs/platform-architecture.md"
    content = architecture_doc.read_text(encoding="utf-8")

    assert "## What Platform Adds Per Pod" in content
    assert "PlatformShimController" in content
    assert "PlatformShimRollout" in content
    assert "PlatformShimReward" in content
    assert "PlatformShimLearner" in content
    assert "PlatformShimServing" in content
    assert "flashrl.platform.runtime.platform_shim_common" in content
    assert "flashrl.platform.runtime.platform_shim_controller" in content
    assert "flashrl.platform.runtime.platform_shim_rollout" in content
    assert "flashrl.platform.runtime.platform_shim_reward" in content
    assert "flashrl.platform.runtime.platform_shim_learner" in content
    assert "flashrl.platform.runtime.platform_shim_serving" in content
    for heading in (
        "## Inside Each Pod",
        "### Controller Pod",
        "### Rollout Pod",
        "### Reward Pod",
        "### Learner Pod",
        "### Serving Pod",
        "#### Init Workflow",
        "#### Execution Workflow",
    ):
        assert heading in content

    assert "platform runtime" in content
    assert "framework distributed" in content
    assert "Controller Container" in content
    assert "Rollout Container" in content
    assert "Reward Container" in content
    assert "Learner Container" in content
    assert "Serving Container" in content
    assert "container=controller" in content
    assert "container=rollout" in content
    assert "container=reward" in content
    assert "container=learner" in content
    assert "container=serving" in content
    assert "<job>-controller" in content
    assert "<job>-rollout" in content
    assert "<job>-reward" in content
    assert "<job>-learner" in content
    assert "<job>-serving" in content
    assert "load_mounted_job" in content
    assert "service_url_for" in content
    assert "storage_path_from_uri" in content
    assert "RolloutClient" in content
    assert "RewardClient" in content
    assert "LearnerClient" in content
    assert "ServingClient" in content
    assert "RolloutService" in content
    assert "RewardService" in content
    assert "LearnerService" in content
    assert "ServingService" in content
    assert "RemoteServingBackend" in content
    assert "http_common.py" in content
    assert "install_common_routes" in content
    assert "create_rollout_service_app" in content
    assert "create_reward_service_app" in content
    assert "create_learner_service_app" in content
    assert "create_serving_service_app" in content
    assert "flashrl.framework.rollout" in content
    assert "flashrl.framework.reward" in content
    assert "flashrl.framework.training" in content
    assert "flashrl.framework.serving" in content
    assert "GRPOTrainer" in content
    assert "/v1/rollout-batches" in content
    assert "/v1/reward-batches" in content
    assert "/v1/optimize-steps" in content
    assert "/v1/generate-grouped" in content
    assert "/v1/activate-weight-version" in content
    for legacy_name in (
        "HttpRolloutClient",
        "HttpRewardClient",
        "HttpLearnerClient",
        "HttpServingClient",
        "HttpServingBackend",
        "LocalLearnerClient",
        "LocalServingClient",
        "create_rollout_app",
        "create_reward_app",
        "create_learner_app",
        "create_serving_app",
        "flashrl.framework.distributed.rollout_service",
        "flashrl.framework.distributed.reward_service",
        "flashrl.framework.distributed.learner_service",
        "flashrl.framework.distributed.serving_service",
        "flashrl.framework.distributed.remote_serving_backend",
        "flashrl.platform.runtime.controller",
        "flashrl.platform.runtime.rollout",
        "flashrl.platform.runtime.reward",
        "flashrl.platform.runtime.learner",
        "flashrl.platform.runtime.serving",
        "flashrl.platform.runtime.pod",
    ):
        assert legacy_name not in content


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


def test_render_job_resources_include_rbac_pdb_and_component_commands() -> None:
    """Rendered children should include platform infrastructure and per-component commands."""
    job = FlashRLJob.model_validate(_job_payload())
    rendered = render_job_resources(job)

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
    assert container["command"] == ["flashrl", "serving"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert "lifecycle" in container
    assert not any(env["name"] == "FLASHRL_COMPONENT" for env in container["env"])
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


def test_flashrl_platform_render_cli_supports_output_file(tmp_path: Path) -> None:
    """Render should accept one explicit config file and one explicit output path."""
    config_path = _write_combined_config(
        tmp_path,
        file_name="config.e2e.yaml",
        job_name="demo-job-e2e",
        namespace="flashrl-e2e",
    )
    output_path = tmp_path / "job.yaml"

    exit_code = platform_main(
        [
            "platform",
            "render",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rendered = output_path.read_text(encoding="utf-8")
    assert "name: demo-job-e2e" in rendered
    assert "namespace: flashrl-e2e" in rendered
    assert "kind: FlashRLJob" in rendered


def test_flashrl_config_rejects_legacy_profiles_section(tmp_path: Path) -> None:
    """Top-level legacy profiles should fail with a direct migration error."""
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
profiles:
  minikube:
    framework:
      trainer:
        batch_size: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Top-level `profiles` is no longer supported"):
        FlashRLConfig.from_yaml(config_path)


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
    assert schema["type"] == "object"
    assert schema["required"] == ["spec"]
    assert schema["properties"]["apiVersion"]["type"] == "string"
    assert schema["properties"]["kind"]["type"] == "string"
    assert schema["properties"]["spec"]["type"] == "object"
    assert schema["properties"]["spec"]["x-kubernetes-preserve-unknown-fields"] is True
    assert schema["properties"]["status"]["type"] == "object"
    assert schema["properties"]["status"]["x-kubernetes-preserve-unknown-fields"] is True


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
