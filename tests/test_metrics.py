"""Tests for Prometheus/Grafana metric stack integration."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import textwrap
from types import SimpleNamespace
import warnings

import pytest
from pydantic import ValidationError
import torch
import torch.nn.functional as F
import yaml

import flashrl.framework.flashrl as flashrl_module
import flashrl.framework.metrics as metrics_module
from flashrl.framework import FlashRL, GrpoConfig, LoggingConfig, MetricsConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.metrics import PrometheusMetricsSink


class FakeTokenizer:
    """A tiny tokenizer for deterministic unit tests."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        encoded = [self._encode(text, max_length=max_length) for text in texts]
        width = max(len(tokens) for tokens in encoded)

        input_ids = []
        attention_mask = []
        for tokens in encoded:
            padding_tokens = [0] * (width - len(tokens))
            if self.padding_side == "left":
                input_ids.append(padding_tokens + tokens)
                attention_mask.append([0] * len(padding_tokens) + [1] * len(tokens))
            else:
                input_ids.append(tokens + padding_tokens)
                attention_mask.append([1] * len(tokens) + [0] * len(padding_tokens))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _encode(self, text: str, *, max_length: int) -> list[int]:
        tokens = [((ord(char) % (self.vocab_size - 1)) + 1) for char in text[:max_length]]
        return tokens or [1]


class FakeCausalLM(torch.nn.Module):
    """A tiny causal LM with trainable logits."""

    def __init__(self, vocab_size: int = 32, bias_shift: float = 0.0) -> None:
        super().__init__()
        base = torch.linspace(-0.2, 0.2, vocab_size)
        self.logit_bias = torch.nn.Parameter(base + bias_shift)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        vocab_size = self.logit_bias.shape[0]
        logits = self.logit_bias.view(1, 1, vocab_size).expand(
            input_ids.shape[0],
            input_ids.shape[1],
            vocab_size,
        )
        token_signal = F.one_hot(input_ids % vocab_size, num_classes=vocab_size).float()
        return SimpleNamespace(logits=logits + 0.05 * token_signal)


class FakeActor:
    """Actor wrapper that matches the current backend contract."""

    def __init__(self, bias_shift: float = 0.0) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_length=128)
        self.tokenizer = FakeTokenizer()
        self.model = FakeCausalLM(self.tokenizer.vocab_size, bias_shift=bias_shift)

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        del kwargs
        return [f"generated::{prompt}" for prompt in prompts]

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


class FakeTrainingBackend:
    """Training backend used for metrics tests."""

    last_config = None

    def __init__(self, config, learning_rate: float = 1e-5) -> None:
        type(self).last_config = config
        self.actor = FakeActor(bias_shift=0.25)
        self.actor.train()
        self.optimizer = torch.optim.SGD(self.actor.model.parameters(), lr=learning_rate)

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.actor.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.actor.model.load_state_dict(torch.load(path, weights_only=False))

    def sync_weights_to(self, serving_backend) -> None:
        serving_backend.actor.model.load_state_dict(self.actor.model.state_dict())


class FakeServingBackend:
    """Serving backend used for metrics tests."""

    last_config = None

    def __init__(self, config) -> None:
        type(self).last_config = config
        self.actor = FakeActor(bias_shift=0.1)
        self.actor.eval()


class FakeReferenceModel:
    """Reference model used for KL computation in tests."""

    last_config = None

    def __init__(self, config) -> None:
        type(self).last_config = config
        self.device = torch.device("cpu")
        self.model = FakeCausalLM(bias_shift=0.0)


def make_rollout_fn(response_suffix: str = "response", repeat: int = 1):
    """Create a rollout function with deterministic responses."""

    def rollout_fn(prompts: list[Prompt], actor) -> list[RolloutOutput]:
        del actor
        outputs = []
        for prompt in prompts:
            response = f"{response_suffix} " + ("detail " * repeat) + prompt.text
            outputs.append(
                RolloutOutput(
                    text=response,
                    log_prob=0.0,
                    conversation=Conversation(
                        messages=[
                            Message(role="user", content=prompt.text),
                            Message(role="assistant", content=response),
                        ]
                    ),
                )
            )
        return outputs

    return rollout_fn


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Reward longer responses slightly more."""
    return RewardOutput(reward=len(rollout.text) / 50.0)


def patch_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch FlashRL to use fake local backends."""
    FakeTrainingBackend.last_config = None
    FakeServingBackend.last_config = None
    FakeReferenceModel.last_config = None
    monkeypatch.setattr(flashrl_module, "TrainingBackend", FakeTrainingBackend)
    monkeypatch.setattr(flashrl_module, "ServingBackend", FakeServingBackend)
    monkeypatch.setattr(flashrl_module, "ReferenceModel", FakeReferenceModel)


def sample_value(registry, name: str) -> tuple[float, dict[str, str]]:
    """Read one metric sample value and labels from a registry."""
    for metric in registry.collect():
        for sample in metric.samples:
            if sample.name == name:
                return float(sample.value), dict(sample.labels)
    raise AssertionError(f"Metric sample not found: {name}")


def write_executable(path: Path, content: str) -> None:
    """Create an executable shell script for command stubbing."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def symlink_required_tool(bin_dir: Path, tool_name: str) -> None:
    """Expose a host tool inside a controlled PATH for shell tests."""
    tool_path = shutil.which(tool_name)
    if tool_path is None:
        pytest.skip(f"required tool not available for shell test: {tool_name}")
    (bin_dir / tool_name).symlink_to(tool_path)


def make_controlled_shell_env(
    tmp_path: Path,
    *,
    docker_script: str | None = None,
    curl_script: str | None = None,
    timeout_seconds: str = "60",
    interval_seconds: str = "1",
) -> dict[str, str]:
    """Build a PATH with stubbed docker/curl commands for dev.sh tests."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    symlink_required_tool(bin_dir, "dirname")
    symlink_required_tool(bin_dir, "sleep")

    if docker_script is not None:
        write_executable(bin_dir / "docker", docker_script)
    if curl_script is not None:
        write_executable(bin_dir / "curl", curl_script)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["FLASHRL_METRICS_READY_TIMEOUT_SECONDS"] = timeout_seconds
    env["FLASHRL_METRICS_READY_INTERVAL_SECONDS"] = interval_seconds
    env["FLASHRL_TEST_DOCKER_LOG"] = str(tmp_path / "docker.log")
    env["FLASHRL_TEST_CURL_LOG"] = str(tmp_path / "curl.log")
    return env


def run_dev_sh(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run dev.sh with a controlled environment."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")

    return subprocess.run(
        [bash, "dev.sh", *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def create_yaml_hook_module(tmp_path: Path) -> str:
    """Create a temporary importable module that exposes YAML hook functions."""
    package_dir = tmp_path / "yaml_hooks"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "demo.py").write_text(
        textwrap.dedent(
            """
            from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput


            DATASET_TEXTS = ["yaml prompt 0", "yaml prompt 1", "yaml prompt 2", "yaml prompt 3"]


            def dataset_fn():
                return [Prompt(text=text) for text in DATASET_TEXTS]


            def rollout_fn(prompts, actor):
                texts = actor.generate([prompt.text for prompt in prompts])
                outputs = []
                for prompt, text in zip(prompts, texts, strict=True):
                    outputs.append(
                        RolloutOutput(
                            text=text,
                            log_prob=0.0,
                            conversation=Conversation(
                                messages=[
                                    Message(role="user", content=prompt.text),
                                    Message(role="assistant", content=text),
                                ]
                            ),
                        )
                    )
                return outputs


            def reward_fn(rollout):
                return RewardOutput(reward=len(rollout.text) / 100.0)
            """
        ),
        encoding="utf-8",
    )
    return "yaml_hooks.demo"


def write_yaml_run_config(
    tmp_path: Path,
    *,
    hook_module: str,
    log_dir: Path,
    metrics_enabled: bool = False,
    common_dtype: str | None = None,
    serving_num_threads: int = 3,
    training_num_threads: int | None = 1,
    group_size: int = 2,
    clip_ratio: float = 0.2,
    kl_coefficient: float = 0.0,
) -> Path:
    """Write a temporary YAML config for FlashRL.from_yaml tests."""
    config_path = tmp_path / "run.yaml"
    training_section = {
        "learning_rate": 1.0e-5,
        "batch_size": 2,
        "max_epochs": 1,
    }
    if training_num_threads is not None:
        training_section["num_threads"] = training_num_threads

    serving_section: dict[str, int] = {}
    if serving_num_threads is not None:
        serving_section["num_threads"] = serving_num_threads

    common_section: dict[str, object] = {"model_name": "fake/model"}
    if common_dtype is not None:
        common_section["dtype"] = common_dtype

    config_path.write_text(
        yaml.safe_dump(
            {
                "common": common_section,
                "training": training_section,
                "serving": serving_section,
                "grpo": {
                    "group_size": group_size,
                    "clip_ratio": clip_ratio,
                    "kl_coefficient": kl_coefficient,
                    "max_new_tokens": 32,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 0,
                    "do_sample": True,
                },
                "logging": {
                    "log_dir": str(log_dir),
                    "console": False,
                    "file": True,
                },
                "metrics": {
                    "enabled": metrics_enabled,
                    "pushgateway_url": "http://localhost:9091",
                    "job_name": "flashrl-test",
                },
                "runtime": {
                    "reference_enabled": False,
                },
                "hooks": {
                    "rollout_fn": f"{hook_module}:rollout_fn",
                    "reward_fn": f"{hook_module}:reward_fn",
                    "dataset_fn": f"{hook_module}:dataset_fn",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_prometheus_metrics_sink_creates_expected_metrics_and_pushes_once() -> None:
    """The sink should own the six expected gauges and push the registry."""
    pushes: list[tuple[str, str]] = []

    sink = PrometheusMetricsSink(
        MetricsConfig(
            enabled=True,
            pushgateway_url="http://localhost:9091",
            job_name="flashrl-test",
        ),
        model_name="fake/model",
        push_fn=lambda url, job, registry: pushes.append((url, job)),
    )

    sink.observe_stage("rollout", 1.25)
    sink.observe_stage("reward", 0.5)
    sink.observe_step(
        loss=0.75,
        reward_mean=1.5,
        kl_mean=0.25,
        step_duration_seconds=2.0,
    )
    sink.push()

    assert {metric.name for metric in sink.registry.collect()} == {
        "flashrl_train_loss",
        "flashrl_reward_mean",
        "flashrl_kl_mean",
        "flashrl_rollout_latency_seconds",
        "flashrl_reward_latency_seconds",
        "flashrl_step_duration_seconds",
    }
    assert sample_value(sink.registry, "flashrl_train_loss")[0] == pytest.approx(0.75)
    assert sample_value(sink.registry, "flashrl_reward_mean")[0] == pytest.approx(1.5)
    assert sample_value(sink.registry, "flashrl_kl_mean")[0] == pytest.approx(0.25)
    assert sample_value(sink.registry, "flashrl_rollout_latency_seconds")[0] == pytest.approx(1.25)
    assert sample_value(sink.registry, "flashrl_reward_latency_seconds")[0] == pytest.approx(0.5)
    step_duration, labels = sample_value(sink.registry, "flashrl_step_duration_seconds")
    assert step_duration == pytest.approx(2.0)
    assert labels == {
        "model": "fake/model",
        "algorithm": "grpo",
        "runtime": "framework_local",
    }
    assert pushes == [("http://localhost:9091", "flashrl-test")]


def test_metrics_config_defaults_enabled() -> None:
    """Metrics should default to enabled for local runs."""
    assert MetricsConfig().enabled is True


def test_grpo_config_rejects_group_size_below_two() -> None:
    """Real GRPO requires at least two samples per prompt group."""
    with pytest.raises(ValidationError, match="group_size"):
        GrpoConfig(group_size=1)


def test_prometheus_metrics_sink_warns_once_and_continues_on_push_failure() -> None:
    """Push failures should not raise and should warn only once per sink."""
    attempts: list[tuple[str, str]] = []

    def failing_push(url: str, job: str, registry) -> None:
        del registry
        attempts.append((url, job))
        raise OSError("connection refused")

    sink = PrometheusMetricsSink(
        MetricsConfig(
            enabled=True,
            pushgateway_url="http://localhost:9091",
            job_name="flashrl-test",
        ),
        model_name="fake/model",
        push_fn=failing_push,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink.push()
        sink.push()

    assert len(attempts) == 2
    assert len(caught) == 1
    assert "best-effort mode" in str(caught[0].message)


def test_flashrl_metrics_push_process_lifetime_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metrics should update and push across consecutive runs with one sink instance."""
    pushes: list[tuple[str, str]] = []

    def fake_push(url: str, job: str, registry) -> None:
        pushes.append((url, job))

    monkeypatch.setattr(metrics_module, "push_to_gateway", fake_push)
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="metrics", repeat=4),
        reward_fn=reward_fn,
        batch_size=4,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
        metrics_config=MetricsConfig(
            enabled=True,
            pushgateway_url="http://localhost:9091",
            job_name="flashrl-test",
        ),
    )
    assert trainer._metrics_sink is not None
    sink_id = id(trainer._metrics_sink)

    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]
    trainer.train(dataset)
    trainer.train(dataset)

    assert id(trainer._metrics_sink) == sink_id
    assert len(pushes) == 4
    assert all(push == ("http://localhost:9091", "flashrl-test") for push in pushes)

    train_loss, _ = sample_value(trainer._metrics_sink.registry, "flashrl_train_loss")
    reward_mean, labels = sample_value(trainer._metrics_sink.registry, "flashrl_reward_mean")
    kl_mean, _ = sample_value(trainer._metrics_sink.registry, "flashrl_kl_mean")
    rollout_latency, _ = sample_value(trainer._metrics_sink.registry, "flashrl_rollout_latency_seconds")
    reward_latency, _ = sample_value(trainer._metrics_sink.registry, "flashrl_reward_latency_seconds")
    step_duration, _ = sample_value(trainer._metrics_sink.registry, "flashrl_step_duration_seconds")

    assert isinstance(train_loss, float)
    assert reward_mean > 0.0
    assert kl_mean == pytest.approx(0.0)
    assert rollout_latency >= 0.0
    assert reward_latency >= 0.0
    assert step_duration > 0.0
    assert labels == {
        "model": "fake/model",
        "algorithm": "grpo",
        "runtime": "framework_local",
    }


def test_flashrl_default_metrics_push_failure_does_not_fail_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default-on metrics should warn and continue when Pushgateway is unavailable."""
    def failing_push(url: str, job: str, registry) -> None:
        del url, job, registry
        raise OSError("pushgateway offline")

    monkeypatch.setattr(metrics_module, "push_to_gateway", failing_push)
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="warn", repeat=2),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )

    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]
    with pytest.warns(RuntimeWarning, match="best-effort mode"):
        trainer.train(dataset)

    assert trainer._metrics_sink is not None
    assert (trainer._run_logger.run_dir / "events.jsonl").exists()


def test_flashrl_rejects_batch_size_not_divisible_by_group_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Grouped GRPO should fail fast if the sampled batch cannot be split into full groups."""
    patch_backends(monkeypatch)

    with pytest.raises(ValueError, match="divisible by grpo.group_size"):
        FlashRL(
            model="fake/model",
            rollout_fn=make_rollout_fn(response_suffix="invalid", repeat=2),
            reward_fn=reward_fn,
            batch_size=3,
            max_epochs=1,
            grpo_config=GrpoConfig(group_size=2),
            logging_config=LoggingConfig(
                log_dir=tmp_path,
                console=False,
                file=True,
            ),
            metrics_config=MetricsConfig(enabled=False),
        )


def test_flashrl_from_yaml_loads_hooks_and_supports_dataset_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FlashRL.from_yaml should build from import-string hooks and train without an explicit dataset."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_yaml_run_config(
        tmp_path,
        hook_module=hook_module,
        log_dir=tmp_path / "yaml-logs",
    )

    trainer = FlashRL.from_yaml(config_path)
    trainer.train()

    assert trainer._dataset_loader is not None
    assert trainer.metrics_config.enabled is False
    assert trainer.serving_config.num_threads == 3
    assert trainer.training_model_config.num_threads == 1
    assert trainer.training_model_config.model_name == "fake/model"
    assert trainer.grpo_config.group_size == 2
    assert trainer.grpo_config.clip_ratio == pytest.approx(0.2)
    assert trainer._run_logger.run_dir.exists()


def test_flashrl_from_yaml_common_defaults_flow_into_training_and_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """common model defaults should flow into both resolved model copies."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_yaml_run_config(
        tmp_path,
        hook_module=hook_module,
        log_dir=tmp_path / "common-logs",
        common_dtype="float16",
        training_num_threads=5,
        serving_num_threads=3,
    )

    trainer = FlashRL.from_yaml(config_path)

    assert trainer.training_model_config.model_name == "fake/model"
    assert trainer.training_model_config.dtype == "float16"
    assert trainer.training_model_config.num_threads == 5
    assert trainer.serving_config.model_name == "fake/model"
    assert trainer.serving_config.dtype == "float16"
    assert trainer.serving_config.num_threads == 3


def test_flashrl_from_yaml_section_overrides_beat_common_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """training and serving should be able to override common defaults independently."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "override.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "common": {
                    "model_name": "fake/model",
                    "dtype": "float16",
                },
                "training": {
                    "dtype": "bfloat16",
                    "num_threads": 5,
                    "batch_size": 2,
                    "max_epochs": 1,
                },
                "serving": {
                    "dtype": "float32",
                    "num_threads": 3,
                },
                "grpo": {
                    "group_size": 2,
                    "clip_ratio": 0.2,
                    "kl_coefficient": 0.0,
                    "max_new_tokens": 32,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 0,
                    "do_sample": True,
                },
                "logging": {
                    "log_dir": str(tmp_path / "override-logs"),
                    "console": False,
                    "file": True,
                },
                "metrics": {
                    "enabled": False,
                },
                "runtime": {
                    "reference_enabled": False,
                },
                "hooks": {
                    "rollout_fn": f"{hook_module}:rollout_fn",
                    "reward_fn": f"{hook_module}:reward_fn",
                    "dataset_fn": f"{hook_module}:dataset_fn",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    trainer = FlashRL.from_yaml(config_path)

    assert trainer.training_model_config.dtype == "bfloat16"
    assert trainer.training_model_config.num_threads == 5
    assert trainer.serving_config.dtype == "float32"
    assert trainer.serving_config.num_threads == 3
    assert FakeTrainingBackend.last_config.dtype == "bfloat16"
    assert FakeTrainingBackend.last_config.num_threads == 5
    assert FakeServingBackend.last_config.dtype == "float32"
    assert FakeServingBackend.last_config.num_threads == 3


def test_flashrl_from_yaml_reference_model_uses_resolved_training_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The reference model should derive from training-side config plus runtime override."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "reference.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            common:
              model_name: fake/model
            training:
              device: cpu
              num_threads: 5
              batch_size: 2
              max_epochs: 1
            serving: {{}}
            grpo:
              group_size: 2
              clip_ratio: 0.2
              kl_coefficient: 0.0
            logging:
              log_dir: {tmp_path / "reference-logs"}
              console: false
              file: true
            metrics:
              enabled: false
            runtime:
              reference_enabled: true
              reference_device: reference-device
            hooks:
              rollout_fn: {hook_module}:rollout_fn
              reward_fn: {hook_module}:reward_fn
              dataset_fn: {hook_module}:dataset_fn
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    trainer = FlashRL.from_yaml(config_path)

    assert trainer.reference_enabled is True
    assert FakeReferenceModel.last_config is not None
    assert FakeReferenceModel.last_config.model_name == "fake/model"
    assert FakeReferenceModel.last_config.num_threads == 5
    assert FakeReferenceModel.last_config.device == "reference-device"


def test_flashrl_from_yaml_requires_model_name_after_common_and_section_merge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both training and serving must resolve a model name after merge."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "missing-model.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            training:
              batch_size: 2
              max_epochs: 1
            serving: {{}}
            grpo:
              group_size: 2
              clip_ratio: 0.2
              kl_coefficient: 0.0
            logging:
              log_dir: {tmp_path / "missing-model-logs"}
              console: false
              file: true
            metrics:
              enabled: false
            runtime:
              reference_enabled: false
            hooks:
              rollout_fn: {hook_module}:rollout_fn
              reward_fn: {hook_module}:reward_fn
              dataset_fn: {hook_module}:dataset_fn
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="common.model_name"):
        FlashRL.from_yaml(config_path)


def test_flashrl_from_yaml_rejects_loop_fields_under_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """common should only allow shared model defaults, not training loop fields."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "invalid-common.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            common:
              model_name: fake/model
              batch_size: 4
            training:
              batch_size: 2
              max_epochs: 1
            serving: {{}}
            grpo:
              group_size: 2
              clip_ratio: 0.2
              kl_coefficient: 0.0
            logging:
              log_dir: {tmp_path / "invalid-common-logs"}
              console: false
              file: true
            metrics:
              enabled: false
            runtime:
              reference_enabled: false
            hooks:
              rollout_fn: {hook_module}:rollout_fn
              reward_fn: {hook_module}:reward_fn
              dataset_fn: {hook_module}:dataset_fn
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="batch_size"):
        FlashRL.from_yaml(config_path)


def test_flashrl_from_yaml_rejects_num_threads_under_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """num_threads is runtime-local and must live under training or serving, not common."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "invalid-common-num-threads.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            common:
              model_name: fake/model
              num_threads: 4
            training:
              batch_size: 2
              max_epochs: 1
              num_threads: 2
            serving:
              num_threads: 1
            grpo:
              group_size: 2
              clip_ratio: 0.2
              kl_coefficient: 0.0
            logging:
              log_dir: {tmp_path / "invalid-common-num-threads-logs"}
              console: false
              file: true
            metrics:
              enabled: false
            runtime:
              reference_enabled: false
            hooks:
              rollout_fn: {hook_module}:rollout_fn
              reward_fn: {hook_module}:reward_fn
              dataset_fn: {hook_module}:dataset_fn
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="num_threads"):
        FlashRL.from_yaml(config_path)


def test_flashrl_train_requires_dataset_without_yaml_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plain Python construction should still require an explicit dataset."""
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="python", repeat=2),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
        metrics_config=MetricsConfig(enabled=False),
    )

    with pytest.raises(ValueError, match="requires a dataset"):
        trainer.train()

    dataset = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    trainer.train(dataset)
    assert trainer._run_logger.run_dir.exists()


def test_flashrl_cli_main_runs_yaml_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The module CLI should run a YAML config in-process."""
    patch_backends(monkeypatch)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_yaml_run_config(
        tmp_path,
        hook_module=hook_module,
        log_dir=tmp_path / "cli-logs",
    )

    result = flashrl_module.main(["--config", str(config_path)])

    assert result == 0


def test_reasoning_example_yaml_runs_with_fake_backends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The committed reasoning example YAML should run through FlashRL.from_yaml."""
    patch_backends(monkeypatch)
    monkeypatch.setattr(metrics_module, "push_to_gateway", lambda *args, **kwargs: None)

    trainer = FlashRL.from_yaml("examples/reasoning/config.yaml")
    trainer.logging_config = LoggingConfig(
        log_dir=tmp_path,
        console=False,
        file=True,
    )
    trainer.metrics_config = MetricsConfig(
        enabled=True,
        pushgateway_url="http://localhost:9091",
        job_name="flashrl-test",
    )
    trainer._metrics_sink = PrometheusMetricsSink(
        trainer.metrics_config,
        model_name=trainer.training_model_config.model_name,
    )
    assert trainer._trainer is not None
    trainer._trainer.metrics_sink = trainer._metrics_sink

    trainer.train()

    assert trainer._run_logger.run_dir.exists()


def test_observability_stack_files_and_docs_exist() -> None:
    """The local compose stack and quickstart docs should be committed."""
    assert Path("metric/docker-compose.yml").exists()
    assert Path("metric/prometheus/prometheus.yml").exists()
    assert Path("metric/grafana/provisioning/datasources/prometheus.yml").exists()
    assert Path("metric/grafana/provisioning/dashboards/dashboards.yml").exists()
    assert Path("metric/grafana/dashboards/flashrl-v1.json").exists()
    assert "observability/docker-compose.yml" not in Path("examples/README.md").read_text(
        encoding="utf-8"
    )
    assert Path("examples/__init__.py").exists()
    assert Path("examples/reasoning/__init__.py").exists()
    assert Path("examples/reasoning/train.py").exists()
    assert Path("examples/reasoning/config.yaml").exists()

    docs = Path("examples/README.md").read_text(encoding="utf-8")
    assert "./dev.sh metrics up" in docs
    assert "endpoint-ready before reporting success" in docs
    assert "python3 -m examples.reasoning.train" in docs
    assert "python3 -m flashrl.framework.flashrl --config examples/reasoning/config.yaml" in docs
    assert "model:" not in docs
    assert "trainer:" not in docs
    assert "common:" in docs
    assert "training:" in docs
    assert "serving:" in docs
    assert "grpo:" in docs
    assert "http://localhost:3000" in docs
    assert "http://localhost:9090" in docs
    assert "http://localhost:9091" in docs
    assert "./dev.sh metrics down" in docs
    assert "./dev.sh metrics reset" in docs

    example_yaml = Path("examples/reasoning/config.yaml").read_text(encoding="utf-8")
    assert "model:" not in example_yaml
    assert "trainer:" not in example_yaml
    assert "common:" in example_yaml
    assert "training:" in example_yaml
    assert "serving:" in example_yaml
    assert "grpo:" in example_yaml
    assert "common:\n  model_name: Qwen/Qwen2.5-0.5B-Instruct\n  num_threads:" not in example_yaml
    assert "training:\n  num_threads: 1" in example_yaml
    assert "serving:\n  num_threads: 1" in example_yaml


def test_dev_sh_metrics_commands_and_compose_validation() -> None:
    """The dev helper should expose metrics commands and the compose file should validate."""
    script = Path("dev.sh").read_text(encoding="utf-8")
    assert "metrics <up|down|reset|status>" in script
    assert "metrics_up()" in script
    assert "metrics_down()" in script
    assert "metrics_reset()" in script
    assert "metrics_status()" in script
    assert 'GRAFANA_URL="http://localhost:3000"' in script
    assert 'PROMETHEUS_URL="http://localhost:9090"' in script
    assert 'PUSHGATEWAY_URL="http://localhost:9091"' in script
    assert "/api/health" in script
    assert "/-/ready" in script

    subprocess.run(["bash", "-n", "dev.sh"], check=True)

    compose = Path("metric/docker-compose.yml").read_text(encoding="utf-8")
    assert "grafana-data" in compose
    assert "prometheus-data" in compose
    assert "pushgateway-data" in compose
    assert "--persistence.file=/data/metrics.db" in compose

    if shutil.which("docker") is not None:
        subprocess.run(
            ["docker", "compose", "-f", "metric/docker-compose.yml", "config"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def test_dev_sh_metrics_up_waits_for_endpoint_readiness(tmp_path: Path) -> None:
    """metrics up should report success only after all three endpoints are ready."""
    env = make_controlled_shell_env(
        tmp_path,
        docker_script="""#!/bin/bash
        echo "$*" >> "$FLASHRL_TEST_DOCKER_LOG"
        if [[ "$*" == *" ps" ]]; then
          echo "NAME STATUS"
        fi
        exit 0
        """,
        curl_script="""#!/bin/bash
        url="${!#}"
        echo "$url" >> "$FLASHRL_TEST_CURL_LOG"
        exit 0
        """,
    )

    result = run_dev_sh("metrics", "up", env=env)

    assert result.returncode == 0
    assert "Waiting for metrics endpoints to become ready..." in result.stdout
    assert "✓ FlashRL metrics stack is running" in result.stdout
    assert "Grafana: http://localhost:3000" in result.stdout
    assert "Prometheus: http://localhost:9090" in result.stdout
    assert "Pushgateway: http://localhost:9091" in result.stdout

    curl_log = Path(env["FLASHRL_TEST_CURL_LOG"]).read_text(encoding="utf-8")
    assert "http://localhost:3000/api/health" in curl_log
    assert "http://localhost:9090/-/ready" in curl_log
    assert "http://localhost:9091/-/ready" in curl_log


def test_dev_sh_metrics_up_fails_on_readiness_timeout(tmp_path: Path) -> None:
    """metrics up should exit non-zero and print diagnostics if endpoints never become ready."""
    env = make_controlled_shell_env(
        tmp_path,
        docker_script="""#!/bin/bash
        echo "$*" >> "$FLASHRL_TEST_DOCKER_LOG"
        if [[ "$*" == *" ps" ]]; then
          echo "NAME STATUS"
          echo "flashrl-grafana   Up"
        fi
        exit 0
        """,
        curl_script="""#!/bin/bash
        url="${!#}"
        echo "$url" >> "$FLASHRL_TEST_CURL_LOG"
        exit 1
        """,
        timeout_seconds="0",
        interval_seconds="0",
    )

    result = run_dev_sh("metrics", "up", env=env)

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "✓ FlashRL metrics stack is running" not in combined
    assert "Error: metrics stack did not become ready within 0s." in combined
    assert "Still not ready: grafana prometheus pushgateway" in combined
    assert "Current container status:" in combined
    assert "flashrl-grafana   Up" in combined

    docker_log = Path(env["FLASHRL_TEST_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert "compose -f" in docker_log
    assert " up -d" in docker_log
    assert " ps" in docker_log


@pytest.mark.parametrize(
    ("missing_command", "docker_script", "curl_script"),
    [
        ("docker", None, "#!/bin/bash\nexit 0\n"),
        ("curl", "#!/bin/bash\nexit 0\n", None),
    ],
)
def test_dev_sh_metrics_up_requires_docker_and_curl(
    tmp_path: Path,
    missing_command: str,
    docker_script: str | None,
    curl_script: str | None,
) -> None:
    """metrics up should fail clearly when docker or curl is unavailable."""
    env = make_controlled_shell_env(
        tmp_path,
        docker_script=docker_script,
        curl_script=curl_script,
        timeout_seconds="0",
        interval_seconds="0",
    )

    result = run_dev_sh("metrics", "up", env=env)

    assert result.returncode != 0
    assert f"required command '{missing_command}' was not found" in result.stderr


def test_dev_sh_metrics_status_prints_endpoint_readiness(tmp_path: Path) -> None:
    """metrics status should show both compose state and endpoint readiness."""
    env = make_controlled_shell_env(
        tmp_path,
        docker_script="""#!/bin/bash
        echo "$*" >> "$FLASHRL_TEST_DOCKER_LOG"
        if [[ "$*" == *" ps" ]]; then
          echo "NAME STATUS"
          echo "flashrl-prometheus   Up"
        fi
        exit 0
        """,
        curl_script="""#!/bin/bash
        url="${!#}"
        echo "$url" >> "$FLASHRL_TEST_CURL_LOG"
        if [[ "$url" == *"/api/health" ]]; then
          exit 0
        fi
        exit 1
        """,
    )

    result = run_dev_sh("metrics", "status", env=env)

    assert result.returncode == 0
    assert "NAME STATUS" in result.stdout
    assert "Endpoint readiness:" in result.stdout
    assert "Grafana: ready" in result.stdout
    assert "Prometheus: not_ready" in result.stdout
    assert "Pushgateway: not_ready" in result.stdout
