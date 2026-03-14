"""Tests for Prometheus/Grafana metric stack integration."""

from __future__ import annotations

import json
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
from flashrl.framework.examples.reasoning import train as reasoning_example
from flashrl.framework import (
    FlashRL,
    GrpoConfig,
    LoggingConfig,
    MetricsConfig,
    PushgatewayMetricsConfig,
    TensorBoardMetricsConfig,
)
from flashrl.framework.config import TrainingConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.metrics import (
    CompositeMetricsSink,
    PrometheusMetricsSink,
    TensorBoardMetricsSink,
)
from flashrl.framework.training import TrainingBackend


class FakeTokenizer:
    """A tiny tokenizer for deterministic unit tests."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = vocab_size - 1
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
        self.generation_defaults: dict[str, object] = {}
        self._batch_call_index = 0
        self._live_rollout_debug_callback = None
        self._live_rollout_debug_context: dict[str, object] = {}
        self._live_rollout_candidate_index: int | None = None

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        call_index = self._batch_call_index
        self._batch_call_index += 1
        outputs = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_token_ids = self.tokenizer._encode(prompt, max_length=self.config.max_length)
            response = f"generated::{prompt}::{call_index}"
            response_token_ids = self.tokenizer._encode(
                response,
                max_length=self.config.max_length,
            )[:4]
            response_token_logprobs = [
                -0.1 - 0.01 * call_index - 0.001 * token_index
                for token_index in range(len(response_token_ids))
            ]
            metadata = {}
            if self._live_rollout_debug_callback is not None:
                debug_payload = {
                    **self._live_rollout_debug_context,
                    "prompt_index": prompt_index,
                    "candidate_index": self._live_rollout_candidate_index or 0,
                    "prompt_text": prompt,
                    "prompt_preview": prompt,
                }
                self._live_rollout_debug_callback("start", debug_payload)
                self._live_rollout_debug_callback("chunk", {**debug_payload, "text": response})
                metadata = {
                    "ttft_seconds": 0.1,
                    "tpot_seconds": 0.02,
                    "generation_seconds": 0.16,
                    "response_token_count": len(response_token_ids),
                }
                self._live_rollout_debug_callback(
                    "done",
                    {
                        **debug_payload,
                        **metadata,
                        "response_preview": response,
                    },
                )
            outputs.append(
                SimpleNamespace(
                    text=response,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    log_prob=float(sum(response_token_logprobs)),
                    metadata=metadata,
                )
            )
        return outputs

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        del kwargs
        grouped_outputs = []
        for prompt in prompts:
            prompt_token_ids = self.tokenizer._encode(prompt, max_length=self.config.max_length)
            prompt_outputs = []
            for candidate_index in range(group_size):
                response = f"generated::{prompt}::{candidate_index}"
                response_token_ids = self.tokenizer._encode(
                    response,
                    max_length=self.config.max_length,
                )[:4]
                response_token_logprobs = [
                    -0.1 - 0.01 * candidate_index - 0.001 * token_index
                    for token_index in range(len(response_token_ids))
                ]
                prompt_outputs.append(
                    SimpleNamespace(
                        text=response,
                        prompt_token_ids=prompt_token_ids,
                        response_token_ids=response_token_ids,
                        response_token_logprobs=response_token_logprobs,
                        log_prob=float(sum(response_token_logprobs)),
                        metadata={},
                    )
                )
            grouped_outputs.append(prompt_outputs)
        return grouped_outputs

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def set_live_rollout_debug(self, callback, context) -> None:
        self._live_rollout_debug_callback = callback
        self._live_rollout_debug_context = dict(context)

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        self._live_rollout_candidate_index = candidate_index

    def clear_live_rollout_debug(self) -> None:
        self._live_rollout_debug_callback = None
        self._live_rollout_debug_context = {}
        self._live_rollout_candidate_index = None


class FakeTrainingBackend(TrainingBackend):
    """Training backend used for metrics tests."""

    last_config = None
    last_reference_config = None

    def __init__(
        self,
        config,
        learning_rate: float = 1e-5,
        grpo_config: GrpoConfig | None = None,
        reference_enabled: bool = False,
        reference_device: str | None = None,
    ) -> None:
        resolved_config = config or TrainingConfig(model_name="fake/model", device="cpu")
        type(self).last_config = resolved_config
        type(self).last_reference_config = None
        super().__init__(
            resolved_config,
            learning_rate=learning_rate,
            grpo_config=grpo_config or GrpoConfig(group_size=2),
            reference_enabled=reference_enabled,
            reference_device=reference_device,
        )
        self.actor = FakeActor(bias_shift=0.25)
        self.actor.train()
        self.device = self.actor.device
        self.optimizer = torch.optim.SGD(self.actor.model.parameters(), lr=learning_rate)
        self.startup_events = [
            {
                "component": "training_backend",
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": resolved_config.num_threads,
                    "duration_seconds": 0.0,
                },
            }
        ]
        if reference_enabled:
            reference_config = resolved_config.model_copy(
                update={"device": reference_device or resolved_config.device}
            )
            type(self).last_reference_config = reference_config
            self.reference = FakeReferenceModel(reference_config)
            self.startup_events.append(
                {
                    "component": "reference_model",
                    "status": "completed",
                    "metadata": {
                        "device": str(self.reference.device),
                        "cpu_threads": reference_config.num_threads,
                        "duration_seconds": 0.0,
                    },
                }
            )


def _encode_text(text: str, *, max_length: int = 128, vocab_size: int = 32) -> list[int]:
    tokens = [((ord(char) % (vocab_size - 1)) + 1) for char in text[:max_length]]
    return tokens or [1]


class FakeServingBackend:
    """Serving backend used for metrics tests."""

    last_config = None

    def __init__(self, config) -> None:
        type(self).last_config = config
        self.config = config
        self._actor = FakeActor(bias_shift=0.1)
        self._actor.eval()
        self.device = self._actor.device
        self.generation_defaults: dict[str, object] = {}

    def generate(self, prompts: list[str], **kwargs):
        return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)
        self._actor.set_generation_defaults(**kwargs)

    def sync_from_training_actor(self, training_actor) -> None:
        self._actor.model.load_state_dict(training_actor.model.state_dict())

    def set_live_rollout_debug(self, callback, context) -> None:
        if not getattr(self.config, "debug_live_rollout", False):
            return
        self._actor.set_live_rollout_debug(callback, context)

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        if not getattr(self.config, "debug_live_rollout", False):
            return
        self._actor.set_live_rollout_candidate_index(candidate_index)

    def clear_live_rollout_debug(self) -> None:
        self._actor.clear_live_rollout_debug()

    def close(self) -> None:
        return None


class FakeReferenceModel:
    """Reference model used for KL computation in tests."""

    last_config = None

    def __init__(self, config) -> None:
        type(self).last_config = config
        self.device = torch.device("cpu")
        self.model = FakeCausalLM(bias_shift=0.0)


class FakeSummaryWriter:
    """Minimal file-backed scalar writer used to test TensorBoard sink wiring."""

    instances: list["FakeSummaryWriter"] = []
    counter = 0

    def __init__(self, log_dir: str) -> None:
        type(self).counter += 1
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"events.out.tfevents.fake.{type(self).counter}"
        self.path.touch()
        self.flush_count = 0
        self.closed = False
        type(self).instances.append(self)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        record = {
            "tag": tag,
            "value": float(scalar_value),
            "step": int(global_step),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def install_fake_tensorboard_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the runtime TensorBoard writer with a deterministic file writer."""
    FakeSummaryWriter.instances.clear()
    FakeSummaryWriter.counter = 0
    monkeypatch.setattr(metrics_module, "_create_summary_writer", lambda log_dir: FakeSummaryWriter(log_dir))


def make_rollout_fn(response_suffix: str = "response", repeat: int = 1):
    """Create a rollout function with deterministic responses."""

    call_index = 0

    def rollout_fn(prompts: list[Prompt], serving_backend) -> list[RolloutOutput]:
        nonlocal call_index
        del serving_backend
        outputs = []
        for prompt in prompts:
            prompt_token_ids = _encode_text(prompt.text)
            response = f"{response_suffix} " + ("detail " * repeat) + prompt.text + f"::{call_index}"
            response_token_ids = _encode_text(response)[:4]
            response_token_logprobs = [
                -0.05 - 0.01 * call_index - 0.001 * token_index
                for token_index in range(len(response_token_ids))
            ]
            outputs.append(
                RolloutOutput(
                    text=response,
                    log_prob=float(sum(response_token_logprobs)),
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    conversation=Conversation(
                        messages=[
                            Message(role="user", content=prompt.text),
                            Message(role="assistant", content=response),
                        ]
                    ),
                )
            )
        call_index += 1
        return outputs

    return rollout_fn


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Reward longer responses slightly more."""
    return RewardOutput(reward=len(rollout.text) / 50.0)


def patch_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch FlashRL to use fake local backends."""
    FakeTrainingBackend.last_config = None
    FakeServingBackend.last_config = None
    FakeTrainingBackend.last_reference_config = None
    FakeReferenceModel.last_config = None
    monkeypatch.setattr(
        flashrl_module,
        "create_training_backend",
        lambda config, learning_rate, grpo_config, reference_enabled=False, reference_device=None: (
            FakeTrainingBackend(
                config,
                learning_rate=learning_rate,
                grpo_config=grpo_config,
                reference_enabled=reference_enabled,
                reference_device=reference_device,
            )
        ),
    )
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None: FakeServingBackend(config),
    )


def sample_value(registry, name: str) -> tuple[float, dict[str, str]]:
    """Read one metric sample value and labels from a registry."""
    for metric in registry.collect():
        for sample in metric.samples:
            if sample.name == name:
                return float(sample.value), dict(sample.labels)
    raise AssertionError(f"Metric sample not found: {name}")


def read_events(run_dir: Path) -> list[dict]:
    """Load structured run events from events.jsonl."""
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_tensorboard_scalars(run_dir: Path) -> list[dict[str, object]]:
    """Load fake TensorBoard scalar records from one run directory."""
    records: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("events.out.tfevents*")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


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


            def rollout_fn(prompts, serving_backend):
                samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
                outputs = []
                for prompt, sample in zip(prompts, samples, strict=True):
                    outputs.append(
                        RolloutOutput(
                            text=sample.text,
                            log_prob=sample.log_prob,
                            prompt_token_ids=sample.prompt_token_ids,
                            response_token_ids=sample.response_token_ids,
                            response_token_logprobs=sample.response_token_logprobs,
                            metadata=dict(getattr(sample, "metadata", {})),
                            conversation=Conversation(
                                messages=[
                                    Message(role="user", content=prompt.text),
                                    Message(role="assistant", content=sample.text),
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
    tensorboard_enabled: bool | None = None,
    pushgateway_enabled: bool = False,
    common_dtype: str | None = None,
    serving_num_threads: int = 3,
    serving_debug_live_rollout: bool = False,
    training_num_threads: int | None = 1,
    group_size: int = 2,
    clip_ratio: float = 0.2,
    kl_coefficient: float = 0.0,
) -> Path:
    """Write a temporary YAML config for FlashRL.from_yaml tests."""
    config_path = tmp_path / "run.yaml"
    if tensorboard_enabled is None:
        tensorboard_enabled = metrics_enabled
    training_section = {
        "learning_rate": 1.0e-5,
        "batch_size": 2,
        "max_epochs": 1,
    }
    if training_num_threads is not None:
        training_section["num_threads"] = training_num_threads

    serving_section: dict[str, object] = {}
    if serving_num_threads is not None:
        serving_section["num_threads"] = serving_num_threads
    serving_section["debug_live_rollout"] = serving_debug_live_rollout

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
                    "tensorboard": {
                        "enabled": tensorboard_enabled,
                    },
                    "pushgateway": {
                        "enabled": pushgateway_enabled,
                        "url": "http://localhost:9091",
                        "job_name": "flashrl-test",
                    },
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
    """The Prometheus sink should own the expected gauges and push the registry."""
    pushes: list[tuple[str, str]] = []

    sink = PrometheusMetricsSink(
        PushgatewayMetricsConfig(
            enabled=True,
            url="http://localhost:9091",
            job_name="flashrl-test",
        ),
        model_name="fake/model",
        push_fn=lambda url, job, registry: pushes.append((url, job)),
    )

    sink.observe_stage({"stage": "rollout", "latency_seconds": 1.25})
    sink.observe_stage({"stage": "reward", "latency_seconds": 0.5})
    sink.observe_step(
        {
            "loss": 0.75,
            "reward_mean": 1.5,
            "kl_divergence": 0.25,
            "step_duration_seconds": 2.0,
        }
    )
    sink.observe_serving_debug({"ttft_seconds": 0.12, "tpot_seconds": 0.03})
    sink.push()

    assert {metric.name for metric in sink.registry.collect()} == {
        "flashrl_train_loss",
        "flashrl_reward_mean",
        "flashrl_kl_mean",
        "flashrl_rollout_latency_seconds",
        "flashrl_reward_latency_seconds",
        "flashrl_step_duration_seconds",
        "flashrl_serving_ttft_seconds",
        "flashrl_serving_tpot_seconds",
    }
    assert sample_value(sink.registry, "flashrl_train_loss")[0] == pytest.approx(0.75)
    assert sample_value(sink.registry, "flashrl_reward_mean")[0] == pytest.approx(1.5)
    assert sample_value(sink.registry, "flashrl_kl_mean")[0] == pytest.approx(0.25)
    assert sample_value(sink.registry, "flashrl_rollout_latency_seconds")[0] == pytest.approx(1.25)
    assert sample_value(sink.registry, "flashrl_reward_latency_seconds")[0] == pytest.approx(0.5)
    step_duration, labels = sample_value(sink.registry, "flashrl_step_duration_seconds")
    assert step_duration == pytest.approx(2.0)
    assert sample_value(sink.registry, "flashrl_serving_ttft_seconds")[0] == pytest.approx(0.12)
    assert sample_value(sink.registry, "flashrl_serving_tpot_seconds")[0] == pytest.approx(0.03)
    assert labels == {
        "model": "fake/model",
        "algorithm": "grpo",
        "runtime": "framework_local",
    }
    assert pushes == [("http://localhost:9091", "flashrl-test")]


def test_metrics_config_defaults_enabled() -> None:
    """Metrics should default to TensorBoard-on and Pushgateway-off."""
    config = MetricsConfig()
    assert config.enabled is True
    assert config.tensorboard.enabled is True
    assert config.pushgateway.enabled is False


def test_composite_metrics_sink_forwards_to_all_children() -> None:
    """Composite metrics sink should broadcast all lifecycle and metric events."""

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def start_run(self, *, run_dir: Path, run_id: str) -> None:
            self.events.append(("start_run", (run_dir, run_id)))

        def observe_stage(self, payload: dict[str, object]) -> None:
            self.events.append(("observe_stage", payload))

        def observe_step(self, payload: dict[str, object]) -> None:
            self.events.append(("observe_step", payload))

        def observe_serving_debug(self, payload: dict[str, object]) -> None:
            self.events.append(("observe_serving_debug", payload))

        def push(self) -> None:
            self.events.append(("push", None))

        def finish_run(self) -> None:
            self.events.append(("finish_run", None))

        def close(self) -> None:
            self.events.append(("close", None))

    left = RecordingSink()
    right = RecordingSink()
    sink = CompositeMetricsSink([left, right])

    sink.start_run(run_dir=Path("/tmp/run"), run_id="run-1")
    sink.observe_stage({"stage": "reward", "step": 1, "latency_seconds": 0.1})
    sink.observe_step({"step": 1, "loss": 0.2})
    sink.observe_serving_debug({"step": 1, "ttft_seconds": 0.1, "tpot_seconds": 0.02})
    sink.push()
    sink.finish_run()
    sink.close()

    assert [name for name, _ in left.events] == [
        "start_run",
        "observe_stage",
        "observe_step",
        "observe_serving_debug",
        "push",
        "finish_run",
        "close",
    ]
    assert left.events == right.events


def test_tensorboard_metrics_sink_writes_expected_scalars_and_flushes(tmp_path: Path) -> None:
    """TensorBoard sink should write scalar records into the run root and flush on push."""
    sink = TensorBoardMetricsSink(
        TensorBoardMetricsConfig(enabled=True),
        writer_factory=lambda log_dir: FakeSummaryWriter(log_dir),
    )

    sink.start_run(run_dir=tmp_path, run_id="run-1")
    sink.observe_stage(
        {
            "step": 3,
            "stage": "rollout",
            "latency_seconds": 0.4,
            "prompt_tokens_mean": 12.0,
            "prompt_tokens_max": 18,
            "response_tokens_mean": 4.0,
            "response_tokens_max": 6,
        }
    )
    sink.observe_stage(
        {
            "step": 3,
            "stage": "reward",
            "latency_seconds": 0.2,
            "reward_mean": 1.5,
            "reward_std": 0.5,
            "reward_min": 1.0,
            "reward_max": 2.0,
            "accuracy_pass_rate": 0.5,
            "format_pass_rate": 1.0,
            "truncation_rate": 0.0,
            "reward_per_item_mean_seconds": 0.01,
        }
    )
    sink.observe_stage(
        {
            "step": 3,
            "stage": "advantage",
            "latency_seconds": 0.1,
            "advantage_mean": 0.0,
            "advantage_std": 1.0,
            "advantage_min": -1.0,
            "advantage_max": 1.0,
        }
    )
    sink.observe_stage(
        {
            "step": 3,
            "stage": "prepare_inputs",
            "latency_seconds": 0.05,
            "full_tokens_mean": 20.0,
            "full_tokens_max": 24,
            "response_tokens_total": 8,
        }
    )
    sink.observe_stage(
        {
            "step": 3,
            "stage": "loss_assembly",
            "latency_seconds": 0.08,
            "importance_sampling_ratio_mean": 1.1,
            "importance_sampling_ratio_std": 0.2,
            "importance_sampling_ratio_min": 0.8,
            "importance_sampling_ratio_max": 1.4,
            "clip_fraction": 0.25,
        }
    )
    sink.observe_stage(
        {
            "step": 3,
            "stage": "optimizer",
            "latency_seconds": 0.03,
            "learning_rate": 1.0e-5,
        }
    )
    sink.observe_step(
        {
            "step": 3,
            "loss": 0.75,
            "policy_loss": 0.5,
            "kl_divergence": 0.25,
            "tokens_per_second": 42.0,
            "step_duration_seconds": 2.0,
        }
    )
    sink.observe_serving_debug({"step": 3, "ttft_seconds": 0.12, "tpot_seconds": 0.03})
    sink.push()
    writer = FakeSummaryWriter.instances[-1]
    sink.finish_run()

    assert writer.flush_count == 1
    assert writer.closed is True

    records = read_tensorboard_scalars(tmp_path)
    assert any(record["tag"] == "train/loss" and record["step"] == 3 for record in records)
    assert any(record["tag"] == "train/learning_rate" and record["step"] == 3 for record in records)
    assert any(record["tag"] == "reward/mean" and record["value"] == pytest.approx(1.5) for record in records)
    assert any(
        record["tag"] == "importance_sampling_ratio/clip_fraction"
        and record["value"] == pytest.approx(0.25)
        for record in records
    )
    assert any(
        record["tag"] == "timing/stage/reward_seconds"
        and record["value"] == pytest.approx(0.2)
        for record in records
    )
    assert any(record["tag"] == "serving/ttft_seconds" for record in records)


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
        PushgatewayMetricsConfig(
            enabled=True,
            url="http://localhost:9091",
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


def test_flashrl_pushgateway_metrics_process_lifetime_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pushgateway metrics should update and push across consecutive runs with one sink instance."""
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
            tensorboard=TensorBoardMetricsConfig(enabled=False),
            pushgateway=PushgatewayMetricsConfig(
                enabled=True,
                url="http://localhost:9091",
                job_name="flashrl-test",
            ),
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
    """Enabled Pushgateway metrics should warn and continue when Pushgateway is unavailable."""
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
        metrics_config=MetricsConfig(
            enabled=True,
            tensorboard=TensorBoardMetricsConfig(enabled=False),
            pushgateway=PushgatewayMetricsConfig(
                enabled=True,
                url="http://localhost:9091",
                job_name="flashrl-test",
            ),
        ),
    )

    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]
    with pytest.warns(RuntimeWarning, match="best-effort mode"):
        trainer.train(dataset)

    assert trainer._metrics_sink is not None
    assert (trainer._run_logger.run_dir / "events.jsonl").exists()


def test_flashrl_default_metrics_create_tensorboard_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default metrics should create TensorBoard event files in each run directory."""
    install_fake_tensorboard_writer(monkeypatch)
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="tensorboard", repeat=2),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )

    trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    assert trainer._metrics_sink is not None
    event_files = sorted(trainer._run_logger.run_dir.glob("events.out.tfevents*"))
    assert event_files
    scalars = read_tensorboard_scalars(trainer._run_logger.run_dir)
    assert any(record["tag"] == "train/loss" for record in scalars)
    assert any(record["tag"] == "reward/mean" for record in scalars)
    assert any(record["tag"] == "importance_sampling_ratio/mean" for record in scalars)


def test_flashrl_disabled_metrics_skip_tensorboard_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Disabled metrics should not create TensorBoard event files."""
    install_fake_tensorboard_writer(monkeypatch)
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="disabled", repeat=2),
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

    trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    assert not list(trainer._run_logger.run_dir.glob("events.out.tfevents*"))


def test_flashrl_tensorboard_writer_reopens_cleanly_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reusing one FlashRL instance should reopen a fresh TensorBoard writer per run."""
    install_fake_tensorboard_writer(monkeypatch)
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="tb-reuse", repeat=2),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )

    dataset = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    trainer.train(dataset)
    first_run_dir = trainer._run_logger.run_dir
    first_writer = FakeSummaryWriter.instances[0]
    trainer.train(dataset)
    second_run_dir = trainer._run_logger.run_dir
    second_writer = FakeSummaryWriter.instances[1]

    assert first_run_dir != second_run_dir
    assert first_writer.closed is True
    assert second_writer.closed is True
    assert list(first_run_dir.glob("events.out.tfevents*"))
    assert list(second_run_dir.glob("events.out.tfevents*"))


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


def test_flashrl_yaml_serving_debug_live_rollout_wires_events_artifacts_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Serving debug YAML should reach runtime and emit debug events, artifacts, and metrics."""
    patch_backends(monkeypatch)
    monkeypatch.setattr(metrics_module, "push_to_gateway", lambda *args, **kwargs: None)
    hook_module = create_yaml_hook_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_yaml_run_config(
        tmp_path,
        hook_module=hook_module,
        log_dir=tmp_path / "debug-logs",
        metrics_enabled=True,
        tensorboard_enabled=False,
        pushgateway_enabled=True,
        serving_debug_live_rollout=True,
    )

    trainer = FlashRL.from_yaml(config_path)
    assert trainer.serving_config.debug_live_rollout is True

    trainer.train()

    assert trainer._run_logger is not None
    events = read_events(trainer._run_logger.run_dir)
    rollout_records = [
        json.loads(line)
        for line in trainer._run_logger.rollouts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transcript = trainer._run_logger.console_path.read_text(encoding="utf-8")

    assert any(event["event"] == "serving_debug" for event in events)
    assert (
        "step 1/4  epoch 1/1  batch 1/4  prompt_window=1-1/4  "
        "prompts_this_step=1/1  completions_per_prompt=2  completions_this_step=2/2"
        in transcript
    )
    assert "  prompt 1/1  yaml prompt 0" in transcript
    assert "    rollout 1/2 running..." in transcript
    assert "    rollout 1/2 done  ttft=100.0ms  tpot=20.0ms" in transcript
    assert "  candidate " not in transcript
    assert "serve step=" not in transcript
    assert any(
        "ttft_seconds" in candidate["rollout"]["metadata"]
        and "tpot_seconds" in candidate["rollout"]["metadata"]
        for record in rollout_records
        for candidate in record["candidates"]
    )
    assert trainer._metrics_sink is not None
    assert sample_value(trainer._metrics_sink.registry, "flashrl_serving_ttft_seconds")[0] == pytest.approx(0.1)
    assert sample_value(trainer._metrics_sink.registry, "flashrl_serving_tpot_seconds")[0] == pytest.approx(0.02)


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
    monkeypatch.setattr(
        reasoning_example,
        "_load_gsm8k_split",
        lambda split, limit=None, **kwargs: [
            {
                "task_id": f"gsm8k-{split}-000001",
                "source": "gsm8k",
                "split": split,
                "problem": "What is 15 + 27?",
                "final_answer": "42",
            },
            {
                "task_id": f"gsm8k-{split}-000002",
                "source": "gsm8k",
                "split": split,
                "problem": "What is 12 + 15 + 8?",
                "final_answer": "35",
            },
            {
                "task_id": f"gsm8k-{split}-000003",
                "source": "gsm8k",
                "split": split,
                "problem": "What is 9 * 6?",
                "final_answer": "54",
            },
            {
                "task_id": f"gsm8k-{split}-000004",
                "source": "gsm8k",
                "split": split,
                "problem": "If I divide 24 by 3, what do I get?",
                "final_answer": "8",
            },
        ],
    )

    def scripted_generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        outputs = []
        for prompt in prompts:
            if "15 + 27" in prompt:
                expected = "42"
                wrong = "41"
            elif "12 + 15 + 8" in prompt:
                expected = "35"
                wrong = "34"
            elif "9 * 6" in prompt:
                expected = "54"
                wrong = "53"
            else:
                expected = "8"
                wrong = "7"
            prompt_token_ids = self.tokenizer._encode(prompt, max_length=self.config.max_length)
            response_templates = [
                f"<think>Work through the arithmetic carefully and conclude the answer is {expected}.</think><answer>{expected}</answer>",
                f"<think>Work through the arithmetic carefully but conclude the answer is {wrong}.</think><answer>{wrong}</answer>",
                f"<think>Work through the arithmetic carefully and conclude the answer is {expected}.</think><answer>{expected}</answer>\nextra",
                f"<think>Work through the arithmetic carefully and conclude the answer is {expected}.",
            ]
            call_index = self._batch_call_index % len(response_templates)
            response = response_templates[call_index]
            self._batch_call_index += 1
            response_token_ids = self.tokenizer._encode(
                response,
                max_length=self.config.max_length,
            )[:16]
            response_token_logprobs = [
                -0.1 - 0.01 * call_index - 0.001 * token_index
                for token_index in range(len(response_token_ids))
            ]
            outputs.append(
                SimpleNamespace(
                    text=response,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    log_prob=float(sum(response_token_logprobs)),
                    metadata={
                        "finish_reason": "length" if response == response_templates[-1] else "stop",
                    },
                )
            )
        return outputs

    monkeypatch.setattr(FakeActor, "generate_batch", scripted_generate_batch)

    trainer = FlashRL.from_yaml("flashrl/framework/examples/reasoning/config.yaml")
    trainer.logging_config = LoggingConfig(
        log_dir=tmp_path,
        console=False,
        file=True,
    )
    trainer.metrics_config = MetricsConfig(
        enabled=True,
        tensorboard=TensorBoardMetricsConfig(enabled=False),
        pushgateway=PushgatewayMetricsConfig(
            enabled=True,
            url="http://localhost:9091",
            job_name="flashrl-test",
        ),
    )
    trainer._metrics_sink = PrometheusMetricsSink(
        trainer.metrics_config.pushgateway,
        model_name=trainer.training_model_config.model_name,
    )
    assert trainer._trainer is not None
    trainer._trainer.metrics_sink = trainer._metrics_sink

    trainer.train()

    assert trainer._run_logger.run_dir.exists()
    rollout_records = [
        json.loads(line)
        for line in trainer._run_logger.rollouts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_done_events = [
        json.loads(line)
        for line in trainer._run_logger.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    reward_values = [
        candidate["reward"]["value"]
        for record in rollout_records
        for candidate in record["candidates"]
    ]
    losses = [
        event["payload"]["loss"]
        for event in step_done_events
        if event["event"] == "step_done"
    ]

    assert any(value > 0.0 for value in reward_values)
    assert any(abs(loss) > 1e-6 for loss in losses)
    assert any(
        "accuracy_pass" in candidate["reward"]["metadata"]
        and "format_pass" in candidate["reward"]["metadata"]
        for record in rollout_records
        for candidate in record["candidates"]
    )


def test_observability_stack_files_and_docs_exist() -> None:
    """The local compose stack and quickstart docs should be committed."""
    assert Path("metric/docker-compose.yml").exists()
    assert Path("metric/prometheus/prometheus.yml").exists()
    assert Path("metric/grafana/provisioning/datasources/prometheus.yml").exists()
    assert Path("metric/grafana/provisioning/dashboards/dashboards.yml").exists()
    assert Path("metric/grafana/dashboards/flashrl-v1.json").exists()
    assert "observability/docker-compose.yml" not in Path(
        "flashrl/framework/examples/README.md"
    ).read_text(
        encoding="utf-8"
    )
    assert not Path("examples").exists()
    assert Path("flashrl/framework/examples/__init__.py").exists()
    assert Path("flashrl/framework/examples/reasoning/__init__.py").exists()
    assert Path("flashrl/framework/examples/reasoning/train.py").exists()
    assert Path("flashrl/framework/examples/reasoning/config.yaml").exists()
    assert Path("flashrl/framework/examples/reasoning/config_vllm.yaml").exists()

    docs = Path("flashrl/framework/examples/README.md").read_text(encoding="utf-8")
    reasoning_docs = Path("flashrl/framework/examples/reasoning/README.md").read_text(
        encoding="utf-8"
    )
    root_docs = Path("README.md").read_text(encoding="utf-8")
    assert "tensorboard --logdir logs" in root_docs
    assert "TensorBoard is the default local metrics path." in docs
    assert "metrics.pushgateway.enabled: true" in docs
    assert "./dev.sh metrics up" in docs
    assert "endpoint-ready before reporting success" in docs
    assert "reasoning/README.md" in docs
    assert "http://localhost:3000" in docs
    assert "tensorboard --logdir logs" in docs
    assert "python3 -m flashrl.framework.examples.reasoning.train" in reasoning_docs
    assert (
        "python3 -m flashrl.framework.flashrl --config "
        "flashrl/framework/examples/reasoning/config.yaml"
    ) in reasoning_docs
    assert "config_vllm.yaml" in reasoning_docs
    assert "--dataset" in reasoning_docs
    assert "aime25" in reasoning_docs
    assert "--train-limit" in reasoning_docs
    assert "--eval-limit" in reasoning_docs
    assert "--checkpoint-out" in reasoning_docs
    assert "available` is the full size" in reasoning_docs
    assert "selected` is the number of rows actually used" in reasoning_docs
    assert "not copied into `console.log`" in reasoning_docs
    assert "math.yaml" not in reasoning_docs
    assert "model:" not in reasoning_docs
    assert "trainer:" not in reasoning_docs
    assert "common.model_name" in reasoning_docs
    assert "training.batch_size" in reasoning_docs
    assert "serving.backend" in reasoning_docs
    assert "grpo.group_size" in reasoning_docs
    assert "http://localhost:9090" in docs
    assert "http://localhost:9091" in docs
    assert "./dev.sh metrics down" in docs
    assert "./dev.sh metrics reset" in docs
    assert "serving.backend: vllm" in docs
    assert "FLASHRL_VLLM_PYTHON" in docs
    assert "optional `vllm` extra" in docs

    example_yaml = Path("flashrl/framework/examples/reasoning/config.yaml").read_text(
        encoding="utf-8"
    )
    vllm_example_yaml = Path(
        "flashrl/framework/examples/reasoning/config_vllm.yaml"
    ).read_text(
        encoding="utf-8"
    )
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "model:" not in example_yaml
    assert "trainer:" not in example_yaml
    assert "common:" in example_yaml
    assert "training:" in example_yaml
    assert "serving:" in example_yaml
    assert "grpo:" in example_yaml
    assert "metrics:" in example_yaml
    assert "runtime:" in example_yaml
    assert "hooks:" in example_yaml
    assert "tensorboard:" in example_yaml
    assert "pushgateway:" in example_yaml
    assert "pushgateway_url:" not in example_yaml
    assert "common:\n  model_name: Qwen/Qwen2.5-0.5B-Instruct\n  num_threads:" not in example_yaml
    assert "training:\n  num_threads: 1" in example_yaml
    assert "serving:\n  num_threads: 1\n  debug_live_rollout:" in example_yaml
    assert "debug_live_rollout:" in docs
    assert "backend: vllm" in vllm_example_yaml
    assert "runtime_python: ${FLASHRL_VLLM_PYTHON}" in vllm_example_yaml
    assert "~/.venv-vllm" not in vllm_example_yaml
    assert "~/.venv-vllm-metal" not in vllm_example_yaml
    assert "vllm_args:" not in vllm_example_yaml
    assert 'requires-python = ">=3.10"' in pyproject
    assert '"tensorboard>=2.14.0"' in pyproject
    assert '[project.optional-dependencies]' in pyproject
    assert 'vllm = [' in pyproject
    assert '"vllm>=0.16.0"' in pyproject
    dependencies_block = pyproject.split("[project.optional-dependencies]", maxsplit=1)[0]
    assert '"vllm>=0.16.0"' not in dependencies_block


def test_dev_sh_metrics_commands_and_compose_validation() -> None:
    """The dev helper should expose metrics commands and the compose file should validate."""
    script = Path("dev.sh").read_text(encoding="utf-8")
    assert "metrics <up|down|reset|status>" in script
    assert "vllm <setup|status>" in script
    assert "metrics_up()" in script
    assert "metrics_down()" in script
    assert "metrics_reset()" in script
    assert "metrics_status()" in script
    assert "vllm_setup()" in script
    assert "vllm_status()" in script
    assert "FLASHRL_VLLM_PYTHON" in script
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


def test_dev_sh_vllm_status_reports_prepared_runtime(tmp_path: Path) -> None:
    """`./dev.sh vllm status` should report a prepared runtime deterministically."""
    fake_home = tmp_path / "home"
    runtime_dir = fake_home / ".venv-vllm-metal" / "bin"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    python_path = runtime_dir / "python"
    python_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_path.chmod(0o755)

    result = subprocess.run(
        ["bash", "-lc", "HOME=\"$1\" ./dev.sh vllm status", "_", str(fake_home)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path.cwd(),
    )

    assert f"Prepared vLLM runtime: {python_path}" in result.stdout
    assert f'export FLASHRL_VLLM_PYTHON="{python_path}"' in result.stdout


def test_source_dev_sh_auto_exports_flashrl_vllm_python(tmp_path: Path) -> None:
    """`source ./dev.sh` should auto-export the default prepared vLLM runtime."""
    fake_home = tmp_path / "home"
    runtime_dir = fake_home / ".venv-vllm-metal" / "bin"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    python_path = runtime_dir / "python"
    python_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_path.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            "HOME=\"$1\"; source ./dev.sh >/dev/null; printf '%s' \"$FLASHRL_VLLM_PYTHON\"",
            "_",
            str(fake_home),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path.cwd(),
    )

    assert result.stdout == str(python_path)


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
