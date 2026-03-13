"""Tests for the managed HTTP vLLM serving backend."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import flashrl.framework.flashrl as flashrl_module
import flashrl.framework.serving as serving_module
import flashrl.framework.serving.huggingface as huggingface_module
import flashrl.framework.serving.vllm as vllm_module
from flashrl.framework.config import LoggingConfig, MetricsConfig, ServingConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.flashrl import FlashRL
from flashrl.framework.serving import (
    HuggingFaceServingBackend,
    ServingBackend,
    VLLMServingBackend,
    create_serving_backend,
)
from tests.conftest import TinyActor

pytestmark = pytest.mark.unit


class LocalActor:
    """Small actor stub used to isolate serving backend factory tests."""

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(load_state_dict=lambda state_dict: None)

    def eval(self) -> None:
        return None

    def generate(self, prompts: list[str], **kwargs):
        del prompts, kwargs
        return []

    def generate_batch(self, prompts: list[str], **kwargs):
        del prompts, kwargs
        return []

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        del prompts, group_size, kwargs
        return []

    def set_generation_defaults(self, **kwargs) -> None:
        del kwargs


class SnapshotModel:
    """Model stub that records save_pretrained calls."""

    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save_pretrained(self, path: str | Path, safe_serialization: bool = False) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_text("weights", encoding="utf-8")
        (target / "config.json").write_text("{}", encoding="utf-8")
        self.saved_paths.append(target)
        assert safe_serialization is True


class SnapshotTokenizer:
    """Tokenizer stub that records save_pretrained calls."""

    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.saved_paths.append(target)


class SnapshotTrainingActor:
    """Training actor stub used to verify snapshot sync behavior."""

    def __init__(self) -> None:
        self.model = SnapshotModel()
        self.tokenizer = SnapshotTokenizer()


class FakeProcess:
    """Minimal `Popen` stand-in for backend tests."""

    def __init__(self, pid: int | None = None) -> None:
        self.stderr = StringIO("")
        self._returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.pid = pid

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = 0


class StubTrainingBackend:
    """Training backend stub for FlashRL lifecycle tests."""

    def __init__(self, config, learning_rate: float = 1e-5) -> None:
        del config
        self.actor = TinyActor(bias_shift=0.25)
        self.actor.train()
        self.optimizer = torch.optim.SGD(self.actor.model.parameters(), lr=learning_rate)

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.actor.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.actor.model.load_state_dict(torch.load(path, weights_only=False))

    def sync_weights_to(self, serving_backend) -> None:
        serving_backend.sync_from_training_actor(self.actor)


class ClosableServingBackend:
    """Serving backend stub that records close calls."""

    def __init__(self) -> None:
        self._actor = TinyActor(bias_shift=0.1)
        self.device = self._actor.device
        self.generation_defaults: dict[str, object] = {}
        self.close_calls = 0

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
        del training_actor

    def set_live_rollout_debug(self, callback, context) -> None:
        del callback, context

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        del candidate_index

    def clear_live_rollout_debug(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1


def build_rollout_fn(prompts: list[Prompt], serving_backend: ServingBackend) -> list[RolloutOutput]:
    """Construct one rollout per prompt using the serving batch API."""
    samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
    return [
        RolloutOutput(
            text=sample.text,
            log_prob=sample.log_prob,
            prompt_token_ids=sample.prompt_token_ids,
            response_token_ids=sample.response_token_ids,
            response_token_logprobs=sample.response_token_logprobs,
            metadata=dict(sample.metadata),
            conversation=Conversation(
                messages=[
                    Message(role="user", content=prompt.text),
                    Message(role="assistant", content=sample.text),
                ]
            ),
        )
        for prompt, sample in zip(prompts, samples, strict=True)
    ]


def failing_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Raise during reward to exercise train-failure cleanup."""
    raise ValueError(f"reward failure for {rollout.text}")


def dataset_fn() -> list[Prompt]:
    """Small dataset for YAML config loading tests."""
    return [Prompt(text="prompt 0"), Prompt(text="prompt 1")]


def _make_fake_runtime(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "runtime" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    python_path = bin_dir / "python"
    python_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_path.chmod(0o755)

    vllm_path = bin_dir / "vllm"
    vllm_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vllm_path.chmod(0o755)
    return python_path


def test_create_serving_backend_returns_huggingface_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving factory should keep the in-process backend as the default."""
    monkeypatch.setattr(huggingface_module, "ActorModel", LocalActor)

    backend = create_serving_backend(ServingConfig(model_name="fake/model"))

    assert isinstance(backend, HuggingFaceServingBackend)


def test_vllm_backend_rejects_debug_live_rollout(tmp_path: Path) -> None:
    """`vllm` should fail fast when serving debug live rollout is enabled."""
    runtime_python = _make_fake_runtime(tmp_path)

    with pytest.raises(ValueError, match="debug_live_rollout"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(runtime_python),
                debug_live_rollout=True,
            )
        )


def test_vllm_backend_rejects_missing_runtime_python(tmp_path: Path) -> None:
    """`vllm` should fail fast when the configured runtime interpreter is missing."""
    missing_python = tmp_path / "missing-python"

    with pytest.raises(ValueError, match="runtime_python does not exist"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(missing_python),
            )
        )


def test_vllm_backend_rejects_reserved_vllm_args(tmp_path: Path) -> None:
    """FlashRL-owned launch flags should not be overridable via `vllm_args`."""
    runtime_python = _make_fake_runtime(tmp_path)

    with pytest.raises(ValueError, match="reserved flag"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(runtime_python),
                vllm_args=["--host=0.0.0.0"],
            )
        )


def test_create_serving_backend_returns_vllm_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The serving factory should dispatch to the managed `vllm` backend."""
    runtime_python = _make_fake_runtime(tmp_path)
    spawned_commands: list[list[str]] = []

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100 + len(spawned_commands))
    monkeypatch.setattr(
        VLLMServingBackend,
        "_spawn_process",
        lambda self, command: spawned_commands.append(command) or FakeProcess(),
    )
    monkeypatch.setattr(
        VLLMServingBackend,
        "_request_json",
        lambda self, url, *, method, payload=None, timeout: {"data": [{"id": "fake/model"}]},
    )

    backend = create_serving_backend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(runtime_python),
            num_replicas=2,
        )
    )

    assert isinstance(backend, VLLMServingBackend)
    assert backend._snapshot_dir is None
    assert len(spawned_commands) == 2
    assert spawned_commands[0][:3] == [str(runtime_python.with_name("vllm")), "serve", "fake/model"]
    assert "--served-model-name" in spawned_commands[0]
    backend.close()


def test_vllm_backend_sync_generate_and_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The backend should route completions over HTTP, restart on sync, and clean up."""
    runtime_python = _make_fake_runtime(tmp_path)
    spawned_commands: list[list[str]] = []
    processes: list[FakeProcess] = []
    requests: list[tuple[str, str, dict[str, object] | None]] = []
    ports = iter([8100, 8101, 8200, 8201])

    def fake_spawn(self, command):
        spawned_commands.append(command)
        process = FakeProcess()
        processes.append(process)
        return process

    def fake_request(self, url, *, method, payload=None, timeout):
        del timeout
        requests.append((method, url, payload))
        if method == "GET":
            return {"data": [{"id": self.config.model_name}]}
        assert payload is not None
        prompts = payload["prompt"]
        assert isinstance(prompts, list)
        grouped_choices = []
        for prompt in prompts:
            prompt_token_ids = [ord(char) for char in prompt] or [0]
            for candidate_index in range(int(payload["n"])):
                grouped_choices.append(
                    {
                        "text": f"vllm::{prompt}::{candidate_index}",
                        "prompt_token_ids": prompt_token_ids,
                        "token_ids": [10 + candidate_index, 20 + candidate_index],
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                        "finish_reason": "stop",
                    }
                )
        return {"choices": grouped_choices}

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: next(ports))
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", fake_spawn)
    monkeypatch.setattr(VLLMServingBackend, "_request_json", fake_request)
    monkeypatch.setattr(
        vllm_module.tempfile,
        "mkdtemp",
        lambda prefix: str(tmp_path / "snapshot"),
    )

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(runtime_python),
            num_replicas=2,
            vllm_args=["--max-num-seqs=32"],
        )
    )
    grouped = backend.generate_grouped(["ab", "cd", "ef"], group_size=2)
    training_actor = SnapshotTrainingActor()
    backend.sync_from_training_actor(training_actor)

    completion_requests = [
        payload
        for method, _, payload in requests
        if method == "POST" and payload is not None
    ]
    assert len(grouped) == 3
    assert [sample.text for sample in grouped[0]] == ["vllm::ab::0", "vllm::ab::1"]
    assert [sample.text for sample in grouped[1]] == ["vllm::cd::0", "vllm::cd::1"]
    assert [sample.text for sample in grouped[2]] == ["vllm::ef::0", "vllm::ef::1"]
    assert any(payload["return_token_ids"] is True for payload in completion_requests)
    assert sorted(len(payload["prompt"]) for payload in completion_requests) == [1, 2]
    assert backend._snapshot_dir is not None
    assert (backend._snapshot_dir / "model.safetensors").exists()
    assert len(training_actor.model.saved_paths) == 1
    assert len(spawned_commands) == 4
    assert spawned_commands[0][2] == "fake/model"
    assert spawned_commands[2][2] == str(backend._snapshot_dir)
    assert processes[0].terminate_calls == 1
    assert processes[1].terminate_calls == 1

    snapshot_dir = backend._snapshot_dir
    backend.close()

    assert snapshot_dir is not None
    assert not snapshot_dir.exists()
    assert processes[2].terminate_calls == 1
    assert processes[3].terminate_calls == 1


def test_vllm_backend_close_uses_process_group_teardown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed vLLM replicas should be torn down via their process group."""
    runtime_python = _make_fake_runtime(tmp_path)
    killpg_calls: list[tuple[int, int]] = []
    process = FakeProcess(pid=4321)

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100)
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", lambda self, command: process)
    monkeypatch.setattr(
        VLLMServingBackend,
        "_request_json",
        lambda self, url, *, method, payload=None, timeout: {"data": [{"id": self.config.model_name}]},
    )
    monkeypatch.setattr(vllm_module.os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(
        vllm_module.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(runtime_python),
        )
    )
    backend.close()

    assert killpg_calls == [(4322, vllm_module.signal.SIGTERM)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_flashrl_from_yaml_parses_vllm_serving_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """YAML-driven runs should preserve serving backend selection and raw vLLM args."""
    runtime_python = _make_fake_runtime(tmp_path)
    captured_config: list[ServingConfig] = []

    class FakeServingBackend:
        def __init__(self, config) -> None:
            captured_config.append(config)
            self.config = config
            self._actor = TinyActor(bias_shift=0.1)
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
            del training_actor

        def set_live_rollout_debug(self, callback, context) -> None:
            del callback, context

        def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
            del candidate_index

        def clear_live_rollout_debug(self) -> None:
            return None

        def close(self) -> None:
            return None

    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "common": {"model_name": "fake/model"},
                "training": {"batch_size": 2, "max_epochs": 1, "num_threads": 1},
                "serving": {
                    "backend": "vllm",
                    "runtime_python": str(runtime_python),
                    "num_replicas": 2,
                    "vllm_args": ["--max-num-seqs=32", "--enable-prefix-caching"],
                },
                "grpo": {"group_size": 2, "clip_ratio": 0.2, "kl_coefficient": 0.0},
                "logging": {"log_dir": str(tmp_path / "logs"), "console": False, "file": True},
                "metrics": {"enabled": False},
                "runtime": {"reference_enabled": False},
                "hooks": {
                    "rollout_fn": "tests.test_vllm_backend:build_rollout_fn",
                    "reward_fn": "tests.test_vllm_backend:failing_reward_fn",
                    "dataset_fn": "tests.test_vllm_backend:dataset_fn",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(flashrl_module, "TrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config: FakeServingBackend(config),
    )

    trainer = FlashRL.from_yaml(config_path)

    assert trainer.serving_config.backend == "vllm"
    assert trainer.serving_config.runtime_python == str(runtime_python)
    assert trainer.serving_config.num_replicas == 2
    assert trainer.serving_config.vllm_args == ["--max-num-seqs=32", "--enable-prefix-caching"]
    assert captured_config[0].backend == "vllm"


def test_flashrl_train_failure_closes_serving_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Training failures should close the serving backend worker."""
    serving_backend = ClosableServingBackend()
    monkeypatch.setattr(flashrl_module, "TrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config: serving_backend,
    )

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=build_rollout_fn,
        reward_fn=failing_reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
        metrics_config=MetricsConfig(enabled=False),
    )

    with pytest.raises(ValueError, match="reward failure"):
        trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    assert serving_backend.close_calls == 1
