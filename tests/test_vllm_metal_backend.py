"""Tests for the vllm_metal serving backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml

import flashrl.framework.flashrl as flashrl_module
import flashrl.framework.serving.huggingface as huggingface_module
import flashrl.framework.serving as serving_module
import flashrl.framework.serving.vllm_metal as vllm_metal_module
from flashrl.framework.serving import (
    HuggingFaceServingBackend,
    ServingBackend,
    create_serving_backend,
)
from flashrl.framework.serving.vllm_metal import VLLMMetalServingBackend
from flashrl.framework.config import LoggingConfig, MetricsConfig, ServingConfig
from flashrl.framework.data_models import Message, Prompt, RewardOutput, RolloutOutput, Conversation
from flashrl.framework.flashrl import FlashRL
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


class FakeRpcClient:
    """Synchronous RPC stand-in for the worker subprocess."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def call(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        normalized_params = dict(params or {})
        self.calls.append((method, normalized_params))
        if method == "health":
            return {"ready": True}
        if method == "reload_weights":
            return {"reloaded": True}
        if method == "generate_grouped":
            prompts = [str(prompt) for prompt in normalized_params["prompts"]]
            group_size = int(normalized_params["group_size"])
            grouped = []
            for prompt in prompts:
                prompt_token_ids = [ord(char) for char in prompt] or [0]
                grouped.append(
                    [
                        {
                            "text": f"vllm::{prompt}::{candidate_index}",
                            "prompt_token_ids": prompt_token_ids,
                            "response_token_ids": [10 + candidate_index, 20 + candidate_index],
                            "response_token_logprobs": [-0.1, -0.2],
                            "log_prob": -0.3,
                            "metadata": {"finish_reason": "stop"},
                        }
                        for candidate_index in range(group_size)
                    ]
                )
            return {"grouped_samples": grouped}
        if method == "shutdown":
            return {"shutdown": True}
        raise AssertionError(f"unexpected RPC method: {method}")

    def close(self) -> None:
        self.closed = True


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


def test_create_serving_backend_returns_huggingface_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving factory should keep the in-process backend as the default."""
    monkeypatch.setattr(huggingface_module, "ActorModel", LocalActor)

    backend = create_serving_backend(ServingConfig(model_name="fake/model"))

    assert isinstance(backend, HuggingFaceServingBackend)


def test_vllm_metal_backend_rejects_debug_live_rollout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """vllm_metal should fail fast when serving debug live rollout is enabled."""
    monkeypatch.setattr(vllm_metal_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vllm_metal_module.platform, "machine", lambda: "arm64")
    with pytest.raises(ValueError, match="debug_live_rollout"):
        VLLMMetalServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm_metal",
                runtime_python=sys.executable,
                debug_live_rollout=True,
            ),
            training_actor=SnapshotTrainingActor(),
        )


def test_vllm_metal_backend_rejects_missing_runtime_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """vllm_metal should fail fast when the configured runtime interpreter is missing."""
    missing_python = tmp_path / "missing-python"
    monkeypatch.setattr(vllm_metal_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vllm_metal_module.platform, "machine", lambda: "arm64")

    with pytest.raises(ValueError, match="runtime_python does not exist"):
        VLLMMetalServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm_metal",
                runtime_python=str(missing_python),
            ),
            training_actor=SnapshotTrainingActor(),
        )


def test_create_serving_backend_returns_vllm_metal_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The serving factory should dispatch to the vllm_metal backend."""
    fake_rpc = FakeRpcClient()
    monkeypatch.setattr(vllm_metal_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vllm_metal_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(vllm_metal_module.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "snapshot"))
    monkeypatch.setattr(VLLMMetalServingBackend, "_launch_worker", lambda self: fake_rpc)

    backend = create_serving_backend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm_metal",
            runtime_python=sys.executable,
        ),
        training_actor=SnapshotTrainingActor(),
    )

    assert isinstance(backend, VLLMMetalServingBackend)
    assert ("health", {}) in fake_rpc.calls
    backend.close()


def test_vllm_metal_backend_sync_generate_and_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The backend should write snapshots, proxy grouped generation, and close the worker."""
    fake_rpc = FakeRpcClient()
    snapshot_dir = tmp_path / "snapshot"
    monkeypatch.setattr(vllm_metal_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vllm_metal_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(vllm_metal_module.tempfile, "mkdtemp", lambda prefix: str(snapshot_dir))
    monkeypatch.setattr(VLLMMetalServingBackend, "_launch_worker", lambda self: fake_rpc)
    training_actor = SnapshotTrainingActor()

    backend = VLLMMetalServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm_metal",
            runtime_python=sys.executable,
        ),
        training_actor=training_actor,
    )
    grouped = backend.generate_grouped(["ab"], group_size=2)
    backend.sync_from_training_actor(training_actor)

    assert (snapshot_dir / "model.safetensors").exists()
    backend.close()
    assert len(grouped) == 1
    assert [sample.text for sample in grouped[0]] == ["vllm::ab::0", "vllm::ab::1"]
    assert any(method == "reload_weights" for method, _ in fake_rpc.calls)
    assert fake_rpc.closed is True


def test_flashrl_from_yaml_parses_vllm_metal_serving_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """YAML-driven runs should preserve serving backend selection and runtime path."""
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
                    "backend": "vllm_metal",
                    "runtime_python": sys.executable,
                },
                "grpo": {"group_size": 2, "clip_ratio": 0.2, "kl_coefficient": 0.0},
                "logging": {"log_dir": str(tmp_path / "logs"), "console": False, "file": True},
                "metrics": {"enabled": False},
                "runtime": {"reference_enabled": False},
                "hooks": {
                    "rollout_fn": "tests.test_vllm_metal_backend:build_rollout_fn",
                    "reward_fn": "tests.test_vllm_metal_backend:failing_reward_fn",
                    "dataset_fn": "tests.test_vllm_metal_backend:dataset_fn",
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
        lambda config, training_actor=None: FakeServingBackend(config),
    )

    trainer = FlashRL.from_yaml(config_path)

    assert trainer.serving_config.backend == "vllm_metal"
    assert trainer.serving_config.runtime_python == sys.executable
    assert captured_config[0].backend == "vllm_metal"


def dataset_fn() -> list[Prompt]:
    """Small dataset for YAML config loading tests."""
    return [Prompt(text="prompt 0"), Prompt(text="prompt 1")]


def test_flashrl_train_failure_closes_serving_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Training failures should close the serving backend worker."""
    serving_backend = ClosableServingBackend()
    monkeypatch.setattr(flashrl_module, "TrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, training_actor=None: serving_backend,
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
