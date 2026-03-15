"""Direct unit tests for training and serving backends."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import flashrl.framework.training.huggingface as training_module
from flashrl.framework.training import (
    FSDP2TrainingBackend,
    HuggingFaceTrainingBackend,
    create_training_backend,
)
from flashrl.framework.config import ModelConfig, ServingConfig, TrainingConfig
import flashrl.framework.serving.huggingface as serving_module
from flashrl.framework.serving import HuggingFaceServingBackend
from tests.conftest import TinyActor

pytestmark = pytest.mark.unit


class BackendActor(TinyActor):
    """Tiny actor subclass used for backend construction tests."""

    def __init__(self, config: ModelConfig) -> None:
        self.config_snapshot = config
        super().__init__(bias_shift=0.15)


def test_training_backend_initializes_train_mode_optimizer_and_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training backend should set threads, construct the actor, and enter train mode."""
    thread_calls: list[int] = []
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: thread_calls.append(value))
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)

    backend = HuggingFaceTrainingBackend(
        TrainingConfig(model_name="fake/model", num_threads=3),
        learning_rate=5e-4,
    )

    assert thread_calls == [3]
    assert backend.actor.model.training is True
    assert isinstance(backend.optimizer, torch.optim.Adam)
    assert backend.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_serving_backend_initializes_eval_mode_and_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving backend should set threads and leave the actor in eval mode."""
    thread_calls: list[int] = []
    monkeypatch.setattr(serving_module, "set_num_threads", lambda value: thread_calls.append(value))
    monkeypatch.setattr(serving_module, "ActorModel", BackendActor)

    backend = HuggingFaceServingBackend(
        ServingConfig(
            model_name="fake/model",
            num_threads=2,
            debug_live_rollout=True,
        )
    )

    assert thread_calls == [2]
    assert backend._actor.model.training is False
    assert backend.config.debug_live_rollout is True


def test_training_backend_checkpoint_round_trip_and_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Checkpoints should round-trip weights and sync should copy them to serving."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(serving_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)
    monkeypatch.setattr(serving_module, "ActorModel", BackendActor)

    config = TrainingConfig(model_name="fake/model", num_threads=1)
    training_backend = HuggingFaceTrainingBackend(
        config,
        learning_rate=1e-3,
    )
    serving_backend = HuggingFaceServingBackend(
        ServingConfig(**config.model_dump(include={"model_name", "device", "dtype", "max_length", "load_in_8bit", "trust_remote_code", "num_threads", "metadata"}))
    )

    with torch.no_grad():
        training_backend.actor.model.logit_bias.fill_(2.0)
        serving_backend._actor.model.logit_bias.fill_(0.0)
    exported_state = training_backend.export_state()

    with torch.no_grad():
        training_backend.actor.model.logit_bias.fill_(5.0)
    training_backend.load_state(exported_state)

    assert torch.allclose(
        training_backend.actor.model.logit_bias,
        torch.full_like(training_backend.actor.model.logit_bias, 2.0),
    )

    training_backend.sync_weights_to(serving_backend)

    assert torch.allclose(
        training_backend.actor.model.logit_bias,
        serving_backend._actor.model.logit_bias,
    )


def test_training_backend_factory_selects_expected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training backend factory should select the requested implementation."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)
    huggingface = create_training_backend(
        TrainingConfig(model_name="fake/model"),
        learning_rate=1e-5,
        role="actor",
    )

    assert isinstance(huggingface, HuggingFaceTrainingBackend)


def test_fsdp2_training_backend_rejects_multi_rank_until_launcher_exists() -> None:
    """FSDP2 backend should reject dp_size > 1 until worker orchestration is enabled."""
    with pytest.raises(ValueError, match="dp_size must be 1"):
        FSDP2TrainingBackend(
            TrainingConfig(model_name="fake/model", backend="fsdp2", dp_size=2),
            learning_rate=1e-5,
        )
