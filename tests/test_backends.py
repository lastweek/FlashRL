"""Direct unit tests for training and serving backends."""

from __future__ import annotations

import pytest
import torch

import flashrl.framework.backends.serving as serving_module
import flashrl.framework.backends.training as training_module
from flashrl.framework.backends.serving import ServingBackend
from flashrl.framework.backends.training import TrainingBackend
from flashrl.framework.config import ModelConfig
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

    backend = TrainingBackend(
        ModelConfig(model_name="fake/model", num_threads=3),
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

    backend = ServingBackend(ModelConfig(model_name="fake/model", num_threads=2))

    assert thread_calls == [2]
    assert backend.actor.model.training is False


def test_training_backend_checkpoint_round_trip_and_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Checkpoints should round-trip weights and sync should copy them to serving."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(serving_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)
    monkeypatch.setattr(serving_module, "ActorModel", BackendActor)

    config = ModelConfig(model_name="fake/model", num_threads=1)
    training_backend = TrainingBackend(config, learning_rate=1e-3)
    serving_backend = ServingBackend(config)

    with torch.no_grad():
        training_backend.actor.model.logit_bias.fill_(2.0)
        serving_backend.actor.model.logit_bias.fill_(0.0)

    checkpoint_path = tmp_path / "backend.pt"
    training_backend.save_checkpoint(str(checkpoint_path))

    with torch.no_grad():
        training_backend.actor.model.logit_bias.fill_(5.0)
    training_backend.load_checkpoint(str(checkpoint_path))

    assert torch.allclose(
        training_backend.actor.model.logit_bias,
        torch.full_like(training_backend.actor.model.logit_bias, 2.0),
    )

    training_backend.sync_weights_to(serving_backend)

    assert torch.allclose(
        training_backend.actor.model.logit_bias,
        serving_backend.actor.model.logit_bias,
    )

