"""Direct unit tests for training and serving backends."""

from __future__ import annotations

from pathlib import Path
import threading
import time

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


def test_training_backend_forward_disables_transformer_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Learner forwards should disable autoregressive KV caching to limit memory growth."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)

    backend = HuggingFaceTrainingBackend(
        TrainingConfig(model_name="fake/model", num_threads=1),
        learning_rate=1e-3,
    )
    reference_backend = training_module.HuggingFaceReferenceBackend(
        TrainingConfig(model_name="fake/model", num_threads=1),
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    logits = backend.forward_logits(input_ids, attention_mask)
    reference_logits = reference_backend.forward_logits(input_ids, attention_mask)

    assert logits.shape == (1, 3, 32)
    assert reference_logits.shape == (1, 3, 32)
    assert backend.actor.model.last_forward_kwargs == {"use_cache": False}
    assert reference_backend.model_copy.model.last_forward_kwargs == {"use_cache": False}


def test_training_backend_optimizer_step_clears_gradients_after_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grad buffers should be released after the optimizer step to reduce idle device pressure."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)

    backend = HuggingFaceTrainingBackend(
        TrainingConfig(model_name="fake/model", num_threads=1),
        learning_rate=1e-3,
    )
    param = next(backend.actor.model.parameters())
    loss = param.sum()

    backend.backward_step(loss)()
    assert param.grad is not None

    backend.optimizer_step()

    assert all(parameter.grad is None for parameter in backend.actor.model.parameters())


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

    startup_version = serving_backend.current_weight_version()
    sync_info = training_backend.sync_weights_to(
        serving_backend,
        source_training_step=3,
        source_epoch=2,
        origin="sync",
    )

    assert torch.allclose(
        training_backend.actor.model.logit_bias,
        serving_backend._actor.model.logit_bias,
    )
    assert startup_version.version_id == 0
    assert startup_version.origin == "startup"
    assert sync_info.version_id == 1
    assert sync_info.source_training_step == 3
    assert sync_info.source_epoch == 2
    assert sync_info.origin == "sync"
    assert serving_backend.current_weight_version().version_id == 1
    assert serving_backend.export_weight_version_state()["next_version_id"] == 2


def test_serving_backend_restore_weight_version_state_keeps_resume_monotonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restored version counters should make the next activation monotonic on resume."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(serving_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)
    monkeypatch.setattr(serving_module, "ActorModel", BackendActor)

    training_backend = HuggingFaceTrainingBackend(
        TrainingConfig(model_name="fake/model", num_threads=1),
        learning_rate=1e-3,
    )
    serving_backend = HuggingFaceServingBackend(
        ServingConfig(model_name="fake/model", num_threads=1)
    )

    serving_backend.restore_weight_version_state({"schema_version": 1, "next_version_id": 7})
    resume_info = training_backend.sync_weights_to(
        serving_backend,
        source_training_step=11,
        source_epoch=4,
        origin="resume",
    )

    assert resume_info.version_id == 7
    assert resume_info.origin == "resume"
    assert resume_info.source_training_step == 11
    assert serving_backend.current_weight_version().version_id == 7
    assert serving_backend.export_weight_version_state()["next_version_id"] == 8


def test_huggingface_serving_backend_serializes_generate_during_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving lifecycle lock should prevent generate from overlapping with sync."""
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(serving_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)
    monkeypatch.setattr(serving_module, "ActorModel", BackendActor)

    training_backend = HuggingFaceTrainingBackend(
        TrainingConfig(model_name="fake/model", num_threads=1),
        learning_rate=1e-3,
    )
    serving_backend = HuggingFaceServingBackend(
        ServingConfig(model_name="fake/model", num_threads=1)
    )

    entered_sync = threading.Event()
    release_sync = threading.Event()
    generate_finished = threading.Event()
    original_load_state_dict = serving_backend._actor.model.load_state_dict

    def blocking_load_state_dict(state_dict, strict: bool = True):
        entered_sync.set()
        assert release_sync.wait(timeout=2.0)
        return original_load_state_dict(state_dict, strict=strict)

    monkeypatch.setattr(serving_backend._actor.model, "load_state_dict", blocking_load_state_dict)

    sync_thread = threading.Thread(
        target=lambda: training_backend.sync_weights_to(
            serving_backend,
            source_training_step=1,
            source_epoch=1,
            origin="sync",
        )
    )
    sync_thread.start()
    assert entered_sync.wait(timeout=2.0)

    generate_thread = threading.Thread(
        target=lambda: serving_backend.generate(["blocked"]) and generate_finished.set()
    )
    generate_thread.start()
    time.sleep(0.05)
    assert generate_finished.is_set() is False

    release_sync.set()
    sync_thread.join(timeout=2.0)
    generate_thread.join(timeout=2.0)

    assert generate_finished.is_set() is True
    assert serving_backend.current_weight_version().version_id == 1


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
