"""Tests for FlashRL logging and local runtime UX."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashrl.flashrl as flashrl_module
from flashrl import FlashRL, LoggingConfig
from flashrl.framework.config import TrainerConfig
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
)
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.run_logger import RunLogger
from flashrl.framework.trainer.grpo import GRPOTrainer


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
    """Training backend used for logging tests."""

    init_count = 0

    def __init__(self, config, learning_rate: float = 1e-5) -> None:
        del config
        type(self).init_count += 1
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
    """Serving backend used for logging tests."""

    init_count = 0

    def __init__(self, config) -> None:
        del config
        type(self).init_count += 1
        self.actor = FakeActor(bias_shift=0.1)
        self.actor.eval()


class FakeReferenceModel:
    """Reference model used for KL computation in tests."""

    init_count = 0

    def __init__(self, config) -> None:
        del config
        type(self).init_count += 1
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


def failing_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Raise to test exception logging."""
    raise ValueError(f"reward failure for {rollout.text[:12]}")


def patch_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch FlashRL to use fake local backends."""
    FakeTrainingBackend.init_count = 0
    FakeServingBackend.init_count = 0
    FakeReferenceModel.init_count = 0
    monkeypatch.setattr(flashrl_module, "TrainingBackend", FakeTrainingBackend)
    monkeypatch.setattr(flashrl_module, "ServingBackend", FakeServingBackend)
    monkeypatch.setattr(flashrl_module, "ReferenceModel", FakeReferenceModel)


def read_events(run_dir: Path) -> list[dict]:
    """Read structured JSONL events from a run directory."""
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_flashrl_eagerly_initializes_runtime_and_reuses_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FlashRL should build runtime components in __init__ and reuse them across runs."""
    patch_backends(monkeypatch)
    counts = {
        "rollout": 0,
        "reward": 0,
        "trainer": 0,
    }

    class CountingRollout(UserDefinedRollout):
        def __init__(self, *args, **kwargs) -> None:
            counts["rollout"] += 1
            super().__init__(*args, **kwargs)

    class CountingReward(UserDefinedReward):
        def __init__(self, *args, **kwargs) -> None:
            counts["reward"] += 1
            super().__init__(*args, **kwargs)

    class CountingTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs) -> None:
            counts["trainer"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(flashrl_module, "UserDefinedRollout", CountingRollout)
    monkeypatch.setattr(flashrl_module, "UserDefinedReward", CountingReward)
    monkeypatch.setattr(flashrl_module, "GRPOTrainer", CountingTrainer)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="reuse", repeat=2),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )
    assert FakeTrainingBackend.init_count == 1
    assert FakeServingBackend.init_count == 1
    assert counts == {
        "rollout": 1,
        "reward": 1,
        "trainer": 1,
    }

    component_ids = {
        "training_backend": id(trainer._training_backend),
        "serving_backend": id(trainer._serving_backend),
        "rollout": id(trainer._rollout_generator),
        "reward": id(trainer._reward),
        "trainer": id(trainer._trainer),
    }
    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]

    trainer.train(dataset)
    first_run_dir = trainer._run_logger.run_dir
    trainer.train(dataset)
    second_run_dir = trainer._run_logger.run_dir

    assert first_run_dir != second_run_dir
    assert component_ids == {
        "training_backend": id(trainer._training_backend),
        "serving_backend": id(trainer._serving_backend),
        "rollout": id(trainer._rollout_generator),
        "reward": id(trainer._reward),
        "trainer": id(trainer._trainer),
    }
    assert FakeTrainingBackend.init_count == 1
    assert FakeServingBackend.init_count == 1
    assert counts == {
        "rollout": 1,
        "reward": 1,
        "trainer": 1,
    }
    assert trainer._trainer.total_steps == 2

    events = read_events(second_run_dir)
    assert any(
        event["event"] == "model_load"
        and event["payload"].get("component") == "training_backend"
        and event["payload"].get("status") == "completed"
        for event in events
    )

    trainer._run_logger.close()


def test_grpo_trainer_without_reference_logs_append_only_hot_path(tmp_path: Path) -> None:
    """Default local trainer path should run without a reference model."""
    training_backend = FakeTrainingBackend(config=None, learning_rate=1e-2)
    serving_backend = FakeServingBackend(config=None)
    rollout = UserDefinedRollout(
        rollout_fn=make_rollout_fn(response_suffix="sample", repeat=20),
        actor=serving_backend.actor,
        config=SimpleNamespace(),
    )
    reward = UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace())
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
            log_every_steps=1,
            sample_every_steps=2,
        ),
        model_name="fake/model",
    )
    trainer = GRPOTrainer(
        config=TrainerConfig(batch_size=2, max_epochs=1, kl_coefficient=0.0),
        training_backend=training_backend,
        serving_backend=serving_backend,
        reference=None,
        reward_fn=reward,
        rollout_generator=rollout,
        run_logger=logger,
    )
    dataset = [Prompt(text=f"prompt {index}") for index in range(6)]

    logger.start_run(
        dataset_size=len(dataset),
        batch_size=2,
        max_epochs=1,
        total_batches=3,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_enabled=False,
        reference_device="auto",
    )
    trainer.train(dataset)
    logger.finish_run(status="completed", total_steps=trainer.total_steps)

    events = read_events(logger.run_dir)
    step_phases = [
        event["payload"]["phase"]
        for event in events
        if event["event"] == "step_phase" and event["payload"]["step"] == 1
    ]
    assert step_phases == [
        "rollout",
        "reward",
        "advantage",
        "tokenize_full",
        "tokenize_prompt",
        "actor_forward",
        "loss_assembly",
        "backward",
        "optimizer",
        "sync",
    ]
    rollout_event = next(
        event
        for event in events
        if event["event"] == "step_phase"
        and event["payload"]["step"] == 1
        and event["payload"]["phase"] == "rollout"
    )
    assert "prompt_tokens_mean" in rollout_event["payload"]
    assert "response_tokens_max" in rollout_event["payload"]

    reward_event = next(
        event
        for event in events
        if event["event"] == "step_phase"
        and event["payload"]["step"] == 1
        and event["payload"]["phase"] == "reward"
    )
    assert "reward_mean" in reward_event["payload"]
    assert "reward_std" in reward_event["payload"]

    step_done = next(event for event in events if event["event"] == "step_done")
    assert step_done["payload"]["reference_enabled"] is False
    assert step_done["payload"]["timings"]["reference_forward_seconds"] == 0.0
    assert [entry["name"] for entry in step_done["payload"]["phase_breakdown"]] == [
        "rollout",
        "reward",
        "advantage",
        "tokenize_full",
        "tokenize_prompt",
        "actor_forward",
        "loss_assembly",
        "backward",
        "optimizer",
        "sync",
    ]
    assert set(step_done["payload"]["phase_groups"]) == {
        "rollout",
        "reward",
        "calculate_loss",
        "train",
    }

    epoch_summary = next(event for event in events if event["event"] == "epoch_summary")
    assert "hot_path_totals" in epoch_summary["payload"]
    assert "hot_path_percentages" in epoch_summary["payload"]
    assert "phase_group_totals" in epoch_summary["payload"]
    assert "phase_group_percentages" in epoch_summary["payload"]
    assert epoch_summary["payload"]["hot_path_totals"].get("reference_forward_seconds", 0.0) == 0.0

    transcript = (logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "runtime=single-device-per-backend reference=disabled" in transcript
    assert "cpu_threads=1" in transcript
    assert "step=1 epoch=1/1 batch=1/3 batch_size=2 phase=rollout stage=rollout" in transcript
    assert "phase=reward stage=reward" in transcript
    assert "phase=loss_assembly stage=calculate_loss" in transcript
    assert "phase=backward stage=train" in transcript
    assert "phase=step_done stage=complete" in transcript
    assert "reference_forward=" not in transcript
    assert "━" not in transcript

    logger.close()


def test_flashrl_default_path_skips_reference_and_uses_append_only_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FlashRL should skip the reference model by default."""
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="console", repeat=8),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=True,
            file=True,
        ),
    )
    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]

    trainer.train(dataset)
    checkpoint_path = tmp_path / "flashrl-checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    trainer.load_checkpoint(str(checkpoint_path))

    output = capsys.readouterr().out
    assert "FlashRL training run" in output
    assert "cpu_threads=1" in output
    assert "reference=disabled" in output
    assert "loaded reference_model" not in output
    assert "epoch 1 summary" in output
    assert "lifecycle " in output
    assert "phase=rollout stage=rollout" in output
    assert "phase=reward stage=reward" in output
    assert "phase=tokenize_full stage=calculate_loss" in output
    assert "phase=backward stage=train" in output
    assert "phase=step_done stage=complete" in output
    assert "stage initializing" not in output
    assert "━" not in output

    events = read_events(trainer._run_logger.run_dir)
    assert all(
        event["payload"].get("component") != "reference_model"
        for event in events
        if event["event"] == "model_load"
    )
    run_finished = next(event for event in events if event["event"] == "run_finished")
    assert "lifecycle_totals" in run_finished["payload"]
    assert "hot_path_totals" in run_finished["payload"]
    assert "phase_group_totals" in run_finished["payload"]
    assert "startup_total_seconds" in run_finished["payload"]["lifecycle_totals"]
    assert run_finished["payload"]["no_training_steps_completed"] is False

    trainer._run_logger.close()


def test_flashrl_reference_enabled_loads_reference_and_logs_kl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enabling the reference should load it and expose KL-related metrics."""
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="kl", repeat=8),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=1,
        kl_coefficient=0.1,
        reference_enabled=True,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )
    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]

    trainer.train(dataset)
    events = read_events(trainer._run_logger.run_dir)
    assert any(
        event["event"] == "model_load"
        and event["payload"].get("component") == "reference_model"
        and event["payload"].get("status") == "completed"
        for event in events
    )
    step_event = next(event for event in events if event["event"] == "step_done")
    assert step_event["payload"]["reference_enabled"] is True
    assert step_event["payload"]["reference_active"] is True
    assert step_event["payload"]["timings"]["reference_forward_seconds"] > 0
    assert any(
        entry["name"] == "reference_forward"
        for entry in step_event["payload"]["phase_breakdown"]
    )
    assert any(
        event["event"] == "step_phase"
        and event["payload"].get("phase") == "reference_forward"
        for event in events
    )

    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "phase=reference_forward stage=calculate_loss" in transcript
    assert "full_tokens_total=" in transcript

    trainer._run_logger.close()


def test_flashrl_checkpoint_works_before_train_and_resume_skips_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Checkpoint load/save should work before train and preserve resume state."""
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="resume", repeat=3),
        reward_fn=reward_fn,
        batch_size=2,
        max_epochs=3,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )
    checkpoint_path = tmp_path / "resume-checkpoint.pt"

    trainer._trainer.current_epoch = 1
    trainer._trainer.total_steps = 7
    trainer.save_checkpoint(str(checkpoint_path))
    trainer._trainer.reset_state()

    trainer.load_checkpoint(str(checkpoint_path))
    assert trainer._trainer.current_epoch == 1
    assert trainer._trainer.total_steps == 7

    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]
    trainer.train(dataset)

    assert trainer._trainer.total_steps == 11
    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "epoch 2/3" in transcript
    assert "epoch 1/3" not in transcript

    trainer._run_logger.close()


def test_flashrl_no_step_summary_is_explicit_on_early_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the run fails before any training step finishes, the summary should say so."""
    patch_backends(monkeypatch)

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=make_rollout_fn(response_suffix="broken", repeat=4),
        reward_fn=failing_reward_fn,
        batch_size=2,
        max_epochs=1,
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
    )
    dataset = [Prompt(text="why did reward fail?")]

    with pytest.raises(ValueError, match="reward failure"):
        trainer.train(dataset)

    events = read_events(trainer._run_logger.run_dir)
    reward_exception = next(
        event
        for event in events
        if event["event"] == "exception"
        and event["payload"].get("context", {}).get("stage") == "reward"
    )
    assert "prompt_preview" in reward_exception["payload"]["context"]
    assert "response_preview" in reward_exception["payload"]["context"]

    run_finished = next(event for event in events if event["event"] == "run_finished")
    assert run_finished["payload"]["no_training_steps_completed"] is True
    assert run_finished["payload"]["hot_path_totals"] == {}
    step_phases = [event for event in events if event["event"] == "step_phase"]
    assert len(step_phases) == 1
    assert step_phases[0]["payload"]["phase"] == "rollout"
    assert all(event["event"] != "step_done" for event in events)

    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "no training steps completed" in transcript

    trainer._run_logger.close()
