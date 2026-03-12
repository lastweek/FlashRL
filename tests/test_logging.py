"""Tests for FlashRL logging and local runtime UX."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashrl.framework.flashrl as flashrl_module
from flashrl.framework import FlashRL, LoggingConfig, MetricsConfig
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


def read_rollouts(run_dir: Path) -> list[dict]:
    """Read full rollout history records from a run directory."""
    return [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_logger_start_run_creates_run_artifacts(tmp_path: Path) -> None:
    """Starting a run should create the run directory and log files."""
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
        model_name="org/model-name",
    )

    logger.start_run(
        dataset_size=4,
        batch_size=2,
        max_epochs=1,
        total_batches=2,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_enabled=False,
        reference_device="auto",
    )

    assert logger.run_dir.exists()
    assert (logger.run_dir / "events.jsonl").exists()
    assert (logger.run_dir / "console.log").exists()
    assert (logger.run_dir / "rollouts.jsonl").exists()

    events = read_events(logger.run_dir)
    assert any(event["event"] == "run_started" for event in events)

    logger.close()


def test_run_logger_sanitizes_model_name_in_run_dir(tmp_path: Path) -> None:
    """Model names with path separators should not leak into the directory layout."""
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
        model_name="Org/Model Name@v1",
    )

    assert "/" not in logger.run_dir.name
    assert "org-model-name-v1" in logger.run_dir.name


def test_run_logger_compact_console_groups_step_output(tmp_path: Path) -> None:
    """Compact mode should render grouped step blocks instead of repeated prefixes."""
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
            console_mode="compact",
        ),
        model_name="org/model-name",
    )

    logger.start_run(
        dataset_size=4,
        batch_size=2,
        max_epochs=1,
        total_batches=2,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_enabled=False,
        reference_device="auto",
    )
    logger.log_model_load(
        "training_backend",
        "completed",
        {
            "device": "cpu",
            "cpu_threads": 1,
            "duration_seconds": 1.25,
        },
    )
    logger.log_epoch_start(epoch=1, total_epochs=1, num_batches=2)
    logger.log_step_stage(
        {
            "step": 1,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 2,
            "batch_size": 2,
            "stage": "rollout",
            "latency_seconds": 0.25,
            "prompt_tokens_mean": 12.5,
            "prompt_tokens_max": 16,
            "response_tokens_mean": 24.0,
            "response_tokens_max": 30,
        }
    )
    logger.log_step_stage(
        {
            "step": 1,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 2,
            "batch_size": 2,
            "stage": "reward",
            "latency_seconds": 0.002,
            "reward_mean": 1.5,
            "reward_std": 0.5,
            "reward_min": 1,
            "reward_max": 2,
            "reward_per_item_mean_seconds": 0.001,
        }
    )
    logger.log_step_done(
        {
            "step": 1,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 2,
            "batch_size": 2,
            "loss": 0.125,
            "policy_loss": 0.1,
            "kl_divergence": 0.025,
            "reward_mean": 1.5,
            "response_tokens_total": 48,
            "tokens_per_second": 96.0,
            "step_duration_seconds": 0.5,
            "stage_timings": {
                "rollout": 0.25,
                "reward": 0.002,
                "reference_forward": 0.0,
            },
            "stage_order": ["rollout", "reward"],
            "reference_enabled": False,
            "reference_active": False,
            "dominant_stage": "rollout",
        }
    )
    logger.log_sample_preview(
        step=1,
        prompt="prompt text",
        response="response text",
        reward=1.5,
    )
    logger.log_epoch_summary(
        {
            "epoch": 1,
            "total_epochs": 1,
            "loss": 0.125,
            "reward": 1.5,
            "kl_divergence": 0.025,
            "tokens_per_second": 96.0,
            "duration_seconds": 0.75,
            "stage_totals": {"rollout": 0.25, "reward": 0.002},
        }
    )
    logger.finish_run(
        status="completed",
        total_steps=1,
        lifecycle_totals={"startup_total_seconds": 1.25, "training_loop_seconds": 0.75},
    )

    transcript = (logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "FlashRL training run\n  run      " in transcript
    assert "load training_backend" in transcript
    assert "step 1/2  epoch 1/1  batch 1/2  batch_size=2" in transcript
    assert "  rollout" in transcript
    assert "prompt_tok 12.5/16" in transcript
    assert "response_tok 24.0/30" in transcript
    assert "  reward" in transcript
    assert "mean 1.5000" in transcript
    assert "  done" in transcript
    assert "dominant rollout" in transcript
    assert "epoch 1/1 summary" in transcript
    assert "  lifecycle startup 1.250s | train_loop 750.0ms" in transcript
    assert "  stages    rollout 250.0ms" in transcript
    assert "step=1 epoch=1/1" not in transcript
    assert "sample" not in transcript

    events = read_events(logger.run_dir)
    assert any(event["event"] == "sample_preview" for event in events)

    logger.close()


def test_framework_is_the_only_supported_flashrl_import_surface() -> None:
    """FlashRL should live under flashrl.framework only."""
    namespace = {}

    exec(
        "from flashrl.framework import FlashRL as ImportedFlashRL, LoggingConfig as ImportedLoggingConfig",
        {},
        namespace,
    )

    assert namespace["ImportedFlashRL"] is FlashRL
    assert namespace["ImportedLoggingConfig"] is LoggingConfig

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.flashrl")


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
        metrics_config=MetricsConfig(enabled=False),
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
            console_mode="verbose",
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
    step_stages = [
        event["payload"]["stage"]
        for event in events
        if event["event"] == "step_stage" and event["payload"]["step"] == 1
    ]
    assert step_stages == [
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
        if event["event"] == "step_stage"
        and event["payload"]["step"] == 1
        and event["payload"]["stage"] == "rollout"
    )
    assert "prompt_tokens_mean" in rollout_event["payload"]
    assert "response_tokens_max" in rollout_event["payload"]
    assert "phase" not in rollout_event["payload"]

    reward_event = next(
        event
        for event in events
        if event["event"] == "step_stage"
        and event["payload"]["step"] == 1
        and event["payload"]["stage"] == "reward"
    )
    assert "reward_mean" in reward_event["payload"]
    assert "reward_std" in reward_event["payload"]

    step_done = next(event for event in events if event["event"] == "step_done")
    assert step_done["payload"]["reference_enabled"] is False
    assert step_done["payload"]["stage_timings"]["reference_forward"] == 0.0
    assert step_done["payload"]["stage_order"] == [
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
    assert step_done["payload"]["dominant_stage"] in step_done["payload"]["stage_timings"]
    assert "phase_breakdown" not in step_done["payload"]
    assert "phase_groups" not in step_done["payload"]
    assert "timings" not in step_done["payload"]

    epoch_summary = next(event for event in events if event["event"] == "epoch_summary")
    assert "stage_totals" in epoch_summary["payload"]
    assert "stage_percentages" in epoch_summary["payload"]
    assert epoch_summary["payload"]["stage_totals"].get("reference_forward", 0.0) == 0.0
    assert "hot_path_totals" not in epoch_summary["payload"]
    assert "phase_group_totals" not in epoch_summary["payload"]

    transcript = (logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "runtime=single-device-per-backend reference=disabled" in transcript
    assert "cpu_threads=1" in transcript
    assert "step=1 epoch=1/1 batch=1/3 batch_size=2 stage=rollout" in transcript
    assert "stage=reward" in transcript
    assert "stage=loss_assembly" in transcript
    assert "stage=backward" in transcript
    assert "stage=complete" in transcript
    assert "phase=" not in transcript
    assert "━" not in transcript
    assert "sample step" not in transcript

    rollouts = read_rollouts(logger.run_dir)
    assert len(rollouts) == len(dataset)
    first_rollout = rollouts[0]
    assert first_rollout["run_id"] == logger.run_id
    assert first_rollout["step"] == 1
    assert first_rollout["sample_index"] == 1
    assert first_rollout["prompt"]["text"] == "prompt 0"
    assert first_rollout["rollout"]["response_text"].startswith("sample detail")
    assert first_rollout["conversation"]["messages"][0]["role"] == "user"
    assert first_rollout["conversation"]["messages"][1]["role"] == "assistant"
    assert first_rollout["reward"]["value"] > 0.0

    logger.close()


def test_grpo_assemble_loss_uses_response_only_grpo_terms() -> None:
    """Loss assembly should use response-only GRPO terms and stored rollout log-probs."""
    training_backend = FakeTrainingBackend(config=None, learning_rate=1e-2)
    serving_backend = FakeServingBackend(config=None)
    reference = FakeReferenceModel(config=None)
    trainer = GRPOTrainer(
        config=TrainerConfig(
            batch_size=2,
            max_epochs=1,
            clip_epsilon=0.2,
            kl_coefficient=0.3,
        ),
        training_backend=training_backend,
        serving_backend=serving_backend,
        reference=reference,
        reward_fn=UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace()),
        rollout_generator=UserDefinedRollout(
            rollout_fn=make_rollout_fn(response_suffix="loss", repeat=2),
            actor=serving_backend.actor,
            config=SimpleNamespace(),
        ),
        run_logger=None,
    )

    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    rollouts = make_rollout_fn(response_suffix="loss", repeat=2)(prompts, serving_backend.actor)
    prompt_texts = [prompt.text for prompt in prompts]
    response_texts = [rollout.text for rollout in rollouts]
    full_texts = [
        prompt + response
        for prompt, response in zip(prompt_texts, response_texts, strict=True)
    ]

    actor = training_backend.actor
    full_inputs, _, _ = trainer._tokenize(actor, full_texts)
    prompt_inputs, _, _ = trainer._tokenize(actor, prompt_texts)
    input_ids = full_inputs["input_ids"].to(actor.device)
    attention_mask = full_inputs["attention_mask"].to(actor.device)
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

    actor_logits = actor.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    with torch.no_grad():
        ref_logits = reference.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    advantages = torch.tensor([1.25, -0.5], dtype=torch.float32)
    rollout_response_log_probs = trainer._compute_rollout_response_log_probs(
        prompt_texts,
        response_texts,
    )

    loss_no_ref, policy_no_ref, kl_no_ref, response_tokens_total = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.config.kl_coefficient,
        clip_epsilon=trainer.config.clip_epsilon,
    )

    assert loss_no_ref.item() == pytest.approx(policy_no_ref.item())
    assert kl_no_ref.item() == pytest.approx(0.0)
    assert response_tokens_total == sum(len(sample) for sample in rollout_response_log_probs)

    prompt_only_mutated_logits = actor_logits.clone()
    for index, prompt_length in enumerate(prompt_lengths.tolist()):
        prompt_positions = max(int(prompt_length) - 1, 0)
        if prompt_positions == 0:
            continue
        prompt_only_mutated_logits[index, :prompt_positions, 0] += 25.0

    mutated_loss, mutated_policy, mutated_kl, _ = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=prompt_only_mutated_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.config.kl_coefficient,
        clip_epsilon=trainer.config.clip_epsilon,
    )

    assert mutated_loss.item() == pytest.approx(loss_no_ref.item())
    assert mutated_policy.item() == pytest.approx(policy_no_ref.item())
    assert mutated_kl.item() == pytest.approx(0.0)

    loss_with_ref, policy_with_ref, kl_with_ref, _ = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.config.kl_coefficient,
        clip_epsilon=trainer.config.clip_epsilon,
    )

    expected_total = policy_with_ref.item() + trainer.config.kl_coefficient * kl_with_ref.item()
    assert loss_with_ref.item() == pytest.approx(expected_total)
    assert kl_with_ref.item() >= 0.0

    shifted_rollout_log_probs = [
        [value - 0.5 for value in sample]
        for sample in rollout_response_log_probs
    ]
    shifted_loss, shifted_policy, _, _ = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=shifted_rollout_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.config.kl_coefficient,
        clip_epsilon=trainer.config.clip_epsilon,
    )

    assert shifted_policy.item() != pytest.approx(policy_no_ref.item())
    assert shifted_loss.item() != pytest.approx(loss_no_ref.item())
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
        metrics_config=MetricsConfig(enabled=False),
    )
    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]

    trainer.train(dataset)
    checkpoint_path = tmp_path / "flashrl-checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    trainer.load_checkpoint(str(checkpoint_path))

    output = capsys.readouterr().out
    assert "FlashRL training run" in output
    assert "  run      " in output
    assert "cpu=1" in output
    assert "reference=disabled" in output
    assert "load training_backend" in output
    assert "load serving_backend" in output
    assert "loaded reference_model" not in output
    assert "step 1/2  epoch 1/1  batch 1/2  batch_size=2" in output
    assert "  rollout" in output
    assert "  reward" in output
    assert "  tokenize_full" in output
    assert "  backward" in output
    assert "  done" in output
    assert "epoch 1/1 summary" in output
    assert "  lifecycle " in output
    assert "  stages    " in output
    assert "step=1 epoch=1/1" not in output
    assert "stage=rollout" not in output
    assert "sample" not in output
    assert "phase=" not in output
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
    assert "stage_totals" in run_finished["payload"]
    assert "stage_percentages" in run_finished["payload"]
    assert "startup_total_seconds" in run_finished["payload"]["lifecycle_totals"]
    assert run_finished["payload"]["no_training_steps_completed"] is False
    assert "hot_path_totals" not in run_finished["payload"]
    assert "phase_group_totals" not in run_finished["payload"]

    rollouts = read_rollouts(trainer._run_logger.run_dir)
    assert len(rollouts) == len(dataset)
    assert rollouts[0]["prompt"]["text"] == "prompt 0"
    assert rollouts[0]["reward"]["value"] > 0.0
    assert rollouts[0]["conversation"]["messages"][0]["content"] == "prompt 0"

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
        metrics_config=MetricsConfig(enabled=False),
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
    assert step_event["payload"]["stage_timings"]["reference_forward"] > 0
    assert "reference_forward" in step_event["payload"]["stage_order"]
    assert any(
        event["event"] == "step_stage"
        and event["payload"].get("stage") == "reference_forward"
        for event in events
    )

    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "reference=enabled" in transcript
    assert "  reference_forward" in transcript
    assert "full_tok_total" in transcript

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
        metrics_config=MetricsConfig(enabled=False),
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
        metrics_config=MetricsConfig(enabled=False),
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
    assert run_finished["payload"]["stage_totals"] == {}
    step_stages = [event for event in events if event["event"] == "step_stage"]
    assert len(step_stages) == 1
    assert step_stages[0]["payload"]["stage"] == "rollout"
    assert all(event["event"] != "step_done" for event in events)

    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "no training steps completed" in transcript

    trainer._run_logger.close()
