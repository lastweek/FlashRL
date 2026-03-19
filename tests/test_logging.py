"""Tests for FlashRL logging and local runtime UX."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashrl.framework.flashrl as flashrl_module
from flashrl.framework import FlashRL, GrpoConfig, LoggingConfig, MetricsConfig
from flashrl.framework.config import AdminConfig, RunConfig, ServingConfig, TrainerConfig, TrainingConfig
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    WeightVersionInfo,
)
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.run_logger import RunLogger, _factor_shared_messages
from flashrl.framework.training import ActorTrainingBackend, TrainingBackend
from flashrl.framework.trainer.grpo.trainer import GRPOTrainer


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


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

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        call_index = self._batch_call_index
        self._batch_call_index += 1
        outputs = []
        for prompt in prompts:
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
            outputs.append(
                SimpleNamespace(
                    text=response,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    response_token_logprobs=response_token_logprobs,
                    log_prob=float(sum(response_token_logprobs)),
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
        del callback, context

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        del candidate_index

    def clear_live_rollout_debug(self) -> None:
        return None


class FakeTrainingBackend(ActorTrainingBackend):
    """Training backend used for logging tests."""

    init_count = 0

    def __init__(
        self,
        config,
        learning_rate: float = 1e-5,
    ) -> None:
        type(self).init_count += 1
        resolved_config = config or TrainingConfig(model_name="fake/model", device="cpu")
        super().__init__(
            resolved_config,
            learning_rate=learning_rate,
        )
        self.model_copy = FakeActor(bias_shift=0.25)
        self.model_copy.train()
        self.device = self.model_copy.device
        self.optimizer = torch.optim.SGD(self.model_copy.model.parameters(), lr=learning_rate)
        self.startup_events = [
            {
                "component": "actor_backend",
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": resolved_config.num_threads,
                    "duration_seconds": 0.0,
                },
            }
        ]


def _encode_text(text: str, *, max_length: int = 128, vocab_size: int = 32) -> list[int]:
    tokens = [((ord(char) % (vocab_size - 1)) + 1) for char in text[:max_length]]
    return tokens or [1]


class FakeServingBackend:
    """Serving backend used for logging tests."""

    init_count = 0

    def __init__(self, config) -> None:
        type(self).init_count += 1
        self.config = config
        self._actor = FakeActor(bias_shift=0.1)
        self._actor.eval()
        self.device = self._actor.device
        self.generation_defaults: dict[str, object] = {}
        self._active_weight_version = WeightVersionInfo(
            version_id=0,
            source_training_step=None,
            source_epoch=None,
            activated_at="2026-03-19T00:00:00Z",
            model_source="fake-serving://startup",
            origin="startup",
        )
        self._next_weight_version_id = 1

    def generate(self, prompts: list[str], **kwargs):
        return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)
        self._actor.set_generation_defaults(**kwargs)

    def sync_from_training_actor(
        self,
        training_actor,
        *,
        source_training_step: int | None = None,
        source_epoch: int | None = None,
        origin: str = "sync",
    ) -> WeightVersionInfo:
        self._actor.model.load_state_dict(training_actor.model.state_dict())
        self._active_weight_version = WeightVersionInfo(
            version_id=self._next_weight_version_id,
            source_training_step=source_training_step,
            source_epoch=source_epoch,
            activated_at=f"2026-03-19T00:00:{self._next_weight_version_id:02d}Z",
            model_source=f"fake-serving://version-{self._next_weight_version_id}",
            origin=origin,
        )
        self._next_weight_version_id += 1
        return self.current_weight_version()

    def current_weight_version(self) -> WeightVersionInfo:
        return self._active_weight_version.model_copy(deep=True)

    def export_weight_version_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "next_version_id": self._next_weight_version_id,
        }

    def restore_weight_version_state(self, state: dict[str, object] | None) -> None:
        if not isinstance(state, dict):
            return
        next_version_id = state.get("next_version_id")
        if isinstance(next_version_id, int):
            self._next_weight_version_id = max(next_version_id, self._next_weight_version_id)

    def set_live_rollout_debug(self, callback, context) -> None:
        self._actor.set_live_rollout_debug(callback, context)

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        self._actor.set_live_rollout_candidate_index(candidate_index)

    def clear_live_rollout_debug(self) -> None:
        self._actor.clear_live_rollout_debug()

    def close(self) -> None:
        return None


class FakeReferenceModel(TrainingBackend):
    """Reference backend used for KL computation in tests."""

    init_count = 0

    def __init__(self, config) -> None:
        resolved_config = config or TrainingConfig(model_name="fake/model", device="cpu")
        type(self).init_count += 1
        super().__init__(resolved_config, role="reference")
        self.device = torch.device("cpu")
        self.model_copy = FakeActor(bias_shift=0.0)
        self.model_copy.eval()
        self.model_copy.model.requires_grad_(False)
        self.startup_events = [
            {
                "component": "reference_backend",
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": resolved_config.num_threads,
                    "duration_seconds": 0.0,
                },
            }
        ]


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


def failing_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Raise to test exception logging."""
    raise ValueError(f"reward failure for {rollout.text[:12]}")


def patch_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch FlashRL to use fake local backends."""
    FakeTrainingBackend.init_count = 0
    FakeServingBackend.init_count = 0
    FakeReferenceModel.init_count = 0
    monkeypatch.setattr(
        flashrl_module,
        "create_training_backend",
        lambda config, role, learning_rate=None: (
            FakeTrainingBackend(
                config,
                learning_rate=float(learning_rate or 1e-5),
            )
            if role == "actor"
            else FakeReferenceModel(config)
        ),
    )
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None, log_dir=None: FakeServingBackend(config),
    )


def build_flashrl(
    tmp_path: Path,
    *,
    rollout_callback,
    reward_callback,
    reference_config: TrainingConfig | None = None,
    serving_config: ServingConfig | None = None,
    trainer_config: TrainerConfig | None = None,
    grpo_config: GrpoConfig | None = None,
    logging_config: LoggingConfig | None = None,
    metrics_config: MetricsConfig | None = None,
) -> FlashRL:
    """Create a FlashRL instance through the explicit role-based config API."""
    return FlashRL(
        actor_config=TrainingConfig(model_name="fake/model", backend="huggingface", device="cpu"),
        reference_config=reference_config,
        serving_config=serving_config
        or ServingConfig(model_name="fake/model", backend="huggingface", device="cpu"),
        trainer_config=trainer_config or TrainerConfig(batch_size=2, max_epochs=1),
        grpo_config=grpo_config or GrpoConfig(group_size=2),
        rollout_fn=rollout_callback,
        reward_fn=reward_callback,
        logging_config=logging_config or LoggingConfig(log_dir=tmp_path, console=False, file=True),
        metrics_config=metrics_config or MetricsConfig(enabled=False),
        admin_config=AdminConfig(admin_enabled=True),
    )


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


def build_managed_run_config(
    *,
    log_dir: Path,
    checkpoint_dir: Path | None = None,
    save_every_steps: int | None = None,
    save_on_run_end: bool = False,
    resume_from: str | None = None,
    max_epochs: int = 1,
) -> dict[str, object]:
    """Build a minimal profile-style config for managed checkpoint tests."""
    checkpointing: dict[str, object] = {}
    if checkpoint_dir is not None:
        checkpointing["directory"] = str(checkpoint_dir)
    if save_every_steps is not None:
        checkpointing["save_every_steps"] = save_every_steps
    if save_on_run_end:
        checkpointing["save_on_run_end"] = True
    if resume_from is not None:
        checkpointing["resume_from"] = resume_from

    return {
        "actor": {
            "model_name": "fake/model",
            "backend": "huggingface",
            "device": "cpu",
        },
        "serving": {
            "model_name": "fake/model",
            "backend": "huggingface",
            "device": "cpu",
        },
        "trainer": {
            "learning_rate": 1.0e-5,
            "batch_size": 2,
            "max_epochs": max_epochs,
            "shuffle_each_epoch": False,
        },
        "grpo": {
            "group_size": 2,
        },
        "logging": {
            "log_dir": str(log_dir),
            "console": False,
            "file": True,
        },
        "metrics": {
            "enabled": False,
        },
        "checkpointing": checkpointing,
        "admin": {
            "admin_enabled": False,
        },
    }


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from terminal output."""
    return ANSI_ESCAPE_RE.sub("", text)


def test_factor_shared_messages_extracts_common_prefix() -> None:
    """Shared rollout context should be stored once per prompt group."""
    messages = [
        [
            {"role": "system", "content": "shared system"},
            {"role": "user", "content": "prompt 0"},
            {"role": "assistant", "content": "answer a"},
        ],
        [
            {"role": "system", "content": "shared system"},
            {"role": "user", "content": "prompt 0"},
            {"role": "assistant", "content": "answer b"},
        ],
    ]

    shared, suffixes = _factor_shared_messages(messages)

    assert shared == messages[0][:2]
    assert suffixes == [[messages[0][2]], [messages[1][2]]]


def test_factor_shared_messages_falls_back_without_common_prefix() -> None:
    """Divergent transcripts should not be force-compacted."""
    messages = [
        [{"role": "user", "content": "prompt a"}],
        [{"role": "assistant", "content": "answer b"}],
    ]

    shared, suffixes = _factor_shared_messages(messages)

    assert shared == []
    assert suffixes == messages


def test_rollout_schema_v2_deduplicates_transcript_payload(tmp_path: Path) -> None:
    """The compact rollout schema should serialize transcript data smaller than v1."""
    logger = RunLogger(
        LoggingConfig(log_dir=tmp_path, console=False, file=True),
        model_name="org/model-name",
    )
    prompt = Prompt(text="prompt 0", metadata={"task_id": "task-0", "source": "unit"})
    rollouts = [
        RolloutOutput(
            text="first answer",
            log_prob=-0.8,
            prompt_token_ids=[1, 2, 3],
            response_token_ids=[4, 5],
            response_token_logprobs=[-0.4, -0.4],
            conversation=Conversation(
                messages=[
                    Message(role="user", content="prompt 0"),
                    Message(role="assistant", content="first answer"),
                ]
            ),
            metadata={
                "finish_reason": "stop",
                "ttft_seconds": 0.1,
                "tpot_seconds": 0.02,
                "generation_seconds": 0.2,
                "response_token_count": 2,
                "prompt_metadata": {"task_id": "task-0", "source": "unit"},
            },
        ),
        RolloutOutput(
            text="second answer",
            log_prob=-1.2,
            prompt_token_ids=[1, 2, 3],
            response_token_ids=[6, 7, 8],
            response_token_logprobs=[-0.4, -0.4, -0.4],
            conversation=Conversation(
                messages=[
                    Message(role="user", content="prompt 0"),
                    Message(role="assistant", content="second answer"),
                ]
            ),
            metadata={
                "finish_reason": "length",
                "ttft_seconds": 0.15,
                "tpot_seconds": 0.03,
                "generation_seconds": 0.45,
                "response_token_count": 3,
                "prompt_metadata": {"task_id": "task-0", "source": "unit"},
            },
        ),
    ]
    rewards = [
        RewardOutput(
            reward=1.0,
            metadata={
                "pass_rate": 1.0,
                "passed_tests": 5,
                "total_tests": 5,
                "accuracy_pass": True,
                "format_pass": True,
                "truncated": False,
                "execution_seconds": 0.12,
                "failure_reason": None,
                "checker_used": True,
                "execution_status": "passed",
                "code_preview": "print(1)",
            },
        ),
        RewardOutput(
            reward=0.25,
            metadata={
                "pass_rate": 0.2,
                "passed_tests": 1,
                "total_tests": 5,
                "accuracy_pass": False,
                "format_pass": True,
                "truncated": True,
                "execution_seconds": 0.3,
                "failure_reason": "wrong_answer",
                "checker_used": True,
                "execution_status": "failed",
                "code_preview": "print(2)",
            },
        ),
    ]

    logger.log_rollout_batch(
        step=1,
        epoch=1,
        batch_index=1,
        batches_in_epoch=1,
        prompts=[prompt, prompt],
        rollouts=rollouts,
        rewards=rewards,
        prompt_indices=[0, 0],
        candidate_indices=[0, 1],
        group_size=2,
        prompt_count=1,
    )

    v2_record = read_rollouts(logger.run_dir)[0]
    legacy_record = {
        "run_id": logger.run_id,
        "run_index": logger.run_index,
        "step": 1,
        "epoch": 1,
        "batch_index": 1,
        "batches_in_epoch": 1,
        "prompt_index": 0,
        "group_size": 2,
        "prompt_count": 1,
        "sample_count": 2,
        "prompt": {
            "text": prompt.text,
            "metadata": prompt.metadata,
        },
        "candidates": [
            {
                "candidate_index": candidate_index,
                "rollout": {
                    "response_text": rollout.text,
                    "log_prob": rollout.log_prob,
                    "prompt_token_count": len(rollout.prompt_token_ids),
                    "response_token_count": len(rollout.response_token_ids),
                    "metadata": rollout.metadata,
                },
                "conversation": rollout.conversation.model_dump(mode="json"),
                "reward": {
                    "value": reward.reward,
                    "metadata": reward.metadata,
                },
            }
            for candidate_index, rollout, reward in zip([0, 1], rollouts, rewards, strict=True)
        ],
    }

    v2_transcript_payload = {
        "input": {
            "shared_messages": v2_record["input"]["shared_messages"],
            "prompt_preview": v2_record["input"]["prompt_preview"],
        },
        "candidates": [
            {
                "candidate_index": candidate["candidate_index"],
                "completion_messages": candidate["completion_messages"],
                "output_preview": candidate["output"]["preview"],
            }
            for candidate in v2_record["candidates"]
        ],
    }
    legacy_transcript_payload = {
        "prompt": legacy_record["prompt"],
        "candidates": [
            {
                "candidate_index": candidate["candidate_index"],
                "conversation": candidate["conversation"],
                "response_text": candidate["rollout"]["response_text"],
            }
            for candidate in legacy_record["candidates"]
        ],
    }

    assert len(json.dumps(v2_transcript_payload, ensure_ascii=True)) < len(
        json.dumps(legacy_transcript_payload, ensure_ascii=True)
    )

    logger.close()


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
        batch_size=4,
        max_epochs=1,
        total_batches=2,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_configured=False,
        group_size=2,
        clip_ratio=0.2,
        prompts_per_step=2,
        steps_per_epoch=2,
        total_planned_steps=2,
    )

    assert logger.run_dir.exists()
    assert (logger.run_dir / "events.jsonl").exists()
    assert (logger.run_dir / "console.log").exists()
    assert (logger.run_dir / "rollouts.jsonl").exists()
    assert not (logger.run_dir / "rollouts.html").exists()

    events = read_events(logger.run_dir)
    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["run_id"] == logger.run_id
    assert run_started["run_index"] == logger.run_index
    assert run_started["payload"]["run_index"] == logger.run_index

    logger.close()


def test_logging_config_defaults_to_visible_logs_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default logging should use a visible logs/ root instead of a hidden dotdir."""
    monkeypatch.chdir(tmp_path)

    default_config = LoggingConfig(console=False, file=True)
    assert default_config.log_dir == "logs"

    logger = RunLogger(default_config, model_name="org/model-name")
    assert logger.run_dir.parent.resolve() == (tmp_path / "logs").resolve()

    custom_root = tmp_path / "custom-logs"
    custom_logger = RunLogger(
        LoggingConfig(log_dir=custom_root, console=False, file=True),
        model_name="org/model-name",
    )
    assert custom_logger.run_dir.parent.resolve() == custom_root.resolve()


def test_run_logger_allocates_monotonic_run_indices(tmp_path: Path) -> None:
    """Run indices should increase monotonically under one log root."""
    config = LoggingConfig(
        log_dir=tmp_path,
        console=False,
        file=True,
    )
    first = RunLogger(config, model_name="org/model-a")
    second = RunLogger(config, model_name="org/model-b")

    assert second.run_index == first.run_index + 1
    assert first.run_dir.name.startswith(f"{first.run_index:06d}-")
    assert second.run_dir.name.startswith(f"{second.run_index:06d}-")


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
    assert logger.run_dir.name[:6].isdigit()
    assert logger.run_dir.name[6] == "-"
    assert "org-model-name-v1" in logger.run_dir.name


def test_flashrl_constructor_emits_bootstrap_stage_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Constructor startup should emit immediate stage lines before training begins."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(),
        reward_callback=reward_fn,
        trainer_config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        serving_config=flashrl_module.ServingConfig(
            model_name="fake/model",
            backend="vllm",
            num_replicas=2,
        ),
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=True,
            file=False,
        ),
        metrics_config=MetricsConfig(enabled=False),
    )

    output = capsys.readouterr().out
    assert "FlashRL startup\n  startup  model=fake/model  actor=huggingface  dp_size=1" in output
    assert "serving=vllm replicas=2  reference=disabled" in output
    assert "  startup  actor_backend     starting backend=huggingface dp_size=1" in output
    assert "  ready    actor_backend     backend=huggingface device=cpu dp_size=1 cpu=1" in output
    assert "  startup  serving_backend   starting backend=vllm replicas=2" in output
    assert "  ready    serving_backend   backend=vllm device=cpu replicas=2" in output
    assert re.search(r"  ready\s+admin\s+url=http://127\.0\.0\.1:\d+", output)
    assert "FlashRL training run" not in output

    trainer.close()


def test_flashrl_constructor_keeps_console_silent_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Constructor startup should stay silent when console logging is disabled."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(),
        reward_callback=reward_fn,
        trainer_config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=False,
        ),
        metrics_config=MetricsConfig(enabled=False),
    )

    assert capsys.readouterr().out == ""
    trainer.close()


def test_run_logger_compact_console_groups_step_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compact mode should render grouped step blocks instead of repeated prefixes."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=True,
            file=True,
            console_mode="compact",
        ),
        model_name="org/model-name",
    )

    logger.start_run(
        dataset_size=4,
        batch_size=4,
        max_epochs=1,
        total_batches=2,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_configured=False,
        group_size=2,
        clip_ratio=0.2,
        prompts_per_step=2,
        steps_per_epoch=2,
        total_planned_steps=2,
        actor_backend="huggingface",
        actor_device="cpu",
        actor_dp_size=1,
        serving_backend="vllm",
        serving_device="auto",
        serving_num_replicas=2,
        admin_base_url="http://127.0.0.1:44123/admin",
        max_new_tokens=256,
    )
    logger.log_model_load(
        "actor_backend",
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
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "dataset_prompt_start": 1,
            "dataset_prompt_end": 2,
            "dataset_prompt_count": 4,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 4,
            "samples_this_step": 4,
            "stage": "rollout",
            "latency_seconds": 0.25,
            "sample_count": 4,
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
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "dataset_prompt_start": 1,
            "dataset_prompt_end": 2,
            "dataset_prompt_count": 4,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 4,
            "samples_this_step": 4,
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
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "dataset_prompt_start": 1,
            "dataset_prompt_end": 2,
            "dataset_prompt_count": 4,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 4,
            "samples_this_step": 4,
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
            "reference_configured": False,
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
    logger.log_step_stage(
        {
            "step": 2,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 2,
            "batches_in_epoch": 2,
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "dataset_prompt_start": 3,
            "dataset_prompt_end": 4,
            "dataset_prompt_count": 4,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 4,
            "samples_this_step": 4,
            "stage": "rollout",
            "latency_seconds": 0.3,
            "sample_count": 4,
            "prompt_tokens_mean": 13.0,
            "prompt_tokens_max": 17,
            "response_tokens_mean": 25.0,
            "response_tokens_max": 31,
        }
    )
    logger.log_step_done(
        {
            "step": 2,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 2,
            "batches_in_epoch": 2,
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "dataset_prompt_start": 3,
            "dataset_prompt_end": 4,
            "dataset_prompt_count": 4,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 4,
            "samples_this_step": 4,
            "loss": 0.25,
            "policy_loss": 0.2,
            "kl_divergence": 0.05,
            "reward_mean": 1.0,
            "response_tokens_total": 50,
            "tokens_per_second": 80.0,
            "step_duration_seconds": 0.6,
            "stage_timings": {
                "rollout": 0.3,
                "reference_forward": 0.0,
            },
            "stage_order": ["rollout"],
            "reference_configured": False,
            "reference_active": False,
            "dominant_stage": "rollout",
        }
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

    captured = capsys.readouterr().out
    plain_captured = strip_ansi(captured)
    transcript = (logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "\x1b[" in captured
    assert "FlashRL training run\n  run      " in plain_captured
    assert "  serving  backend=vllm  device=auto  replicas=2" in plain_captured
    assert "  admin    http://127.0.0.1:44123/admin" in plain_captured
    assert "mapping  dataset_prompts=4  prompts_per_step=2  completions_per_step=4" in plain_captured
    assert "progress steps_per_epoch=2  total_steps=2" in plain_captured
    assert (
        "step 1/2  epoch 1/1  batch 1/2  prompt_window=1-2/4  "
        "prompts_this_step=2/2  completions_per_prompt=2  completions_this_step=4/4"
        in plain_captured
    )
    assert "FlashRL training run\n  run      " in transcript
    assert "  serving  backend=vllm  device=auto  replicas=2" in transcript
    assert "  admin    http://127.0.0.1:44123/admin" in transcript
    assert "  grpo     completions_per_prompt=2  clip=0.2000  max_new_tokens=256" in transcript
    assert "  mapping  dataset_prompts=4  prompts_per_step=2  completions_per_step=4" in transcript
    assert "  progress steps_per_epoch=2  total_steps=2" in transcript
    assert "load actor_backend" not in transcript
    assert (
        "step 1/2  epoch 1/1  batch 1/2  prompt_window=1-2/4  "
        "prompts_this_step=2/2  completions_per_prompt=2  completions_this_step=4/4"
        in transcript
    )
    assert "  rollout" in transcript
    assert "prompt_tok 12.5/16" in transcript
    assert "response_tok 24.0/30" in transcript
    assert "  reward" in transcript
    assert "mean 1.5000" in transcript
    assert "  done" in transcript
    assert "dominant rollout" in transcript
    assert (
        "\n  ------------------------------------------------------------------------\n"
        "step 2/2  epoch 1/1  batch 2/2  prompt_window=3-4/4  "
        "prompts_this_step=2/2  completions_per_prompt=2  completions_this_step=4/4"
        in transcript
    )
    assert (
        "\n  ------------------------------------------------------------------------\n"
        "step 2/2  epoch 1/1  batch 2/2  prompt_window=3-4/4  "
        "prompts_this_step=2/2  completions_per_prompt=2  completions_this_step=4/4"
        in plain_captured
    )
    assert "\n  done" in transcript
    assert "\n\nepoch 1/1 summary" not in transcript
    assert "epoch 1/1 summary" in transcript
    assert "  lifecycle startup 1.250s | train_loop 750.0ms" in transcript
    assert "  stages    rollout " in transcript
    assert "step=1 epoch=1/1" not in transcript
    assert "sample step" not in transcript
    assert "viewer" not in transcript
    assert "\x1b[" not in transcript

    events = read_events(logger.run_dir)
    assert any(event["event"] == "sample_preview" for event in events)
    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["payload"]["run_index"] == logger.run_index
    assert run_started["payload"]["dataset_prompt_count"] == 4
    assert run_started["payload"]["planned_prompts_per_step"] == 2
    assert run_started["payload"]["planned_samples_per_step"] == 4
    assert run_started["payload"]["steps_per_epoch"] == 2
    assert run_started["payload"]["total_planned_steps"] == 2

    logger.close()


def test_run_logger_verbose_console_separates_step_blocks(tmp_path: Path) -> None:
    """Verbose mode should also insert one blank line between completed steps."""
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
            console_mode="verbose",
        ),
        model_name="org/model-name",
    )

    logger.log_epoch_start(epoch=1, total_epochs=1, num_batches=2)
    logger.log_step_stage(
        {
            "step": 1,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 2,
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "stage": "rollout",
            "latency_seconds": 0.25,
            "sample_count": 4,
            "prompt_tokens_mean": 12.5,
            "prompt_tokens_max": 16,
            "response_tokens_mean": 24.0,
            "response_tokens_max": 30,
        }
    )
    logger.log_step_done(
        {
            "step": 1,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 2,
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 2,
            "loss": 0.125,
            "policy_loss": 0.1,
            "kl_divergence": 0.025,
            "reward_mean": 1.5,
            "response_tokens_total": 48,
            "tokens_per_second": 96.0,
            "step_duration_seconds": 0.5,
            "stage_timings": {"rollout": 0.25},
            "stage_order": ["rollout"],
            "reference_configured": False,
            "reference_active": False,
            "dominant_stage": "rollout",
        }
    )
    logger.log_step_stage(
        {
            "step": 2,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 2,
            "batches_in_epoch": 2,
            "batch_size": 2,
            "prompt_count": 1,
            "group_size": 2,
            "dataset_prompt_start": 3,
            "dataset_prompt_end": 3,
            "dataset_prompt_count": 3,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 4,
            "samples_this_step": 2,
            "stage": "rollout",
            "latency_seconds": 0.3,
            "sample_count": 2,
            "prompt_tokens_mean": 13.0,
            "prompt_tokens_max": 17,
            "response_tokens_mean": 25.0,
            "response_tokens_max": 31,
        }
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
            "stage_totals": {"rollout": 0.25},
        }
    )

    transcript = logger.console_path.read_text(encoding="utf-8")

    assert (
        "\n  ------------------------------------------------------------------------\n"
        "step 2/?  epoch 1/1  batch 2/2  prompt_window=3-3/3  "
        "prompts_this_step=1/2  completions_per_prompt=2  completions_this_step=2/4\n"
        "step=2 epoch=1/1 batch=2/2 prompt_window=3-3/3 completions_this_step=2 "
        "prompts_this_step=1 planned_prompts_per_step=2 completions_per_prompt=2 "
        "planned_completions_per_step=4 stage=rollout"
    ) in transcript
    assert "\n\nepoch 1 summary" not in transcript


def test_run_logger_serving_debug_uses_compact_terminal_status_and_keeps_file_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Serving debug should stay step-first in both terminal and console transcripts."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=True,
            file=True,
            console_mode="compact",
        ),
        model_name="org/model-name",
    )

    first_prompt_payload = {
        "step": 1,
        "epoch": 1,
        "total_epochs": 1,
        "batch_index": 1,
        "batches_in_epoch": 1,
        "prompt_count": 1,
        "group_size": 4,
        "prompt_index": 0,
        "candidate_index": 0,
        "ttft_seconds": 0.2,
        "tpot_seconds": 0.05,
        "generation_seconds": 0.35,
        "response_token_count": 6,
        "response_preview": "partial text",
        "prompt_text": "Please solve this step by step.\nQuestion: What is 15 + 27?",
        "prompt_preview": "Please solve this step by step.",
    }
    second_candidate_payload = {**first_prompt_payload, "candidate_index": 1, "response_preview": "second candidate"}
    second_prompt_payload = {
        **first_prompt_payload,
        "prompt_count": 2,
        "prompt_index": 1,
        "candidate_index": 0,
        "response_preview": "next prompt candidate",
        "prompt_text": (
            "This is a deliberately long prompt that should wrap across multiple lines and "
            "eventually be truncated because the live rollout terminal view should show the "
            "original prompt once, but keep it readable for debugging and inspection. "
            "This extra sentence ensures the prompt grows beyond the adaptive terminal limit. "
            "And this final sentence pushes it far enough that the logger must append the "
            "truncated marker instead of printing the full text."
        ),
        "prompt_preview": "What is 8 × 7?",
    }

    logger.log_serving_debug_start(first_prompt_payload)
    logger.log_serving_debug_chunk({**first_prompt_payload, "text": "part"})
    logger.log_serving_debug_chunk({**first_prompt_payload, "text": "ial"})
    logger.log_serving_debug_done(first_prompt_payload)
    logger.log_serving_debug_start(second_candidate_payload)
    logger.log_serving_debug_chunk({**second_candidate_payload, "text": "second"})
    logger.log_serving_debug_done(second_candidate_payload)
    logger.log_serving_debug_start(second_prompt_payload)
    logger.log_serving_debug_chunk({**second_prompt_payload, "text": "next"})
    logger.log_serving_debug_done(second_prompt_payload)
    logger.log_step_done(
        {
            "step": 1,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 1,
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 4,
            "loss": 0.125,
            "policy_loss": 0.1,
            "kl_divergence": 0.025,
            "reward_mean": 1.5,
            "response_tokens_total": 48,
            "tokens_per_second": 96.0,
            "step_duration_seconds": 0.5,
            "stage_timings": {"rollout": 0.25},
            "stage_order": ["rollout"],
            "reference_configured": False,
            "reference_active": False,
            "dominant_stage": "rollout",
        }
    )
    logger.log_step_stage(
        {
            "step": 2,
            "epoch": 1,
            "total_epochs": 1,
            "batch_index": 1,
            "batches_in_epoch": 1,
            "batch_size": 4,
            "prompt_count": 2,
            "group_size": 4,
            "dataset_prompt_start": 1,
            "dataset_prompt_end": 2,
            "dataset_prompt_count": 2,
            "planned_prompts_per_step": 2,
            "planned_samples_per_step": 8,
            "samples_this_step": 4,
            "stage": "rollout",
            "latency_seconds": 0.25,
            "sample_count": 4,
            "prompt_tokens_mean": 12.5,
            "prompt_tokens_max": 16,
            "response_tokens_mean": 24.0,
            "response_tokens_max": 30,
        }
    )

    captured = capsys.readouterr().out
    plain_captured = strip_ansi(captured)
    transcript = logger.console_path.read_text(encoding="utf-8")
    events = read_events(logger.run_dir)

    assert "\x1b[" in captured
    assert (
        "step 1/?  epoch 1/1  batch 1/1  prompt_window=?-?/?  "
        "prompts_this_step=1/?  completions_per_prompt=4  completions_this_step=?/?"
        in plain_captured
    )
    assert plain_captured.index("step 1/?  epoch 1/1") < plain_captured.index("  prompt 1/1")
    assert "  prompt 1/1  Please solve this step by step. Question: What is 15 + 27?" in plain_captured
    assert "    rollout 1/4 running..." in plain_captured
    assert "    rollout 1/4 done  ttft=200.0ms  tpot=50.0ms  tokens=6  total=350.0ms" in plain_captured
    assert "    rollout 2/4 running..." in plain_captured
    assert "    rollout 2/4 done  ttft=200.0ms  tpot=50.0ms  tokens=6  total=350.0ms" in plain_captured
    assert "  prompt 2/2  This is a deliberately long prompt that should wrap across multiple lines" in plain_captured
    assert "\r" in captured
    assert "partial" not in plain_captured
    assert "serve step=1 epoch=1/1 batch=1/1 prompt=1/1 completions_per_prompt=4" not in plain_captured
    assert (
        "step 1/?  epoch 1/1  batch 1/1  prompt_window=?-?/?  "
        "prompts_this_step=1/?  completions_per_prompt=4  completions_this_step=?/?"
        in transcript
    )
    assert transcript.index("step 1/?  epoch 1/1") < transcript.index("  prompt 1/1")
    assert "\n    rollout 2/4 running...\n    rollout 2/4 done  ttft=200.0ms  tpot=50.0ms  tokens=6  total=350.0ms" in transcript
    assert "\n\n  prompt 2/2  This is a deliberately long prompt that should wrap across multiple lines" in transcript
    assert "[truncated]" not in transcript
    assert "serve step=" not in transcript
    assert "  candidate " not in transcript
    assert "serve_done" not in transcript
    assert (
        "\n  ------------------------------------------------------------------------\n"
        "step 2/?  epoch 1/1  batch 1/1  prompt_window=1-2/2  "
        "prompts_this_step=2/2  completions_per_prompt=4  completions_this_step=4/8"
        in transcript
    )
    assert "\n  ------------------------------------------------------------------------\n  ------------------------------------------------------------------------\nstep 2/?" not in transcript
    assert "part" not in transcript
    assert "ial" not in transcript
    assert "\x1b[" not in transcript
    serving_events = [event for event in events if event["event"] == "serving_debug"]
    assert len(serving_events) == 3
    assert serving_events[0]["payload"]["candidate_index"] == 0
    assert serving_events[0]["payload"]["response_preview"] == "partial text"


def test_static_viewer_exists_and_contains_run_history_sections() -> None:
    """The unified viewer should preserve the run history workspace."""
    viewer_path = Path("docs/viewer.html")

    assert viewer_path.exists()
    html = viewer_path.read_text(encoding="utf-8")
    assert "Open run folder" in html
    assert ".flashrl-runs" not in html
    assert "logs/" in html
    assert "schema_version" in html
    assert "showDirectoryPicker" in html
    assert "Run List" in html
    assert "Overview" in html
    assert "Timeline" in html
    assert "Console" in html
    assert "Rollouts" in html
    assert "Live Runtime" in html
    assert "Run History" in html
    assert "shared_messages" in html
    assert "completion_messages" in html
    assert "step-filter" in html
    assert "epoch-filter" in html
    assert "events.jsonl" in html
    assert "console.log" in html
    assert "rollouts.jsonl" in html


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

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(response_suffix="reuse", repeat=2),
        reward_callback=reward_fn,
        trainer_config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
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
        "actor_backend": id(trainer._actor_backend),
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
        "actor_backend": id(trainer._actor_backend),
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
        and event["payload"].get("component") == "actor_backend"
        and event["payload"].get("status") == "completed"
        for event in events
    )

    trainer._run_logger.close()


def test_flashrl_train_orchestrates_private_run_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FlashRL.train() should read as one ordered run lifecycle."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(response_suffix="order", repeat=2),
        reward_callback=reward_fn,
        trainer_config=TrainerConfig(batch_size=2, max_epochs=1, shuffle_each_epoch=False),
        logging_config=LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
        ),
        metrics_config=MetricsConfig(enabled=False),
    )
    dataset = [Prompt(text=f"prompt {index}") for index in range(2)]

    call_order: list[str] = []
    for method_name in (
        "_resolve_train_dataset",
        "_build_train_run_state",
        "_reset_run_lifecycle_totals",
        "_open_run_logger",
        "_prepare_trainer_for_run",
        "_start_run_metrics",
        "_execute_training_loop",
        "_handle_train_failure",
        "_finalize_train_run",
    ):
        original = getattr(trainer, method_name)

        def wrapper(*args, __name=method_name, __original=original, **kwargs):
            call_order.append(__name)
            return __original(*args, **kwargs)

        monkeypatch.setattr(trainer, method_name, wrapper)

    trainer.train(dataset)

    assert call_order == [
        "_resolve_train_dataset",
        "_build_train_run_state",
        "_reset_run_lifecycle_totals",
        "_open_run_logger",
        "_prepare_trainer_for_run",
        "_start_run_metrics",
        "_execute_training_loop",
        "_finalize_train_run",
    ]
    assert trainer._run_logger.run_dir.exists()
    trainer._run_logger.close()


def test_grpo_trainer_without_reference_logs_append_only_hot_path(tmp_path: Path) -> None:
    """Default local trainer path should run without a reference model."""
    training_backend = FakeTrainingBackend(config=None, learning_rate=1e-2)
    serving_backend = FakeServingBackend(config=None)
    rollout = UserDefinedRollout(
        rollout_fn=make_rollout_fn(response_suffix="sample", repeat=20),
        serving_backend=serving_backend,
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
        config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        grpo_config=GrpoConfig(group_size=2, kl_coefficient=0.0),
        actor_backend=training_backend,
        serving_backend=serving_backend,
        reference_backend=None,
        reward_fn=reward,
        rollout_generator=rollout,
        run_logger=logger,
    )
    dataset = [Prompt(text=f"prompt {index}") for index in range(6)]

    logger.start_run(
        dataset_size=len(dataset),
        batch_size=4,
        max_epochs=1,
        total_batches=3,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_configured=False,
        group_size=2,
        clip_ratio=0.2,
        prompts_per_step=2,
        steps_per_epoch=3,
        total_planned_steps=3,
        actor_backend="huggingface",
        actor_device="cpu",
        actor_dp_size=1,
        serving_backend="huggingface",
        serving_device="cpu",
        admin_base_url="http://127.0.0.1:43123/admin",
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
        "prepare_inputs",
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
    sync_event = next(
        event
        for event in events
        if event["event"] == "step_stage"
        and event["payload"]["step"] == 1
        and event["payload"]["stage"] == "sync"
    )
    assert sync_event["payload"]["weight_version_id"] == 1

    step_done = next(event for event in events if event["event"] == "step_done")
    assert step_done["payload"]["reference_configured"] is False
    assert step_done["payload"]["rollout_weight_version"]["version_id"] == 0
    assert step_done["payload"]["rollout_weight_version"]["origin"] == "startup"
    assert step_done["payload"]["stage_timings"]["reference_forward"] == 0.0
    assert step_done["payload"]["stage_order"] == [
        "rollout",
        "reward",
        "advantage",
        "prepare_inputs",
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
    assert "  runtime  single-device-per-backend  reference=disabled" in transcript
    assert "  actor    backend=huggingface  device=cpu  dp_size=1" in transcript
    assert "  serving  backend=huggingface  device=cpu" in transcript
    assert "  admin    http://127.0.0.1:43123/admin" in transcript
    assert "cpu=1" in transcript
    assert (
        "step=1 epoch=1/1 batch=1/3 prompt_window=1-2/6 completions_this_step=4 "
        "prompts_this_step=2 planned_prompts_per_step=2 completions_per_prompt=2 "
        "planned_completions_per_step=4 stage=rollout"
    ) in transcript
    assert "stage=reward" in transcript
    assert "stage=loss_assembly" in transcript
    assert "stage=backward" in transcript
    assert "stage=complete" in transcript
    assert "phase=" not in transcript
    assert "━" not in transcript
    assert "sample step" not in transcript
    assert "viewer" not in transcript

    rollouts = read_rollouts(logger.run_dir)
    assert len(rollouts) == len(dataset)
    first_rollout = rollouts[0]
    assert first_rollout["run_id"] == logger.run_id
    assert first_rollout["run_index"] == logger.run_index
    assert first_rollout["step"] == 1
    assert first_rollout["prompt_index"] == 0
    assert first_rollout["schema_version"] == 3
    assert first_rollout["group_size"] == 2
    assert first_rollout["prompt_count"] == 2
    assert first_rollout["candidate_count"] == 2
    assert first_rollout["batch_candidate_count"] == 4
    assert first_rollout["input"]["prompt_preview"] == "prompt 0"
    assert first_rollout["input"]["prompt_token_count"] > 0
    assert first_rollout["input"]["shared_messages"][0]["content"] == "prompt 0"
    assert first_rollout["serving"]["weight_version"]["version_id"] == 0
    assert first_rollout["serving"]["weight_version"]["origin"] == "startup"
    assert first_rollout["summary"]["reward_mean"] > 0.0
    assert "sample_index" not in first_rollout
    assert "sample_count" not in first_rollout
    assert "prompt" not in first_rollout
    assert len(first_rollout["candidates"]) == 2
    assert [candidate["candidate_index"] for candidate in first_rollout["candidates"]] == [0, 1]
    first_candidate = first_rollout["candidates"][0]
    assert first_candidate["output"]["preview"].startswith("sample detail")
    assert first_candidate["output"]["response_token_count"] > 0
    assert first_candidate["output"]["avg_log_prob_per_token"] is not None
    assert first_candidate["output"]["weight_version"] == first_rollout["serving"]["weight_version"]
    assert first_candidate["completion_messages"][0]["role"] == "assistant"
    assert "rollout" not in first_candidate
    assert "conversation" not in first_candidate
    assert first_candidate["reward"]["value"] > 0.0

    logger.close()


def test_reward_metadata_rates_are_logged_at_step_level(tmp_path: Path) -> None:
    """Reward metadata booleans should be aggregated into per-step and per-epoch rates."""
    training_backend = FakeTrainingBackend(config=None, learning_rate=1e-2)
    serving_backend = FakeServingBackend(config=None)
    rollout = UserDefinedRollout(
        rollout_fn=make_rollout_fn(response_suffix="rates", repeat=1),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )

    def reward_with_metadata(rollout: RolloutOutput) -> RewardOutput:
        is_first_candidate = rollout.text.endswith("::0")
        return RewardOutput(
            reward=1.0 if is_first_candidate else 0.0,
            metadata={
                "accuracy_pass": is_first_candidate,
                "format_pass": True,
                "truncated": not is_first_candidate,
            },
        )

    logger = RunLogger(
        LoggingConfig(
            log_dir=tmp_path,
            console=False,
            file=True,
            console_mode="compact",
        ),
        model_name="fake/model",
    )
    trainer = GRPOTrainer(
        config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        grpo_config=GrpoConfig(group_size=2, kl_coefficient=0.0),
        actor_backend=training_backend,
        serving_backend=serving_backend,
        reference_backend=None,
        reward_fn=UserDefinedReward(reward_fn=reward_with_metadata, config=SimpleNamespace()),
        rollout_generator=rollout,
        run_logger=logger,
    )
    dataset = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]

    logger.start_run(
        dataset_size=len(dataset),
        batch_size=4,
        max_epochs=1,
        total_batches=1,
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        runtime_shape="single-device-per-backend",
        reference_configured=False,
        group_size=2,
        clip_ratio=0.2,
        prompts_per_step=2,
        steps_per_epoch=1,
        total_planned_steps=1,
        actor_backend="huggingface",
        actor_device="cpu",
        actor_dp_size=1,
        serving_backend="huggingface",
        serving_device="cpu",
    )
    trainer.train(dataset)
    logger.finish_run(status="completed", total_steps=trainer.total_steps)

    events = read_events(logger.run_dir)
    reward_event = next(
        event
        for event in events
        if event["event"] == "step_stage" and event["payload"]["stage"] == "reward"
    )
    assert reward_event["payload"]["accuracy_pass_rate"] == pytest.approx(0.5)
    assert reward_event["payload"]["format_pass_rate"] == pytest.approx(1.0)
    assert reward_event["payload"]["truncation_rate"] == pytest.approx(0.5)

    step_done = next(event for event in events if event["event"] == "step_done")
    assert step_done["payload"]["accuracy_pass_rate"] == pytest.approx(0.5)
    assert step_done["payload"]["format_pass_rate"] == pytest.approx(1.0)
    assert step_done["payload"]["truncation_rate"] == pytest.approx(0.5)

    epoch_summary = next(event for event in events if event["event"] == "epoch_summary")
    assert epoch_summary["payload"]["accuracy_pass_rate"] == pytest.approx(0.5)
    assert epoch_summary["payload"]["format_pass_rate"] == pytest.approx(1.0)
    assert epoch_summary["payload"]["truncation_rate"] == pytest.approx(0.5)

    logger.close()


def test_grpo_assemble_loss_uses_response_only_grpo_terms() -> None:
    """Loss assembly should use response-only GRPO terms and stored rollout log-probs."""
    from flashrl.framework.trainer.grpo.loss_variants import assemble_grpo_loss

    # Create simple test data
    device = torch.device("cpu")
    batch_size = 2
    seq_length = 8
    prompt_length = 4
    response_length = seq_length - prompt_length

    # Create token IDs: [0, 1, 2, 3] for prompt, [4, 5, 6, 7] for response
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]], device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_lengths = torch.tensor([prompt_length, prompt_length], device=device)

    # Create mock logits
    vocab_size = 100
    actor_logits = torch.randn(batch_size, seq_length, vocab_size, device=device)
    ref_logits = torch.randn(batch_size, seq_length, vocab_size, device=device)

    # Create rollout response log probs (4 tokens per response)
    rollout_response_log_probs = [
        [-0.1, -0.2, -0.3, -0.4],
        [-0.5, -0.6, -0.7, -0.8],
    ]

    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)

    # Create GrpoConfig
    from flashrl.framework.config import GrpoConfig
    config = GrpoConfig(
        group_size=2,
        kl_coefficient=0.3,
        clip_ratio=0.2,
    )

    # Test without reference model
    result_no_ref = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        config=config,
    )

    assert result_no_ref.loss.item() == pytest.approx(result_no_ref.policy_loss.item())
    assert result_no_ref.kl_divergence.item() == pytest.approx(0.0)
    assert result_no_ref.response_tokens_total == 8

    # Test with reference model
    result_with_ref = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        config=config,
    )

    assert result_with_ref.kl_divergence.item() >= 0.0
    expected_total = result_with_ref.policy_loss.item() + 0.3 * result_with_ref.kl_divergence.item()
    assert result_with_ref.loss.item() == pytest.approx(expected_total)


def test_user_defined_rollout_generate_grouped_is_prompt_major_and_validates_count() -> None:
    """Grouped rollout should flatten prompt-major candidates and validate grouped shape."""
    serving_backend = FakeServingBackend(config=None)
    rollout = UserDefinedRollout(
        rollout_fn=make_rollout_fn(response_suffix="group", repeat=1),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )
    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]

    expanded_prompts, rollouts, prompt_indices, candidate_indices = rollout.generate_grouped(
        prompts,
        group_size=2,
    )

    assert len(expanded_prompts) == 4
    assert len(rollouts) == 4
    assert [prompt.text for prompt in expanded_prompts] == [
        "prompt 0",
        "prompt 0",
        "prompt 1",
        "prompt 1",
    ]
    assert prompt_indices == [0, 0, 1, 1]
    assert candidate_indices == [0, 1, 0, 1]

    invalid_rollout = UserDefinedRollout(
        rollout_fn=lambda prompts, serving_backend: make_rollout_fn(
            response_suffix="invalid",
            repeat=1,
        )(prompts[:-1], serving_backend),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="one output per input prompt"):
        invalid_rollout.generate_grouped(prompts, group_size=2)


def test_grpo_advantages_are_normalized_within_each_prompt_group() -> None:
    """GRPO advantages should be computed per prompt group, not across the whole flat batch."""
    trainer_serving_backend = FakeServingBackend(config=None)
    trainer = GRPOTrainer(
        config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        grpo_config=GrpoConfig(group_size=2),
        actor_backend=FakeTrainingBackend(config=None, learning_rate=1e-2),
        serving_backend=trainer_serving_backend,
        reference_backend=None,
        reward_fn=UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace()),
        rollout_generator=UserDefinedRollout(
            rollout_fn=make_rollout_fn(response_suffix="adv", repeat=1),
            serving_backend=trainer_serving_backend,
            config=SimpleNamespace(),
        ),
        run_logger=None,
    )

    # Use compute_advantages helper function directly
    from flashrl.framework.trainer.grpo.grpo_helpers import compute_advantages

    advantages = compute_advantages(
        rewards=[
            RewardOutput(reward=1.0),
            RewardOutput(reward=3.0),
            RewardOutput(reward=10.0),
            RewardOutput(reward=12.0),
        ],
        prompt_count=2,
        group_size=2,
    )

    assert advantages.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])

def test_flashrl_default_path_skips_reference_and_uses_append_only_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FlashRL should skip the reference model by default."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(response_suffix="console", repeat=8),
        reward_callback=reward_fn,
        trainer_config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
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
    assert "FlashRL startup\n  startup  model=fake/model  actor=huggingface  dp_size=1" in output
    assert "serving=huggingface  reference=disabled" in output
    assert "  startup  actor_backend     starting backend=huggingface dp_size=1" in output
    assert "  ready    actor_backend     backend=huggingface device=cpu dp_size=1 cpu=1" in output
    assert "  startup  serving_backend   starting backend=huggingface" in output
    assert re.search(r"  ready\s+admin\s+url=http://127\.0\.0\.1:\d+", output)
    assert "  ========================================================================" in output
    assert "FlashRL training run" in output
    assert "  run      " in output
    assert "cpu=1" in output
    assert "reference=disabled" in output
    assert "  actor    backend=huggingface  device=cpu  dp_size=1" in output
    assert "  serving  backend=huggingface  device=cpu" in output
    assert re.search(r"  admin    http://127\.0\.0\.1:\d+", output)
    assert "completions_per_prompt=2  clip=0.2000  max_new_tokens=512" in output
    assert "load actor_backend" not in output
    assert "load serving_backend" not in output
    assert "loaded reference_backend" not in output
    assert "mapping  dataset_prompts=4  prompts_per_step=2  completions_per_step=4" in output
    assert "progress steps_per_epoch=2  total_steps=2" in output
    assert (
        "step 1/2  epoch 1/1  batch 1/2  prompt_window=1-2/4  "
        "prompts_this_step=2/2  completions_per_prompt=2  completions_this_step=4/4"
        in output
    )
    assert "  rollout" in output
    assert "  reward" in output
    assert "  prepare_inputs" in output
    assert "  backward" in output
    assert "  done" in output
    assert "epoch 1/1 summary" in output
    assert "  lifecycle " in output
    assert "  stages    " in output
    assert "step=1 epoch=1/1" not in output
    assert "stage=rollout" not in output
    assert "sample step" not in output
    assert "phase=" not in output
    assert "stage initializing" not in output
    assert "━" not in output
    assert "viewer" not in output

    events = read_events(trainer._run_logger.run_dir)
    assert all(
        event["payload"].get("component") != "reference_backend"
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
    assert rollouts[0]["schema_version"] == 3
    assert rollouts[0]["input"]["prompt_preview"] == "prompt 0"
    assert rollouts[0]["serving"]["weight_version"]["version_id"] == 0
    assert rollouts[0]["candidates"][0]["reward"]["value"] > 0.0
    assert (
        rollouts[0]["candidates"][0]["output"]["weight_version"]
        == rollouts[0]["serving"]["weight_version"]
    )
    assert rollouts[0]["input"]["shared_messages"][0]["content"] == "prompt 0"
    assert rollouts[0]["prompt_index"] == 0
    assert rollouts[0]["candidates"][0]["candidate_index"] == 0
    assert "sample_index" not in rollouts[0]

    trainer._run_logger.close()


def test_flashrl_reference_backend_loads_and_logs_kl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configuring the reference backend should load it and expose KL-related metrics."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(response_suffix="kl", repeat=8),
        reward_callback=reward_fn,
        reference_config=TrainingConfig(model_name="fake/model", backend="huggingface", device="cpu"),
        trainer_config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        grpo_config=GrpoConfig(group_size=2, kl_coefficient=0.1),
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
        and event["payload"].get("component") == "reference_backend"
        and event["payload"].get("status") == "completed"
        for event in events
    )
    step_event = next(event for event in events if event["event"] == "step_done")
    assert step_event["payload"]["reference_configured"] is True
    assert step_event["payload"]["reference_active"] is True
    assert step_event["payload"]["stage_timings"]["reference_forward"] > 0
    assert "reference_forward" in step_event["payload"]["stage_order"]
    assert any(
        event["event"] == "step_stage"
        and event["payload"].get("stage") == "reference_forward"
        for event in events
    )

    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "reference=configured" in transcript
    assert "  reference_forward" in transcript
    assert "full_tok_total" in transcript

    trainer._run_logger.close()


def test_flashrl_checkpoint_works_before_train_and_resume_skips_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Checkpoint load/save should work before train and preserve resume state."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(response_suffix="resume", repeat=3),
        reward_callback=reward_fn,
        trainer_config=TrainerConfig(batch_size=2, max_epochs=3, shuffle_each_epoch=False),
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

    assert trainer._trainer.total_steps == 15
    transcript = (trainer._run_logger.run_dir / "console.log").read_text(encoding="utf-8")
    assert "epoch 2/3" in transcript
    assert "epoch 1/3" not in transcript

    trainer._run_logger.close()


def test_run_config_latest_checkpoint_requires_directory() -> None:
    """Managed latest resume should require an explicit checkpoint directory."""
    config = build_managed_run_config(log_dir=Path("/tmp/flashrl-logs"), resume_from="latest")

    with pytest.raises(
        ValueError,
        match="checkpointing.directory is required when checkpointing.resume_from='latest'",
    ):
        RunConfig.from_dict(config)

    config["checkpointing"] = {
        "directory": "/tmp/flashrl-checkpoints",
        "resume_from": "latest",
    }
    resolved = RunConfig.from_dict(config)
    assert resolved.checkpointing.resume_from == "latest"


def test_managed_checkpointing_saves_intervals_and_resumes_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed save-every-N checkpoints should resume into the same run directory."""
    patch_backends(monkeypatch)
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]

    trainer = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="managed", repeat=2),
        reward_fn=reward_fn,
        run_config=build_managed_run_config(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_every_steps=2,
        ),
    )
    trainer.train(dataset)

    first_run_dir = trainer._run_logger.run_dir
    assert (checkpoint_dir / "step-00000002.pt").exists()
    assert (checkpoint_dir / "step-00000004.pt").exists()
    latest_manifest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_manifest["checkpoint_path"] == str(checkpoint_dir / "step-00000004.pt")
    assert latest_manifest["run_dir"] == str(first_run_dir)

    trainer_resume = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="managed", repeat=2),
        reward_fn=reward_fn,
        run_config=build_managed_run_config(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_every_steps=2,
            resume_from="latest",
        ),
    )
    trainer_resume.train(dataset)

    assert trainer_resume._run_logger.run_dir == first_run_dir
    assert trainer_resume._trainer.total_steps == 8
    assert (checkpoint_dir / "step-00000006.pt").exists()
    assert (checkpoint_dir / "step-00000008.pt").exists()

    run_dirs = [path for path in log_dir.iterdir() if path.is_dir()]
    assert run_dirs == [first_run_dir]

    events = read_events(first_run_dir)
    assert any(event["event"] == "run_resumed" for event in events)
    assert any(
        event["event"] == "checkpoint"
        and event["payload"]["action"] == "load"
        and event["payload"]["trigger"] == "resume"
        for event in events
    )
    assert any(
        event["event"] == "checkpoint"
        and event["payload"]["action"] == "save"
        and event["payload"]["trigger"] == "interval"
        and event["payload"]["step"] == 8
        for event in events
    )

    latest_manifest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_manifest["checkpoint_path"] == str(checkpoint_dir / "step-00000008.pt")
    rollouts = read_rollouts(first_run_dir)
    version_ids = [item["serving"]["weight_version"]["version_id"] for item in rollouts]
    assert version_ids[:4] == [0, 1, 2, 3]
    assert version_ids[4:] == [5, 6, 7, 8]
    assert rollouts[4]["serving"]["weight_version"]["origin"] == "resume"

    trainer._run_logger.close()
    trainer_resume._run_logger.close()


def test_managed_checkpointing_writes_final_checkpoint_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed final checkpointing should emit final.pt and update latest.json."""
    patch_backends(monkeypatch)
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    dataset = [Prompt(text=f"prompt {index}") for index in range(2)]

    trainer = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="final", repeat=1),
        reward_fn=reward_fn,
        run_config=build_managed_run_config(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_on_run_end=True,
        ),
    )
    trainer.train(dataset)

    final_checkpoint = checkpoint_dir / "final.pt"
    assert final_checkpoint.exists()
    latest_manifest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_manifest["checkpoint_path"] == str(final_checkpoint)

    events = read_events(trainer._run_logger.run_dir)
    assert any(
        event["event"] == "checkpoint"
        and event["payload"]["action"] == "save"
        and event["payload"]["trigger"] == "final"
        for event in events
    )

    trainer._run_logger.close()


def test_managed_checkpointing_defaults_to_run_local_checkpoint_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without explicit overrides, interval and final checkpoints should stay under the run dir."""
    patch_backends(monkeypatch)
    log_dir = tmp_path / "logs"
    dataset = [Prompt(text=f"prompt {index}") for index in range(2)]

    trainer = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="default-dir", repeat=1),
        reward_fn=reward_fn,
        run_config=build_managed_run_config(
            log_dir=log_dir,
            save_every_steps=1,
            save_on_run_end=True,
        ),
    )
    trainer.train(dataset)

    run_dir = trainer._run_logger.run_dir
    checkpoint_dir = run_dir / "checkpoints"
    assert (checkpoint_dir / "step-00000001.pt").exists()
    assert (checkpoint_dir / "step-00000002.pt").exists()
    final_checkpoint = checkpoint_dir / "final.pt"
    assert final_checkpoint.exists()
    latest_manifest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_manifest["checkpoint_path"] == str(final_checkpoint)
    assert latest_manifest["run_dir"] == str(run_dir)

    trainer._run_logger.close()


def test_managed_checkpointing_explicit_final_path_overrides_default_final_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`final_path` should only move the final checkpoint, not the managed manifest root."""
    patch_backends(monkeypatch)
    log_dir = tmp_path / "logs"
    explicit_final_path = tmp_path / "exports" / "final.pt"
    dataset = [Prompt(text=f"prompt {index}") for index in range(2)]
    run_config = build_managed_run_config(
        log_dir=log_dir,
        save_on_run_end=True,
    )
    checkpointing = dict(run_config.get("checkpointing", {}))
    checkpointing["final_path"] = str(explicit_final_path)
    run_config["checkpointing"] = checkpointing

    trainer = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="explicit-final", repeat=1),
        reward_fn=reward_fn,
        run_config=run_config,
    )
    trainer.train(dataset)

    run_dir = trainer._run_logger.run_dir
    assert explicit_final_path.exists()
    assert not (run_dir / "checkpoints" / "final.pt").exists()
    latest_manifest = json.loads(
        (run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest_manifest["checkpoint_path"] == str(explicit_final_path)
    assert latest_manifest["run_dir"] == str(run_dir)

    trainer._run_logger.close()


def test_managed_resume_fails_fast_when_original_run_dir_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed append-resume should reject checkpoints without a usable prior run dir."""
    patch_backends(monkeypatch)
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    dataset = [Prompt(text=f"prompt {index}") for index in range(2)]

    trainer = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="broken-resume", repeat=1),
        reward_fn=reward_fn,
        run_config=build_managed_run_config(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_every_steps=1,
        ),
    )
    trainer.train(dataset)

    checkpoint_path = checkpoint_dir / "step-00000002.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["checkpoint_metadata"]["run"]["run_dir"] = str(tmp_path / "missing-run-dir")
    torch.save(checkpoint, checkpoint_path)

    trainer_resume = FlashRL(
        rollout_fn=make_rollout_fn(response_suffix="broken-resume", repeat=1),
        reward_fn=reward_fn,
        run_config=build_managed_run_config(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from=str(checkpoint_path),
        ),
    )

    with pytest.raises(FileNotFoundError, match="original run directory"):
        trainer_resume.train(dataset)

    trainer._run_logger.close()


def test_flashrl_no_step_summary_is_explicit_on_early_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the run fails before any training step finishes, the summary should say so."""
    patch_backends(monkeypatch)

    trainer = build_flashrl(
        tmp_path,
        rollout_callback=make_rollout_fn(response_suffix="broken", repeat=4),
        reward_callback=failing_reward_fn,
        trainer_config=TrainerConfig(batch_size=2, max_epochs=1, shuffle_each_epoch=False),
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
