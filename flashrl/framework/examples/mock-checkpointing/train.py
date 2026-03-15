"""Offline mock example for managed FlashRL checkpointing."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch
import torch.nn.functional as F

from flashrl.framework import FlashRL, TrainingConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.training import ActorTrainingBackend, TrainingBackend


DEFAULT_LOG_DIR = Path("logs/mock-checkpointing")
DEFAULT_CHECKPOINT_DIR = Path("logs/mock-checkpointing-checkpoints")
DEFAULT_SAVE_EVERY_STEPS = 2
DEFAULT_PROMPT_COUNT = 6
DEFAULT_MAX_EPOCHS = 1


class MockTokenizer:
    """A tiny deterministic tokenizer for the offline example."""

    def __init__(self, vocab_size: int = 48) -> None:
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
            input_ids.append(tokens + padding_tokens)
            attention_mask.append([1] * len(tokens) + [0] * len(padding_tokens))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _encode(self, text: str, *, max_length: int) -> list[int]:
        tokens = [((ord(char) % (self.vocab_size - 1)) + 1) for char in text[:max_length]]
        return tokens or [1]


class MockCausalLM(torch.nn.Module):
    """A tiny trainable LM for local checkpointing demos."""

    def __init__(self, vocab_size: int = 48, bias_shift: float = 0.0) -> None:
        super().__init__()
        base = torch.linspace(-0.25, 0.25, vocab_size)
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


class MockActor:
    """Minimal actor wrapper that matches the training backend contract."""

    def __init__(self, bias_shift: float = 0.0) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_length=128)
        self.tokenizer = MockTokenizer()
        self.model = MockCausalLM(self.tokenizer.vocab_size, bias_shift=bias_shift)
        self.generation_defaults: dict[str, object] = {}

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


class MockTrainingBackend(ActorTrainingBackend):
    """Tiny local training backend for the checkpointing demo."""

    def __init__(
        self,
        config,
        learning_rate: float = 1e-5,
    ) -> None:
        resolved_config = config or TrainingConfig(model_name="mock/model", device="cpu")
        super().__init__(
            resolved_config,
            learning_rate=learning_rate,
        )
        self.model_copy = MockActor(bias_shift=0.2)
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


class MockReferenceBackend(TrainingBackend):
    """Tiny frozen reference backend for the checkpointing demo."""

    def __init__(self, config) -> None:
        resolved_config = config or TrainingConfig(model_name="mock/model", device="cpu")
        super().__init__(resolved_config, role="reference")
        self.model_copy = MockActor(bias_shift=0.0)
        self.model_copy.eval()
        self.model_copy.model.requires_grad_(False)
        self.device = self.model_copy.device
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


class MockServingBackend(ServingBackend):
    """Tiny serving backend used only so FlashRL can build the runtime."""

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.generation_defaults: dict[str, object] = {}
        self._actor = MockActor(bias_shift=0.05)
        self._actor.eval()

    def generate(self, prompts: list[str], **kwargs):
        return [output.text for output in self.generate_batch(prompts, **kwargs)]

    def generate_batch(self, prompts: list[str], **kwargs):
        del kwargs
        return [
            SimpleNamespace(
                text=f"mock serving response for {prompt}",
                prompt_token_ids=self._actor.tokenizer._encode(prompt, max_length=128),
                response_token_ids=[1, 2, 3],
                response_token_logprobs=[-0.1, -0.1, -0.1],
                log_prob=-0.3,
            )
            for prompt in prompts
        ]

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        del kwargs
        return [self.generate_batch([prompt] * group_size) for prompt in prompts]

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)

    def sync_from_training_actor(self, training_actor) -> None:
        self._actor.model.load_state_dict(training_actor.model.state_dict())

    def close(self) -> None:
        return None


def build_mock_dataset(prompt_count: int) -> list[Prompt]:
    """Build a small deterministic dataset for the mock training run."""
    return [
        Prompt(
            text=f"mock prompt {index}",
            metadata={
                "task_id": f"mock-task-{index}",
                "source": "mock-checkpointing",
                "split": "train",
            },
        )
        for index in range(prompt_count)
    ]


def rollout_fn(prompts: list[Prompt], serving_backend: ServingBackend) -> list[RolloutOutput]:
    """Create deterministic rollouts without external model downloads."""
    del serving_backend
    outputs: list[RolloutOutput] = []
    for prompt in prompts:
        prompt_token_ids = [((ord(char) % 47) + 1) for char in prompt.text[:128]] or [1]
        task_id = str(prompt.metadata.get("task_id", "unknown"))
        response = (
            f"<think>Inspect checkpoint flow for {task_id}.</think>"
            f"<answer>resume-safe output for {prompt.text}</answer>"
        )
        response_token_ids = [((ord(char) % 47) + 1) for char in response[:32]] or [1]
        response_token_logprobs = [
            -0.05 - (0.001 * token_index)
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
    return outputs


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Reward longer responses slightly more so the run produces non-zero metrics."""
    return RewardOutput(reward=len(rollout.text) / 100.0)


@contextmanager
def install_mock_backends() -> Iterator[None]:
    """Temporarily route FlashRL through the offline mock backends."""
    import flashrl.framework.flashrl as flashrl_module

    original_training_factory = flashrl_module.create_training_backend
    original_serving_factory = flashrl_module.create_serving_backend
    flashrl_module.create_training_backend = (
        lambda config, role, learning_rate=None: (
            MockTrainingBackend(config, learning_rate=float(learning_rate or 1e-5))
            if role == "actor"
            else MockReferenceBackend(config)
        )
    )
    flashrl_module.create_serving_backend = lambda config, startup_logger=None: MockServingBackend(
        config
    )
    try:
        yield
    finally:
        flashrl_module.create_training_backend = original_training_factory
        flashrl_module.create_serving_backend = original_serving_factory


def build_run_config(args: argparse.Namespace) -> dict[str, object]:
    """Build a profile-style run config that enables managed checkpointing."""
    checkpointing: dict[str, object] = {
        "directory": str(Path(args.checkpoint_dir)),
        "save_every_steps": int(args.save_every_steps),
    }
    if args.resume_from is not None:
        checkpointing["resume_from"] = args.resume_from
    if args.save_on_run_end:
        checkpointing["save_on_run_end"] = True

    return {
        "actor": {
            "model_name": "mock/checkpoint-model",
            "backend": "huggingface",
            "device": "cpu",
        },
        "serving": {
            "model_name": "mock/checkpoint-model",
            "backend": "huggingface",
            "device": "cpu",
        },
        "trainer": {
            "learning_rate": 1.0e-5,
            "batch_size": 2,
            "max_epochs": int(args.max_epochs),
            "shuffle_each_epoch": False,
        },
        "grpo": {
            "group_size": 2,
        },
        "logging": {
            "log_dir": str(Path(args.log_dir)),
            "console": True,
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the mock checkpointing example."""
    parser = argparse.ArgumentParser(description="Offline mock FlashRL checkpointing demo.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--save-every-steps", type=int, default=DEFAULT_SAVE_EVERY_STEPS)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--save-on-run-end", action="store_true")
    parser.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the offline checkpointing example."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    dataset = build_mock_dataset(args.prompt_count)
    flashrl: FlashRL | None = None

    try:
        with install_mock_backends():
            flashrl = FlashRL(
                rollout_fn=rollout_fn,
                reward_fn=reward_fn,
                run_config=build_run_config(args),
            )
            flashrl.train(dataset)
        assert flashrl._run_logger is not None
        checkpoint_dir = Path(args.checkpoint_dir)
        print(f"run_dir={flashrl._run_logger.run_dir}")
        print(f"checkpoint_dir={checkpoint_dir}")
        latest_manifest = checkpoint_dir / "latest.json"
        if latest_manifest.exists():
            print(f"latest_manifest={latest_manifest}")
        return 0
    except Exception as exc:
        print(f"mock checkpointing example failed: {exc}")
        return 1
    finally:
        if flashrl is not None:
            flashrl.close()


if __name__ == "__main__":
    raise SystemExit(main())
