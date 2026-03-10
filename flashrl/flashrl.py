"""FlashRL: Unified RL training API.

This module provides a simple, unified API for RL training that handles
model loading, training loops, and checkpointing.
"""

from typing import Callable

from flashrl.framework.config import (
    TrainerConfig,
    ModelConfig,
    RolloutConfig,
    RewardConfig,
)
from flashrl.framework.backends.training import TrainingBackend
from flashrl.framework.backends.serving import ServingBackend
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.trainer.grpo import GRPOTrainer
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    RewardOutput,
)


class FlashRL:
    """Unified FlashRL trainer - simple API for RL training.

    This class provides a clean, simple interface for RL training. Users provide:
    - Model name (HuggingFace model path)
    - Rollout function (how to generate responses)
    - Reward function (how to score responses)

    FlashRL handles:
    - Loading backends (training, serving, reference)
    - Training loop
    - Checkpointing

    Example:
        def my_rollout_fn(prompts, actor):
            return actor.generate([p.text for p in prompts])

        def my_reward_fn(rollout):
            return RewardOutput(reward=len(rollout.text) / 100.0)

        flashrl = FlashRL(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            rollout_fn=my_rollout_fn,
            reward_fn=my_reward_fn,
            learning_rate=1e-5,
            batch_size=4,
            max_epochs=2,
        )

        dataset = [Prompt(text=p) for p in my_prompts]
        flashrl.train(dataset)
    """

    def __init__(
        self,
        model: str,
        rollout_fn: Callable[[list[Prompt], "ActorModel"], list[RolloutOutput]],
        reward_fn: Callable[[RolloutOutput], RewardOutput],
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        max_epochs: int = 10,
        kl_coefficient: float = 0.1,
        device: str | None = None,
        max_length: int = 2048,
    ) -> None:
        """Initialize FlashRL trainer.

        Args:
            model: HuggingFace model name/path.
            rollout_fn: Function to generate rollouts from prompts.
                Takes (list[Prompt], ActorModel) and returns list[RolloutOutput].
            reward_fn: Function to compute rewards from rollouts.
                Takes RolloutOutput and returns RewardOutput.
            learning_rate: Learning rate for optimizer.
            batch_size: Training batch size.
            max_epochs: Maximum training epochs.
            kl_coefficient: KL divergence penalty coefficient.
            device: Device to use (None = auto-detect).
            max_length: Maximum sequence length.
        """
        # Store user functions
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn

        # Create configs
        self.model_config = ModelConfig(
            model_name=model,
            device=device,
            max_length=max_length,
        )
        self.trainer_config = TrainerConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            kl_coefficient=kl_coefficient,
        )
        self.rollout_config = RolloutConfig()
        self.reward_config = RewardConfig()

        # Load models lazily in train() for faster initialization
        self._training_backend: TrainingBackend | None = None
        self._serving_backend: ServingBackend | None = None
        self._reference: ReferenceModel | None = None
        self._trainer: GRPOTrainer | None = None

    def _load_models(self) -> None:
        """Load training and serving backends, and reference model."""
        print(f"Loading model: {self.model_config.model_name}")

        self._training_backend = TrainingBackend(
            self.model_config,
            learning_rate=self.trainer_config.learning_rate,
        )
        print("✓ Training backend loaded")

        self._serving_backend = ServingBackend(self.model_config)
        print("✓ Serving backend loaded")

        self._reference = ReferenceModel(self.model_config)
        print("✓ Reference model loaded")

    def train(self, dataset: list[Prompt]) -> None:
        """Train on dataset.

        Args:
            dataset: List of prompts to train on.
        """
        # Lazy load models
        if self._training_backend is None:
            self._load_models()

        # Create wrapper adapters
        rollout_generator = UserDefinedRollout(
            rollout_fn=self.rollout_fn,
            actor=self._serving_backend.actor,  # Use serving backend for generation
            config=self.rollout_config,
        )

        reward_fn = UserDefinedReward(
            reward_fn=self.reward_fn,
            config=self.reward_config,
        )

        # Create GRPO trainer with backends
        self._trainer = GRPOTrainer(
            config=self.trainer_config,
            training_backend=self._training_backend,
            serving_backend=self._serving_backend,
            reference=self._reference,
            reward_fn=reward_fn,
            rollout_generator=rollout_generator,
        )

        # Train
        print(f"\nTraining on {len(dataset)} prompts")
        print("=" * 60)
        self._trainer.train(dataset)
        print("=" * 60)
        print("✓ Training complete")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        if self._trainer is None:
            raise RuntimeError("No trainer to save. Call train() first.")

        self._trainer.save_checkpoint(path)
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        if self._trainer is None:
            raise RuntimeError("No trainer to load into. Call train() first.")

        self._trainer.load_checkpoint(path)
        print(f"✓ Checkpoint loaded from {path}")
