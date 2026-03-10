"""GRPO (Group Relative Policy Optimization) trainer."""

from typing import TYPE_CHECKING, Any
import torch
import torch.nn.functional as F

from flashrl.framework.config import TrainerConfig
from flashrl.framework.trainer.base import BaseTrainer
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    RewardOutput,
    TrainingBatch,
)
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.reward.base import BaseReward
from flashrl.framework.rollout.base import BaseRollout

if TYPE_CHECKING:
    from flashrl.framework.backends.training import TrainingBackend
    from flashrl.framework.backends.serving import ServingBackend


class GRPOTrainer(BaseTrainer):
    """GRPO trainer implementation.

    GRPO uses group-based relative advantages instead of a value function.
    For each prompt, we generate multiple outputs (a group) and normalize
    rewards within the group to compute advantages.
    """

    def __init__(
        self,
        config: TrainerConfig,
        training_backend: "TrainingBackend",
        serving_backend: "ServingBackend",
        reference: ReferenceModel,
        reward_fn: BaseReward,
        rollout_generator: BaseRollout,
    ) -> None:
        """Initialize GRPO trainer.

        Args:
            config: Trainer configuration.
            training_backend: Training backend with training model and optimizer.
            serving_backend: Serving backend with serving model for generation.
            reference: Reference model (frozen) for KL divergence.
            reward_fn: Reward function.
            rollout_generator: Rollout generation strategy.
        """
        super().__init__(config)
        self.training_backend = training_backend
        self.serving_backend = serving_backend
        self.reference = reference
        self.reward_fn = reward_fn
        self.rollout_generator = rollout_generator

    def train(self, dataset: Any) -> None:
        """Train on the given dataset.

        Args:
            dataset: Training data (list of prompts or similar).
        """
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch + 1}/{self.config.max_epochs}")

            # TODO: Proper data loading and batching
            # For now, assume dataset is list of prompts
            for prompts in self._batch_prompts(dataset):
                batch = self._create_batch(prompts)
                metrics = self.step(batch)

                if self.total_steps % 10 == 0:
                    print(f"Step {self.total_steps}: loss={metrics.get('loss', 0):.4f}")

                self.total_steps += 1

    def step(self, batch: TrainingBatch) -> dict[str, float]:
        """Perform one GRPO training step.

        Args:
            batch: Training batch with rollouts and rewards.

        Returns:
            Dictionary of metrics (loss, kl_divergence, etc).
        """
        # Compute group-based advantages
        advantages = self._compute_advantages(batch.rewards)

        # Compute GRPO loss
        loss_dict = self._compute_grpo_loss(batch, advantages)

        # Backward pass using training backend optimizer
        self.training_backend.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.training_backend.optimizer.step()

        return {
            "loss": loss_dict["loss"].item(),
            "policy_loss": loss_dict["policy_loss"].item(),
            "kl_divergence": loss_dict["kl_divergence"].item(),
        }

    def _compute_advantages(
        self,
        rewards: list[RewardOutput],
    ) -> torch.Tensor:
        """Compute group-based relative advantages.

        For GRPO, advantages are computed by normalizing rewards within
        each group (prompts that generated these rollouts).

        Args:
            rewards: List of reward outputs.

        Returns:
            Advantages tensor.
        """
        reward_values = torch.tensor(
            [r.reward for r in rewards],
            dtype=torch.float32,
        )

        # Group-based normalization (relative to mean)
        mean = reward_values.mean()
        std = reward_values.std(dim=0, keepdim=True)
        advantages = (reward_values - mean) / (std + 1e-8)

        return advantages

    def _compute_grpo_loss(
        self,
        batch: TrainingBatch,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute GRPO loss.

        Loss = -advantage * log_ratio + kl_coefficient * kl_divergence

        log_ratio = log(actor_prob) - log(ref_prob) for each response token.

        Args:
            batch: Training batch.
            advantages: Per-sequence advantages (batch_size,).

        Returns:
            Dictionary with loss tensors.
        """
        actor = self.training_backend.actor
        ref = self.reference
        device = actor.device

        # Reconstruct texts from batch
        prompts = [p.text for p in batch.prompts]
        responses = [r.text for r in batch.rollouts]
        full_texts = [p + r for p, r in zip(prompts, responses)]

        # Tokenize full sequences (right padding)
        full_inputs = actor.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=actor.config.max_length,
            return_tensors="pt",
        )
        input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs["attention_mask"].to(device)

        # Tokenize prompts to find where responses start
        prompt_inputs = actor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=actor.config.max_length,
            return_tensors="pt",
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

        # Forward pass through actor (with gradients)
        actor_logits = actor.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        # Forward pass through reference (no gradients)
        with torch.no_grad():
            ref_logits = ref.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        # Compute per-token log probs.
        # logits[i] predicts token[i+1], so we shift:
        #   shift_logits = logits[:, :-1]  (predicts tokens 1..N)
        #   shift_ids = input_ids[:, 1:]    (actual tokens 1..N)
        actor_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
        ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)

        shift_ids = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        # Gather log prob of each actual token
        actor_token_lp = torch.gather(
            actor_log_probs, dim=-1, index=shift_ids.unsqueeze(-1)
        ).squeeze(-1)
        ref_token_lp = torch.gather(
            ref_log_probs, dim=-1, index=shift_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Build response mask: only compute loss on response tokens.
        # In shifted arrays, response tokens start at index (prompt_length - 1)
        # because shift_ids[prompt_length-1] = input_ids[prompt_length] = first response token.
        response_mask = torch.zeros_like(shift_mask)
        for i in range(len(prompts)):
            p_len = prompt_lengths[i].item()
            response_mask[i, p_len - 1 :] = shift_mask[i, p_len - 1 :]

        # Log ratio = actor log prob - reference log prob
        log_ratio = actor_token_lp - ref_token_lp

        # Mask to response tokens only
        masked_log_ratio = log_ratio * response_mask

        # Broadcast advantages (per-sequence) to per-token
        advantages_expanded = advantages.to(device).unsqueeze(-1) * response_mask

        # Policy loss: -mean(advantage * log_ratio) over response tokens
        num_response_tokens = response_mask.sum().clamp(min=1)
        policy_loss = -(advantages_expanded * masked_log_ratio).sum() / num_response_tokens

        # KL divergence: mean of log ratio over response tokens
        kl_div = (masked_log_ratio).sum() / num_response_tokens
        kl_penalty = self.config.kl_coefficient * kl_div

        # Total loss
        loss = policy_loss + kl_penalty

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "kl_divergence": kl_div,
        }

    def _create_batch(self, prompts: list[Prompt]) -> TrainingBatch:
        """Create a training batch from prompts.

        Args:
            prompts: List of input prompts.

        Returns:
            Training batch with rollouts and rewards.
        """
        # Generate rollouts (rollout generator already has serving backend actor)
        rollouts = self.rollout_generator.generate(prompts)

        # Compute rewards
        rewards = [self.reward_fn.compute(r) for r in rollouts]

        return TrainingBatch(
            prompts=prompts,
            conversations=[r.conversation for r in rollouts],
            rollouts=rollouts,
            rewards=rewards,
        )

    def _batch_prompts(self, dataset: Any, batch_size: int | None = None) -> list[list[Prompt]]:
        """Split dataset into batches.

        Args:
            dataset: Training data.
            batch_size: Batch size (default: from config).

        Yields:
            Batches of prompts.
        """
        size = batch_size or self.config.batch_size

        # TODO: Proper data loading
        # For now, simple batching
        for i in range(0, len(dataset), size):
            yield dataset[i : i + size]

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save(
            {
                "actor_state_dict": self.training_backend.actor.model.state_dict(),
                "optimizer_state_dict": self.training_backend.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, weights_only=False)
        self.training_backend.actor.model.load_state_dict(checkpoint["actor_state_dict"])
        self.training_backend.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.total_steps = checkpoint["total_steps"]
