"""GRPO (Group Relative Policy Optimization) trainer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from flashrl.framework.config import TrainerConfig
from flashrl.framework.data_models import (
    Prompt,
    RewardOutput,
    TrainingBatch,
)
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout

if TYPE_CHECKING:
    from flashrl.framework.backends.serving import ServingBackend
    from flashrl.framework.backends.training import TrainingBackend
    from flashrl.framework.run_logger import RunLogger


class GRPOTrainer:
    """GRPO trainer implementation.

    The local path can run with or without a frozen reference model. When the
    reference is disabled, FlashRL uses a simplified no-KL objective so the
    tutorial path stays lightweight on a Mac.
    """

    def __init__(
        self,
        config: TrainerConfig,
        training_backend: "TrainingBackend",
        serving_backend: "ServingBackend",
        reference: ReferenceModel | None,
        reward_fn: UserDefinedReward,
        rollout_generator: UserDefinedRollout,
        run_logger: "RunLogger | None" = None,
    ) -> None:
        """Initialize GRPO trainer."""
        self.config = config
        self.run_logger = run_logger
        self.current_epoch = 0
        self.total_steps = 0
        self.training_backend = training_backend
        self.serving_backend = serving_backend
        self.reference = reference
        self.reward_fn = reward_fn
        self.rollout_generator = rollout_generator

    def attach_run_logger(self, run_logger: "RunLogger | None") -> None:
        """Attach or clear the current run-scoped logger."""
        self.run_logger = run_logger

    def reset_state(self) -> None:
        """Reset per-run trainer state."""
        self.current_epoch = 0
        self.total_steps = 0

    def train(self, dataset: Any) -> None:
        """Train on the given dataset."""
        batch_size = self.config.batch_size
        batches_in_epoch = math.ceil(len(dataset) / batch_size) if len(dataset) else 0
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_number = epoch + 1

            if self.run_logger is not None:
                self.run_logger.log_epoch_start(
                    epoch_number,
                    self.config.max_epochs,
                    batches_in_epoch,
                )

            for batch_index, prompts in enumerate(self._batch_prompts(dataset, batch_size), start=1):
                step = self.total_steps + 1

                # Generate rollouts
                rollouts = self.rollout_generator.generate(prompts)

                # Compute rewards
                rewards = [self.reward_fn.compute(rollout) for rollout in rollouts]

                # Create training batch
                batch = TrainingBatch(
                    prompts=prompts,
                    conversations=[rollout.conversation for rollout in rollouts],
                    rollouts=rollouts,
                    rewards=rewards,
                )

                # Compute advantages
                advantages = self._compute_advantages(batch.rewards)

                # Compute loss
                loss_dict = self._compute_grpo_loss(batch, advantages)

                # Optimize
                self.training_backend.optimizer.zero_grad()
                loss_dict["loss"].backward()
                self.training_backend.optimizer.step()
                self.training_backend.sync_weights_to(self.serving_backend)

                # Log progress
                if self.run_logger is not None:
                    metrics = {
                        "loss": float(loss_dict["loss"].item()),
                        "policy_loss": float(loss_dict["policy_loss"].item()),
                        "kl_divergence": float(loss_dict["kl_divergence"].item()),
                    }
                    self.run_logger.log_step(step, epoch_number, metrics)

                self.total_steps = step

    def step(self, batch: TrainingBatch) -> dict[str, float]:
        """Perform one GRPO training step."""
        advantages = self._compute_advantages(batch.rewards)
        loss_dict = self._compute_grpo_loss(batch, advantages)

        self.training_backend.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.training_backend.optimizer.step()
        self.training_backend.sync_weights_to(self.serving_backend)

        return {
            "loss": float(loss_dict["loss"].item()),
            "policy_loss": float(loss_dict["policy_loss"].item()),
            "kl_divergence": float(loss_dict["kl_divergence"].item()),
        }

    def _compute_advantages(
        self,
        rewards: list[RewardOutput],
    ) -> torch.Tensor:
        """Compute group-based relative advantages."""
        reward_values = torch.tensor(
            [reward.reward for reward in rewards],
            dtype=torch.float32,
        )
        mean = reward_values.mean()
        std = reward_values.std(unbiased=False)
        return (reward_values - mean) / (std + 1e-8)

    def _compute_grpo_loss(
        self,
        batch: TrainingBatch,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute GRPO loss with optional KL regularization."""
        actor = self.training_backend.actor
        device = actor.device

        prompts = [prompt.text for prompt in batch.prompts]
        responses = [rollout.text for rollout in batch.rollouts]
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]

        # Tokenize full sequences
        full_inputs = actor.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=actor.config.max_length,
            return_tensors="pt",
        )
        input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs["attention_mask"].to(device)

        # Tokenize prompts to find response boundaries
        prompt_inputs = actor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=actor.config.max_length,
            return_tensors="pt",
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

        # Forward pass through actor
        actor_logits = actor.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        # Forward pass through reference (if enabled)
        ref_logits = None
        if self.reference is not None and self.config.kl_coefficient > 0.0:
            with torch.no_grad():
                ref_logits = self.reference.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

        # Compute loss
        actor_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
        shift_ids = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        actor_token_lp = torch.gather(
            actor_log_probs,
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Create response mask
        response_mask = torch.zeros_like(shift_mask)
        for index, prompt_length in enumerate(prompt_lengths):
            prompt_start = int(prompt_length.item()) - 1
            response_mask[index, prompt_start:] = shift_mask[index, prompt_start:]

        # Compute policy signal and KL divergence
        if ref_logits is not None:
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_token_lp = torch.gather(
                ref_log_probs,
                dim=-1,
                index=shift_ids.unsqueeze(-1),
            ).squeeze(-1)
            policy_signal = actor_token_lp - ref_token_lp
            kl_divergence = (policy_signal * response_mask).sum() / response_mask.sum().clamp(min=1)
        else:
            policy_signal = actor_token_lp
            kl_divergence = torch.zeros((), device=device, dtype=actor_logits.dtype)

        # Compute policy loss with advantages
        masked_policy_signal = policy_signal * response_mask
        advantages_expanded = advantages.to(device).unsqueeze(-1) * response_mask

        num_response_tokens = response_mask.sum().clamp(min=1)
        policy_loss = -(advantages_expanded * masked_policy_signal).sum() / num_response_tokens
        loss = policy_loss + self.config.kl_coefficient * kl_divergence

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "kl_divergence": kl_divergence,
        }

    def _batch_prompts(self, dataset: Any, batch_size: int | None = None) -> list[list[Prompt]]:
        """Split dataset into batches."""
        size = batch_size or self.config.batch_size
        for index in range(0, len(dataset), size):
            yield dataset[index:index + size]

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
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
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.training_backend.actor.model.load_state_dict(checkpoint["actor_state_dict"])
        self.training_backend.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.total_steps = checkpoint["total_steps"]
        self.training_backend.sync_weights_to(self.serving_backend)
