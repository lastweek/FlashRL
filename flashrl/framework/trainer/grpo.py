"""GRPO (Group Relative Policy Optimization) trainer."""

from __future__ import annotations

import math
import time
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
# Using concrete reward and rollout classes
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
# Removed BaseTrainer inheritance - using direct class

if TYPE_CHECKING:
    from flashrl.framework.backends.serving import ServingBackend
    from flashrl.framework.backends.training import TrainingBackend
    from flashrl.framework.run_logger import RunLogger


HOT_PATH_SUMMARY_KEYS = [
    "rollout_seconds",
    "reward_seconds",
    "advantage_seconds",
    "tokenize_full_seconds",
    "tokenize_prompt_seconds",
    "actor_forward_seconds",
    "reference_forward_seconds",
    "loss_assembly_seconds",
    "backward_seconds",
    "optimizer_step_seconds",
    "weight_sync_seconds",
    "logging_seconds",
]

PHASE_BREAKDOWN_SPEC = [
    ("rollout", "rollout_seconds"),
    ("reward", "reward_seconds"),
    ("advantage", "advantage_seconds"),
    ("tokenize_full", "tokenize_full_seconds"),
    ("tokenize_prompt", "tokenize_prompt_seconds"),
    ("actor_forward", "actor_forward_seconds"),
    ("reference_forward", "reference_forward_seconds"),
    ("loss_assembly", "loss_assembly_seconds"),
    ("backward", "backward_seconds"),
    ("optimizer", "optimizer_step_seconds"),
    ("sync", "weight_sync_seconds"),
]

PHASE_GROUP_SPECS = [
    ("rollout", ("rollout_seconds",)),
    ("reward", ("reward_seconds",)),
    (
        "calculate_loss",
        (
            "advantage_seconds",
            "tokenize_full_seconds",
            "tokenize_prompt_seconds",
            "actor_forward_seconds",
            "reference_forward_seconds",
            "loss_assembly_seconds",
        ),
    ),
    (
        "train",
        (
            "backward_seconds",
            "optimizer_step_seconds",
            "weight_sync_seconds",
        ),
    ),
]


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
            epoch_metrics: list[dict[str, Any]] = []
            epoch_started_at = time.perf_counter()

            if self.run_logger is not None:
                self.run_logger.log_epoch_start(
                    epoch_number,
                    self.config.max_epochs,
                    batches_in_epoch,
                )

            for batch_index, prompts in enumerate(self._batch_prompts(dataset, batch_size), start=1):
                step = self.total_steps + 1
                step_started_at = time.perf_counter()
                step_context = self._build_step_context(
                    step=step,
                    epoch=epoch_number,
                    total_epochs=self.config.max_epochs,
                    batch_index=batch_index,
                    batches_in_epoch=batches_in_epoch,
                    batch_size=len(prompts),
                )
                logging_seconds = 0.0

                self._update_stage(epoch_number, "rollout", batch_index)
                rollout_started_at = time.perf_counter()
                rollouts = self._generate_rollouts(prompts, epoch_number, batch_index, step)
                rollout_seconds = time.perf_counter() - rollout_started_at
                token_stats = self._compute_step_token_stats(prompts, rollouts)
                logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="rollout",
                        stage="rollout",
                        latency_seconds=rollout_seconds,
                        prompt_tokens_mean=token_stats["prompt_tokens_mean"],
                        prompt_tokens_max=token_stats["prompt_tokens_max"],
                        response_tokens_mean=token_stats["response_tokens_mean"],
                        response_tokens_max=token_stats["response_tokens_max"],
                    )
                )

                self._update_stage(epoch_number, "reward", batch_index)
                reward_started_at = time.perf_counter()
                rewards = self._compute_rewards(prompts, rollouts, epoch_number, batch_index, step)
                reward_seconds = time.perf_counter() - reward_started_at
                reward_values = torch.tensor(
                    [reward.reward for reward in rewards],
                    dtype=torch.float32,
                )
                logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="reward",
                        stage="reward",
                        latency_seconds=reward_seconds,
                        **self._tensor_summary("reward", reward_values),
                    )
                )

                batch = TrainingBatch(
                    prompts=prompts,
                    conversations=[rollout.conversation for rollout in rollouts],
                    rollouts=rollouts,
                    rewards=rewards,
                )

                self._update_stage(epoch_number, "advantage", batch_index)
                advantage_started_at = time.perf_counter()
                advantages = self._compute_advantages(batch.rewards)
                advantage_seconds = time.perf_counter() - advantage_started_at
                logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="advantage",
                        stage="calculate_loss",
                        latency_seconds=advantage_seconds,
                        **self._tensor_summary("advantage", advantages),
                    )
                )

                self._update_stage(epoch_number, "forward", batch_index)
                loss_dict = self._compute_grpo_loss(
                    batch,
                    advantages,
                    step_context=step_context,
                    token_stats=token_stats,
                )
                logging_seconds += float(loss_dict.pop("phase_logging_seconds", 0.0))

                self._update_stage(epoch_number, "backward", batch_index)
                backward_started_at = time.perf_counter()
                self.training_backend.optimizer.zero_grad()
                loss_dict["loss"].backward()
                backward_seconds = time.perf_counter() - backward_started_at
                logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="backward",
                        stage="train",
                        latency_seconds=backward_seconds,
                        loss=float(loss_dict["loss"].item()),
                    )
                )

                self._update_stage(epoch_number, "optimizer", batch_index)
                optimizer_started_at = time.perf_counter()
                self.training_backend.optimizer.step()
                optimizer_step_seconds = time.perf_counter() - optimizer_started_at
                logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="optimizer",
                        stage="train",
                        latency_seconds=optimizer_step_seconds,
                        learning_rate=float(self.training_backend.optimizer.param_groups[0]["lr"]),
                    )
                )

                self._update_stage(epoch_number, "sync", batch_index)
                sync_started_at = time.perf_counter()
                self.training_backend.sync_weights_to(self.serving_backend)
                weight_sync_seconds = time.perf_counter() - sync_started_at

                step_duration_seconds = time.perf_counter() - step_started_at
                response_tokens = int(round(float(loss_dict["response_tokens"].item())))
                tokens_per_second = float(response_tokens / max(step_duration_seconds, 1e-8))
                logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="sync",
                        stage="train",
                        latency_seconds=weight_sync_seconds,
                        step_duration_seconds=float(step_duration_seconds),
                        tokens_per_second=tokens_per_second,
                    )
                )
                reference_active = bool(self.reference is not None and self.config.kl_coefficient > 0.0)
                timings = {
                    "rollout_seconds": float(rollout_seconds),
                    "reward_seconds": float(reward_seconds),
                    "reward_per_item_mean_seconds": float(
                        reward_seconds / max(len(rewards), 1)
                    ),
                    "advantage_seconds": float(advantage_seconds),
                    "tokenize_full_seconds": float(loss_dict["tokenize_full_seconds"]),
                    "tokenize_prompt_seconds": float(loss_dict["tokenize_prompt_seconds"]),
                    "actor_forward_seconds": float(loss_dict["actor_forward_seconds"]),
                    "reference_forward_seconds": float(loss_dict["reference_forward_seconds"]),
                    "loss_assembly_seconds": float(loss_dict["loss_assembly_seconds"]),
                    "backward_seconds": float(backward_seconds),
                    "optimizer_step_seconds": float(optimizer_step_seconds),
                    "weight_sync_seconds": float(weight_sync_seconds),
                }

                metrics = {
                    "epoch": float(epoch_number),
                    "total_epochs": float(self.config.max_epochs),
                    "step": float(step),
                    "batch_index": float(batch_index),
                    "batches_in_epoch": float(batches_in_epoch),
                    "batch_size": float(len(prompts)),
                    "prompt_count": float(len(prompts)),
                    "response_tokens": float(response_tokens),
                    "response_tokens_total": float(response_tokens),
                    "reference_enabled": self.reference is not None,
                    "reference_active": reference_active,
                    "loss": float(loss_dict["loss"].item()),
                    "policy_loss": float(loss_dict["policy_loss"].item()),
                    "kl_divergence": float(loss_dict["kl_divergence"].item()),
                    "reward_mean": float(reward_values.mean().item()),
                    "reward_std": float(reward_values.std(unbiased=False).item()),
                    "reward_min": float(reward_values.min().item()),
                    "reward_max": float(reward_values.max().item()),
                    "advantage_mean": float(advantages.mean().item()),
                    "advantage_std": float(advantages.std(unbiased=False).item()),
                    "advantage_min": float(advantages.min().item()),
                    "advantage_max": float(advantages.max().item()),
                    "tokens_per_second": tokens_per_second,
                    "step_duration_seconds": float(step_duration_seconds),
                    "timings": timings,
                    "phase_breakdown": self._build_phase_breakdown(
                        timings,
                        reference_active=reference_active,
                    ),
                    "phase_groups": self._build_phase_groups(timings),
                }
                metrics["accounted_seconds"] = float(
                    sum(
                        value
                        for key, value in timings.items()
                        if key != "reward_per_item_mean_seconds"
                    )
                )
                metrics["unattributed_seconds"] = max(
                    0.0,
                    metrics["step_duration_seconds"] - metrics["accounted_seconds"],
                )
                metrics["dominant_phase"] = max(
                    (
                        (key, value)
                        for key, value in timings.items()
                        if key != "reward_per_item_mean_seconds"
                    ),
                    key=lambda item: item[1],
                )[0]

                log_payload = self._batch_metrics_payload(metrics)
                if self.run_logger is not None:
                    logging_seconds += self._log_step_done(log_payload)
                    if self.run_logger.should_log_sample(step):
                        logging_seconds += self._log_sample_preview(
                            step=step,
                            prompt=prompts[0].text,
                            response=rollouts[0].text,
                            reward=rewards[0].reward,
                        )
                    self.run_logger.record_logging_overhead(logging_seconds)

                metrics["timings"]["logging_seconds"] = logging_seconds
                metrics["accounted_seconds"] += logging_seconds
                metrics["unattributed_seconds"] = max(
                    0.0,
                    metrics["step_duration_seconds"] - metrics["accounted_seconds"],
                )
                metrics["dominant_phase"] = max(
                    (
                        (key, value)
                        for key, value in metrics["timings"].items()
                        if key != "reward_per_item_mean_seconds"
                    ),
                    key=lambda item: item[1],
                )[0]

                epoch_metrics.append(metrics)
                self.total_steps = step

            if self.run_logger is not None:
                self.run_logger.log_epoch_summary(
                    epoch_number,
                    self._summarize_epoch_metrics(epoch_metrics),
                    duration_seconds=time.perf_counter() - epoch_started_at,
                )

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
        *,
        step_context: dict[str, Any] | None = None,
        token_stats: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor | float]:
        """Compute GRPO loss with optional KL regularization."""
        actor = self.training_backend.actor
        device = actor.device
        phase_logging_seconds = 0.0

        prompts = [prompt.text for prompt in batch.prompts]
        responses = [rollout.text for rollout in batch.rollouts]
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]

        tokenize_full_started_at = time.perf_counter()
        full_inputs = actor.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=actor.config.max_length,
            return_tensors="pt",
        )
        input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs["attention_mask"].to(device)
        tokenize_full_seconds = time.perf_counter() - tokenize_full_started_at
        if step_context is not None and token_stats is not None:
            phase_logging_seconds += self._log_step_phase(
                self._phase_payload(
                    step_context,
                    phase="tokenize_full",
                    stage="calculate_loss",
                    latency_seconds=tokenize_full_seconds,
                    full_tokens_mean=token_stats["full_tokens_mean"],
                    full_tokens_max=token_stats["full_tokens_max"],
                )
            )

        tokenize_prompt_started_at = time.perf_counter()
        prompt_inputs = actor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=actor.config.max_length,
            return_tensors="pt",
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        tokenize_prompt_seconds = time.perf_counter() - tokenize_prompt_started_at
        if step_context is not None and token_stats is not None:
            phase_logging_seconds += self._log_step_phase(
                self._phase_payload(
                    step_context,
                    phase="tokenize_prompt",
                    stage="calculate_loss",
                    latency_seconds=tokenize_prompt_seconds,
                    prompt_tokens_mean=token_stats["prompt_tokens_mean"],
                    prompt_tokens_max=token_stats["prompt_tokens_max"],
                )
            )

        actor_forward_started_at = time.perf_counter()
        actor_logits = actor.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        actor_forward_seconds = time.perf_counter() - actor_forward_started_at
        if step_context is not None and token_stats is not None:
            phase_logging_seconds += self._log_step_phase(
                self._phase_payload(
                    step_context,
                    phase="actor_forward",
                    stage="calculate_loss",
                    latency_seconds=actor_forward_seconds,
                    full_tokens_total=token_stats["full_tokens_total"],
                )
            )

        reference_forward_seconds = 0.0
        ref_logits = None
        if self.reference is not None and self.config.kl_coefficient > 0.0:
            reference_forward_started_at = time.perf_counter()
            with torch.no_grad():
                ref_logits = self.reference.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            reference_forward_seconds = time.perf_counter() - reference_forward_started_at
            if step_context is not None and token_stats is not None:
                phase_logging_seconds += self._log_step_phase(
                    self._phase_payload(
                        step_context,
                        phase="reference_forward",
                        stage="calculate_loss",
                        latency_seconds=reference_forward_seconds,
                        full_tokens_total=token_stats["full_tokens_total"],
                    )
                )

        loss_assembly_started_at = time.perf_counter()
        actor_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
        shift_ids = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        actor_token_lp = torch.gather(
            actor_log_probs,
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)

        response_mask = torch.zeros_like(shift_mask)
        for index, prompt_length in enumerate(prompt_lengths):
            prompt_start = int(prompt_length.item()) - 1
            response_mask[index, prompt_start:] = shift_mask[index, prompt_start:]

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

        masked_policy_signal = policy_signal * response_mask
        advantages_expanded = advantages.to(device).unsqueeze(-1) * response_mask

        num_response_tokens = response_mask.sum().clamp(min=1)
        policy_loss = -(advantages_expanded * masked_policy_signal).sum() / num_response_tokens
        loss = policy_loss + self.config.kl_coefficient * kl_divergence
        loss_assembly_seconds = time.perf_counter() - loss_assembly_started_at
        if step_context is not None:
            phase_logging_seconds += self._log_step_phase(
                self._phase_payload(
                    step_context,
                    phase="loss_assembly",
                    stage="calculate_loss",
                    latency_seconds=loss_assembly_seconds,
                    loss=float(loss.item()),
                    policy_loss=float(policy_loss.item()),
                    kl_divergence=float(kl_divergence.item()),
                    response_tokens_total=float(num_response_tokens.detach().item()),
                )
            )

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "kl_divergence": kl_divergence,
            "response_tokens": num_response_tokens.detach(),
            "tokenize_full_seconds": tokenize_full_seconds,
            "tokenize_prompt_seconds": tokenize_prompt_seconds,
            "actor_forward_seconds": actor_forward_seconds,
            "reference_forward_seconds": reference_forward_seconds,
            "loss_assembly_seconds": loss_assembly_seconds,
            "phase_logging_seconds": phase_logging_seconds,
        }

    def _batch_prompts(self, dataset: Any, batch_size: int | None = None) -> list[list[Prompt]]:
        """Split dataset into batches."""
        size = batch_size or self.config.batch_size
        for index in range(0, len(dataset), size):
            yield dataset[index:index + size]

    def _generate_rollouts(
        self,
        prompts: list[Prompt],
        epoch: int,
        batch_index: int,
        step: int,
    ):
        try:
            return self.rollout_generator.generate(prompts)
        except Exception as exc:
            if self.run_logger is not None:
                self.run_logger.log_exception(
                    exc,
                    context={
                        "stage": "rollout",
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "step": step,
                        "prompt_previews": [self._preview(prompt.text) for prompt in prompts[:3]],
                    },
                )
            raise

    def _compute_rewards(
        self,
        prompts: list[Prompt],
        rollouts,
        epoch: int,
        batch_index: int,
        step: int,
    ) -> list[RewardOutput]:
        rewards: list[RewardOutput] = []
        for index, rollout in enumerate(rollouts):
            try:
                rewards.append(self.reward_fn.compute(rollout))
            except Exception as exc:
                if self.run_logger is not None:
                    prompt_text = prompts[index].text if index < len(prompts) else ""
                    self.run_logger.log_exception(
                        exc,
                        context={
                            "stage": "reward",
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "step": step,
                            "prompt_preview": self._preview(prompt_text),
                            "response_preview": self._preview(rollout.text),
                        },
                    )
                raise
        return rewards

    def _batch_metrics_payload(self, metrics: dict[str, Any]) -> dict[str, Any]:
        payload = dict(metrics)
        payload["epoch"] = int(payload["epoch"])
        payload["total_epochs"] = int(payload["total_epochs"])
        payload["step"] = int(payload["step"])
        payload["batch_index"] = int(payload["batch_index"])
        payload["batches_in_epoch"] = int(payload["batches_in_epoch"])
        payload["batch_size"] = int(payload["batch_size"])
        payload["prompt_count"] = int(payload["prompt_count"])
        payload["response_tokens"] = int(payload["response_tokens"])
        payload["response_tokens_total"] = int(payload["response_tokens_total"])
        payload["timings"] = {
            key: float(value)
            for key, value in payload["timings"].items()
            if key != "logging_seconds"
        }
        payload["phase_breakdown"] = [
            {
                "name": str(item["name"]),
                "seconds": float(item["seconds"]),
            }
            for item in payload["phase_breakdown"]
        ]
        payload["phase_groups"] = {
            str(key): float(value)
            for key, value in payload["phase_groups"].items()
        }
        return payload

    def _summarize_epoch_metrics(self, epoch_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if not epoch_metrics:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl_divergence": 0.0,
                "reward_mean": 0.0,
                "tokens_per_second": 0.0,
                "hot_path_totals": {},
                "hot_path_percentages": {},
                "phase_group_totals": {},
                "phase_group_percentages": {},
                "avg_step_seconds": 0.0,
                "slowest_step_seconds": 0.0,
                "slowest_phase": "n/a",
                "no_training_steps_completed": True,
            }

        averaged_keys = [
            "loss",
            "policy_loss",
            "kl_divergence",
            "reward_mean",
            "tokens_per_second",
        ]
        summary = {
            key: sum(float(metric[key]) for metric in epoch_metrics) / len(epoch_metrics)
            for key in averaged_keys
        }

        hot_path_totals = {
            key: sum(float(metric["timings"].get(key, 0.0)) for metric in epoch_metrics)
            for key in HOT_PATH_SUMMARY_KEYS
        }
        hot_path_totals = {
            key: value
            for key, value in hot_path_totals.items()
            if value > 0
        }
        hot_path_denominator = sum(hot_path_totals.values())
        hot_path_percentages = {
            key: (value / hot_path_denominator) * 100.0 if hot_path_denominator > 0 else 0.0
            for key, value in hot_path_totals.items()
        }
        phase_group_totals = {
            name: sum(float(metric["phase_groups"].get(name, 0.0)) for metric in epoch_metrics)
            for name, _ in PHASE_GROUP_SPECS
        }
        phase_group_totals = {
            key: value
            for key, value in phase_group_totals.items()
            if value > 0
        }
        phase_group_denominator = sum(phase_group_totals.values())
        phase_group_percentages = {
            key: (value / phase_group_denominator) * 100.0
            if phase_group_denominator > 0
            else 0.0
            for key, value in phase_group_totals.items()
        }

        slowest_metric = max(epoch_metrics, key=lambda metric: float(metric["step_duration_seconds"]))
        summary.update(
            {
                "hot_path_totals": hot_path_totals,
                "hot_path_percentages": hot_path_percentages,
                "phase_group_totals": phase_group_totals,
                "phase_group_percentages": phase_group_percentages,
                "avg_step_seconds": sum(
                    float(metric["step_duration_seconds"]) for metric in epoch_metrics
                ) / len(epoch_metrics),
                "slowest_step_seconds": float(slowest_metric["step_duration_seconds"]),
                "slowest_phase": str(slowest_metric["dominant_phase"]),
                "no_training_steps_completed": False,
            }
        )
        return summary

    def _preview(self, text: str, limit: int = 140) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _build_step_context(
        self,
        *,
        step: int,
        epoch: int,
        total_epochs: int,
        batch_index: int,
        batches_in_epoch: int,
        batch_size: int,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch_index": batch_index,
            "batches_in_epoch": batches_in_epoch,
            "batch_size": batch_size,
        }

    def _phase_payload(
        self,
        step_context: dict[str, Any],
        *,
        phase: str,
        stage: str,
        latency_seconds: float,
        **fields: Any,
    ) -> dict[str, Any]:
        payload = {
            **step_context,
            "phase": phase,
            "stage": stage,
            "latency_seconds": float(latency_seconds),
        }
        payload.update(fields)
        return payload

    def _log_step_phase(self, payload: dict[str, Any]) -> float:
        if self.run_logger is None:
            return 0.0
        started_at = time.perf_counter()
        self.run_logger.log_step_phase(payload)
        return time.perf_counter() - started_at

    def _log_step_done(self, payload: dict[str, Any]) -> float:
        if self.run_logger is None:
            return 0.0
        started_at = time.perf_counter()
        self.run_logger.log_step_done(payload)
        return time.perf_counter() - started_at

    def _log_sample_preview(
        self,
        *,
        step: int,
        prompt: str,
        response: str,
        reward: float,
    ) -> float:
        if self.run_logger is None:
            return 0.0
        started_at = time.perf_counter()
        self.run_logger.log_sample_preview(
            step=step,
            prompt=prompt,
            response=response,
            reward=reward,
        )
        return time.perf_counter() - started_at

    def _tensor_summary(self, prefix: str, values: torch.Tensor) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(values.mean().item()),
            f"{prefix}_std": float(values.std(unbiased=False).item()),
            f"{prefix}_min": float(values.min().item()),
            f"{prefix}_max": float(values.max().item()),
        }

    def _compute_step_token_stats(
        self,
        prompts: list[Prompt],
        rollouts,
    ) -> dict[str, float]:
        actor = self.training_backend.actor
        tokenizer = actor.tokenizer
        max_length = actor.config.max_length

        prompt_lengths = self._token_lengths(
            tokenizer,
            [prompt.text for prompt in prompts],
            max_length=max_length,
        )
        response_lengths = self._token_lengths(
            tokenizer,
            [rollout.text for rollout in rollouts],
            max_length=max_length,
        )
        full_lengths = self._token_lengths(
            tokenizer,
            [prompt.text + rollout.text for prompt, rollout in zip(prompts, rollouts)],
            max_length=max_length,
        )

        return {
            **self._length_stats("prompt_tokens", prompt_lengths),
            **self._length_stats("response_tokens", response_lengths),
            **self._length_stats("full_tokens", full_lengths),
        }

    def _token_lengths(
        self,
        tokenizer,
        texts: list[str],
        *,
        max_length: int,
    ) -> list[int]:
        if not texts:
            return []
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return [int(length) for length in tokenized["attention_mask"].sum(dim=1).tolist()]

    def _length_stats(self, prefix: str, lengths: list[int]) -> dict[str, float]:
        if not lengths:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_total": 0.0,
            }
        total = float(sum(lengths))
        return {
            f"{prefix}_mean": total / len(lengths),
            f"{prefix}_max": float(max(lengths)),
            f"{prefix}_total": total,
        }

    def _build_phase_breakdown(
        self,
        timings: dict[str, float],
        *,
        reference_active: bool,
    ) -> list[dict[str, float | str]]:
        breakdown = []
        for name, key in PHASE_BREAKDOWN_SPEC:
            if name == "reference_forward" and not reference_active:
                continue
            breakdown.append(
                {
                    "name": name,
                    "seconds": float(timings.get(key, 0.0)),
                }
            )
        return breakdown

    def _build_phase_groups(self, timings: dict[str, float]) -> dict[str, float]:
        return {
            name: sum(float(timings.get(key, 0.0)) for key in keys)
            for name, keys in PHASE_GROUP_SPECS
        }

    def _update_stage(self, epoch: int, stage: str, batch_index: int) -> None:
        if self.run_logger is not None:
            self.run_logger.update_stage(
                epoch=epoch,
                stage=stage,
                batch_index=batch_index,
            )

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
