"""GRPO (Group Relative Policy Optimization) trainer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import TYPE_CHECKING, Any, Callable

import torch

from flashrl.framework.config import GrpoConfig, TrainerConfig
from flashrl.framework.data_models import LearnerBatch, Prompt, RewardOutput, TrainingBatch
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.training import OptimizationResult

if TYPE_CHECKING:
    from flashrl.framework.metrics import MetricsSink
    from flashrl.framework.run_logger import RunLogger
    from flashrl.framework.serving import ServingBackend
    from flashrl.framework.training import TrainingBackend


STAGE_ORDER = (
    "rollout",
    "reward",
    "advantage",
    "prepare_inputs",
    "actor_forward",
    "reference_forward",
    "loss_assembly",
    "backward",
    "optimizer",
    "sync",
)


@dataclass(frozen=True)
class StepContext:
    """Stable metadata shared by all events in a training step."""

    step: int
    epoch: int
    total_epochs: int
    batch_index: int
    batches_in_epoch: int
    batch_size: int
    prompt_count: int
    group_size: int
    dataset_prompt_start: int
    dataset_prompt_end: int
    dataset_prompt_count: int
    planned_prompts_per_step: int
    planned_samples_per_step: int

    def payload(self) -> dict[str, int]:
        """Return the event payload fields shared by all step-stage logs."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "batch_index": self.batch_index,
            "batches_in_epoch": self.batches_in_epoch,
            "batch_size": self.batch_size,
            "prompt_count": self.prompt_count,
            "group_size": self.group_size,
            "dataset_prompt_start": self.dataset_prompt_start,
            "dataset_prompt_end": self.dataset_prompt_end,
            "dataset_prompt_count": self.dataset_prompt_count,
            "planned_prompts_per_step": self.planned_prompts_per_step,
            "planned_samples_per_step": self.planned_samples_per_step,
            "completions_per_prompt": self.group_size,
            "planned_completions_per_step": self.planned_samples_per_step,
            "samples_this_step": self.batch_size,
            "completions_this_step": self.batch_size,
        }

class GRPOTrainer:
    """GRPO trainer implementation with detailed step logging."""

    def __init__(
        self,
        config: TrainerConfig,
        grpo_config: GrpoConfig,
        training_backend: "TrainingBackend",
        serving_backend: "ServingBackend",
        reward_fn: UserDefinedReward,
        rollout_generator: UserDefinedRollout,
        reference: Any | None = None,
        run_logger: "RunLogger | None" = None,
        metrics_sink: "MetricsSink | None" = None,
    ) -> None:
        """Initialize GRPO trainer."""
        self.config = config
        self.run_logger = run_logger
        self.metrics_sink = metrics_sink
        self.current_epoch = 0
        self.total_steps = 0
        self.training_backend = training_backend
        self.serving_backend = serving_backend
        self.grpo_config = grpo_config
        if reference is not None and getattr(self.training_backend, "reference", None) is None:
            self.training_backend.reference = reference
            self.training_backend.reference_enabled = True
        self.reference = getattr(self.training_backend, "reference", reference)
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
        prompts_per_step = self._prompts_per_step()
        batches_in_epoch = math.ceil(len(dataset) / prompts_per_step) if len(dataset) else 0

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_number = epoch + 1
            epoch_started_at = time.perf_counter()
            epoch_step_payloads: list[dict[str, Any]] = []
            epoch_dataset = list(dataset)
            if self.config.shuffle_each_epoch and len(epoch_dataset) > 1:
                random.Random(self.config.seed + epoch).shuffle(epoch_dataset)

            if self.run_logger is not None:
                self.run_logger.log_epoch_start(
                    epoch_number,
                    self.config.max_epochs,
                    batches_in_epoch,
                )

            for batch_index, prompts in enumerate(
                self._batch_prompts(epoch_dataset, prompts_per_step),
                start=1,
            ):
                next_step = self.total_steps + 1
                sample_count = len(prompts) * self.grpo_config.group_size
                context = StepContext(
                    step=next_step,
                    epoch=epoch_number,
                    total_epochs=self.config.max_epochs,
                    batch_index=batch_index,
                    batches_in_epoch=batches_in_epoch,
                    batch_size=sample_count,
                    prompt_count=len(prompts),
                    group_size=self.grpo_config.group_size,
                    dataset_prompt_start=((batch_index - 1) * prompts_per_step) + 1,
                    dataset_prompt_end=min(batch_index * prompts_per_step, len(dataset)),
                    dataset_prompt_count=len(dataset),
                    planned_prompts_per_step=prompts_per_step,
                    planned_samples_per_step=prompts_per_step * self.grpo_config.group_size,
                )
                step_payload = self._run_logged_step(prompts, context)
                epoch_step_payloads.append(step_payload)
                self.total_steps = context.step

            if self.run_logger is not None:
                epoch_duration_seconds = time.perf_counter() - epoch_started_at
                epoch_summary = self._build_epoch_summary(
                    epoch=epoch_number,
                    total_epochs=self.config.max_epochs,
                    duration_seconds=epoch_duration_seconds,
                    step_payloads=epoch_step_payloads,
                )
                self.run_logger.log_epoch_summary(epoch_summary)

    def _run_logged_step(
        self,
        prompts: list[Prompt],
        context: StepContext,
    ) -> dict[str, Any]:
        """Run one full training step and emit detailed stage logs."""
        step_started_at = time.perf_counter()
        if self.run_logger is not None:
            self.run_logger.log_step_start(context.payload())

        rollout_started_at = time.perf_counter()
        self._install_serving_debug(context)
        try:
            grouped_prompts, rollouts, prompt_indices, candidate_indices = (
                self.rollout_generator.generate_grouped(prompts, context.group_size)
            )
        finally:
            self._clear_serving_debug()
        rollout_seconds = time.perf_counter() - rollout_started_at
        prompt_lengths = [len(rollout.prompt_token_ids) for rollout in rollouts]
        response_lengths = [len(rollout.response_token_ids) for rollout in rollouts]
        self._log_stage(
            context,
            "rollout",
            rollout_seconds,
            {
                "sample_count": len(grouped_prompts),
                "prompt_tokens_mean": self._mean(prompt_lengths),
                "prompt_tokens_max": max(prompt_lengths, default=0),
                "response_tokens_mean": self._mean(response_lengths),
                "response_tokens_max": max(response_lengths, default=0),
            },
        )

        rewards, reward_seconds = self._compute_rewards(context, grouped_prompts, rollouts)
        reward_values = [reward.reward for reward in rewards]
        reward_stats = self._summary_stats("reward", reward_values)
        reward_rate_stats = self._reward_rate_stats(rewards)
        reward_per_item_mean_seconds = reward_seconds / len(rewards) if rewards else 0.0
        self._log_stage(
            context,
            "reward",
            reward_seconds,
            {
                **reward_stats,
                **reward_rate_stats,
                "reward_per_item_mean_seconds": reward_per_item_mean_seconds,
            },
        )
        if self.run_logger is not None:
            self.run_logger.log_rollout_batch(
                step=context.step,
                epoch=context.epoch,
                batch_index=context.batch_index,
                batches_in_epoch=context.batches_in_epoch,
                prompts=grouped_prompts,
                rollouts=rollouts,
                rewards=rewards,
                prompt_indices=prompt_indices,
                candidate_indices=candidate_indices,
                group_size=context.group_size,
                prompt_count=context.prompt_count,
            )

        batch = TrainingBatch(
            prompts=grouped_prompts,
            conversations=[rollout.conversation for rollout in rollouts],
            rollouts=rollouts,
            rewards=rewards,
            group_size=context.group_size,
            prompt_count=context.prompt_count,
            prompt_indices=prompt_indices,
            candidate_indices=candidate_indices,
        )

        advantages, advantage_seconds = self._measure(
            lambda: self._compute_advantages(batch.rewards, batch.prompt_count, batch.group_size)
        )
        advantage_values = [float(value) for value in advantages.tolist()]
        advantage_stats = self._summary_stats("advantage", advantage_values)
        self._log_stage(
            context,
            "advantage",
            advantage_seconds,
            advantage_stats,
        )

        learner_batch = self._build_learner_batch(batch, advantages)
        result = self._optimize_batch(learner_batch, context)
        stage_timings = {
            "rollout": rollout_seconds,
            "reward": reward_seconds,
            "advantage": advantage_seconds,
            **result.stage_timings,
        }
        step_duration_seconds = time.perf_counter() - step_started_at
        response_tokens_total = result.response_tokens_total
        if step_duration_seconds > 0.0:
            tokens_per_second = response_tokens_total / step_duration_seconds
        else:
            tokens_per_second = 0.0

        stage_order: list[str] = []
        for stage in STAGE_ORDER:
            if stage == "reference_forward" and not result.reference_active:
                continue
            stage_order.append(stage)

        if stage_timings:
            dominant_stage = max(stage_timings.items(), key=lambda item: item[1])[0]
        else:
            dominant_stage = "n/a"

        reward_mean = self._mean(reward_values)
        reference_enabled = bool(getattr(self.training_backend, "reference_loaded", False))

        payload = {
            **context.payload(),
            "loss": result.loss,
            "policy_loss": result.policy_loss,
            "kl_divergence": result.kl_divergence,
            "learning_rate": result.learning_rate,
            "reward_mean": reward_mean,
            **reward_rate_stats,
            "response_tokens_total": response_tokens_total,
            "tokens_per_second": tokens_per_second,
            "step_duration_seconds": step_duration_seconds,
            "stage_timings": stage_timings,
            "stage_order": stage_order,
            "reference_enabled": reference_enabled,
            "reference_active": result.reference_active,
            "dominant_stage": dominant_stage,
            "sample_count": len(grouped_prompts),
        }

        if self.run_logger is not None:
            self.run_logger.log_step_done(payload)
            self.run_logger.log_sample_preview(
                step=context.step,
                prompt=grouped_prompts[0].text if grouped_prompts else "",
                response=rollouts[0].text if rollouts else "",
                reward=reward_values[0] if reward_values else 0.0,
            )
        if self.metrics_sink is not None:
            self.metrics_sink.observe_step(payload)
            self.metrics_sink.push()

        return payload

    def _compute_rewards(
        self,
        context: StepContext,
        prompts: list[Prompt],
        rollouts: list[Any],
    ) -> tuple[list[RewardOutput], float]:
        """Compute rewards and attach prompt/response context on failure."""
        started_at = time.perf_counter()
        try:
            rewards = [self.reward_fn.compute(rollout) for rollout in rollouts]
        except Exception as exc:
            if self.run_logger is not None:
                self.run_logger.log_exception(
                    exc,
                    context={
                        "stage": "reward",
                        "epoch": context.epoch,
                        "step": context.step,
                        "batch_index": context.batch_index,
                        "prompt_preview": self._truncate(prompts[0].text if prompts else ""),
                        "response_preview": self._truncate(rollouts[0].text if rollouts else ""),
                    },
                )
            raise
        return rewards, time.perf_counter() - started_at

    def _build_learner_batch(
        self,
        batch: TrainingBatch,
        advantages: torch.Tensor,
    ) -> LearnerBatch:
        """Convert controller-owned rollout outputs into one learner batch."""
        return LearnerBatch(
            prompt_token_ids=[
                [int(token_id) for token_id in rollout.prompt_token_ids]
                for rollout in batch.rollouts
            ],
            response_token_ids=[
                [int(token_id) for token_id in rollout.response_token_ids]
                for rollout in batch.rollouts
            ],
            response_token_logprobs=[
                [float(value) for value in rollout.response_token_logprobs]
                for rollout in batch.rollouts
            ],
            advantages=[float(value) for value in advantages.tolist()],
            group_size=batch.group_size,
            prompt_count=batch.prompt_count,
            prompt_indices=list(batch.prompt_indices),
            candidate_indices=list(batch.candidate_indices),
            metadata=dict(batch.metadata),
        )

    def _optimize_batch(
        self,
        learner_batch: LearnerBatch,
        context: StepContext | None,
    ) -> OptimizationResult:
        """Delegate learner-side tensor work to the training backend."""
        result = self.training_backend.optimize_batch(learner_batch)
        for stage in (
            "prepare_inputs",
            "actor_forward",
            "reference_forward",
            "loss_assembly",
            "backward",
            "optimizer",
        ):
            if stage == "reference_forward" and not result.reference_active:
                continue
            self._log_stage(
                context,
                stage,
                result.stage_timings.get(stage, 0.0),
                result.stage_metrics.get(stage, {}),
            )

        _, sync_seconds = self._measure(
            lambda: self.training_backend.sync_weights_to(self.serving_backend)
        )
        result.stage_timings["sync"] = sync_seconds
        if context is not None:
            partial_step_seconds = sum(result.stage_timings.values())
            tokens_per_second = (
                result.response_tokens_total / partial_step_seconds
                if partial_step_seconds > 0.0
                else 0.0
            )
            self._log_stage(
                context,
                "sync",
                sync_seconds,
                {
                    "step_duration_seconds": partial_step_seconds,
                    "tokens_per_second": tokens_per_second,
                },
            )
        return result

    def _reference_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility wrapper around the backend-owned reference model."""
        return self.training_backend._reference_logits(input_ids, attention_mask)

    def _assemble_loss(self, **kwargs: Any):
        """Compatibility wrapper around backend loss assembly helpers."""
        return self.training_backend._assemble_loss(**kwargs)

    def _backward_step(self, loss: torch.Tensor) -> Callable[[], None]:
        """Compatibility wrapper around backend backward handling."""
        return self.training_backend._backward_step(loss)

    def _prepare_inputs(
        self,
        batch: TrainingBatch,
        actor: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
        """Compatibility wrapper around backend input preparation."""
        learner_batch = LearnerBatch(
            prompt_token_ids=[
                [int(token_id) for token_id in rollout.prompt_token_ids]
                for rollout in batch.rollouts
            ],
            response_token_ids=[
                [int(token_id) for token_id in rollout.response_token_ids]
                for rollout in batch.rollouts
            ],
            response_token_logprobs=[
                [float(value) for value in rollout.response_token_logprobs]
                for rollout in batch.rollouts
            ],
            advantages=[0.0 for _ in batch.rollouts],
            group_size=getattr(batch, "group_size", 1),
            prompt_count=getattr(batch, "prompt_count", len(batch.rollouts)),
        )
        return self.training_backend._prepare_inputs(learner_batch, actor, device)

    def _compute_rollout_log_probs_from_actor(
        self,
        actor: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> list[list[float]]:
        """Compatibility wrapper around backend rollout-logprob recovery."""
        return self.training_backend._compute_rollout_log_probs_from_actor(
            actor=actor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )

    def _build_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        duration_seconds: float,
        step_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate per-step payloads into an epoch summary."""
        if not step_payloads:
            return {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "loss": 0.0,
                "reward": 0.0,
                "kl_divergence": 0.0,
                "tokens_per_second": 0.0,
                "duration_seconds": duration_seconds,
                "stage_totals": {},
            }

        stage_totals: dict[str, float] = {}
        for payload in step_payloads:
            self._accumulate_totals(stage_totals, payload["stage_timings"])

        return {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": self._mean([payload["loss"] for payload in step_payloads]),
            "reward": self._mean([payload["reward_mean"] for payload in step_payloads]),
            "kl_divergence": self._mean([payload["kl_divergence"] for payload in step_payloads]),
            "tokens_per_second": self._mean([payload["tokens_per_second"] for payload in step_payloads]),
            "duration_seconds": duration_seconds,
            "stage_totals": stage_totals,
            **self._mean_payload_metrics(
                step_payloads,
                ("accuracy_pass_rate", "format_pass_rate", "truncation_rate"),
            ),
        }

    def _log_stage(
        self,
        context: StepContext | None,
        stage: str,
        latency_seconds: float,
        extra: dict[str, Any],
    ) -> None:
        """Emit one step-stage event if a run logger and context are available."""
        if context is None:
            return
        payload = {
            **context.payload(),
            "stage": stage,
            "latency_seconds": latency_seconds,
            **extra,
        }
        if self.run_logger is not None:
            self.run_logger.log_step_stage(payload)
        if self.metrics_sink is not None:
            self.metrics_sink.observe_stage(payload)

    def _install_serving_debug(self, context: StepContext) -> None:
        """Install serving live-rollout debug hooks for the current step when enabled."""
        config = getattr(self.serving_backend, "config", None)
        if not getattr(config, "debug_live_rollout", False):
            return
        if not hasattr(self.serving_backend, "set_live_rollout_debug"):
            return
        self.serving_backend.set_live_rollout_debug(
            callback=self._handle_serving_debug_event,
            context=context.payload(),
        )

    def _clear_serving_debug(self) -> None:
        """Clear serving live-rollout debug hooks after the rollout stage completes."""
        if hasattr(self.serving_backend, "clear_live_rollout_debug"):
            self.serving_backend.clear_live_rollout_debug()

    def _handle_serving_debug_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Dispatch actor-level live-rollout debug events to logging and metrics."""
        if kind == "start":
            if self.run_logger is not None:
                self.run_logger.log_serving_debug_start(payload)
            return
        if kind == "chunk":
            if self.run_logger is not None:
                self.run_logger.log_serving_debug_chunk(payload)
            return
        if kind == "done":
            if self.run_logger is not None:
                self.run_logger.log_serving_debug_done(payload)
            if self.metrics_sink is not None:
                self.metrics_sink.observe_serving_debug(payload)

    def _compute_advantages(
        self,
        rewards: list[RewardOutput],
        prompt_count: int,
        group_size: int,
    ) -> torch.Tensor:
        """Compute GRPO advantages within each prompt group."""
        reward_values = torch.tensor(
            [reward.reward for reward in rewards],
            dtype=torch.float32,
        )
        expected_samples = prompt_count * group_size
        if reward_values.numel() != expected_samples:
            raise ValueError(
                "Reward count must match prompt_count * group_size for GRPO "
                f"(expected {expected_samples}, got {reward_values.numel()})."
            )
        if reward_values.numel() == 0:
            return reward_values
        grouped_rewards = reward_values.view(prompt_count, group_size)
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_stds = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)
        return ((grouped_rewards - group_means) / (group_stds + 1e-8)).reshape(-1)

    def _batch_prompts(self, dataset: Any, batch_size: int | None = None) -> Any:
        """Split dataset into batches."""
        size = batch_size or self.config.batch_size
        for index in range(0, len(dataset), size):
            yield dataset[index:index + size]

    def _prompts_per_step(self) -> int:
        """Return the number of unique prompts consumed by each grouped GRPO step."""
        return self.config.batch_size // self.grpo_config.group_size

    def _measure(self, operation: Callable[[], Any]) -> tuple[Any, float]:
        """Measure one operation with perf_counter."""
        started_at = time.perf_counter()
        result = operation()
        return result, time.perf_counter() - started_at

    def _summary_stats(self, prefix: str, values: list[float]) -> dict[str, float]:
        """Compute mean/std/min/max stats for a list of floats."""
        if not values:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
            }
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            f"{prefix}_mean": float(tensor.mean().item()),
            f"{prefix}_std": float(tensor.std(unbiased=False).item()),
            f"{prefix}_min": float(tensor.min().item()),
            f"{prefix}_max": float(tensor.max().item()),
        }

    def _reward_rate_stats(self, rewards: list[RewardOutput]) -> dict[str, float]:
        """Aggregate common boolean reward metadata into per-step rates."""
        mappings = (
            ("accuracy_pass", "accuracy_pass_rate"),
            ("format_pass", "format_pass_rate"),
            ("truncated", "truncation_rate"),
        )
        stats: dict[str, float] = {}
        for source_key, target_key in mappings:
            values = [
                float(bool(reward.metadata[source_key]))
                for reward in rewards
                if source_key in reward.metadata
            ]
            if values:
                stats[target_key] = self._mean(values)
        return stats

    def _mean_payload_metrics(
        self,
        payloads: list[dict[str, Any]],
        keys: tuple[str, ...],
    ) -> dict[str, float]:
        """Average selected float payload metrics across a list of step payloads."""
        result: dict[str, float] = {}
        for key in keys:
            values = [float(payload[key]) for payload in payloads if key in payload]
            if values:
                result[key] = self._mean(values)
        return result

    def _accumulate_totals(
        self,
        target: dict[str, float],
        update: dict[str, float],
    ) -> None:
        for key, value in update.items():
            target[key] = target.get(key, 0.0) + float(value)

    def _mean(self, values: list[float] | list[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _truncate(self, text: str, *, limit: int = 240) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        self.training_backend.save_checkpoint(
            path,
            controller_state={
                "epoch": self.current_epoch,
                "total_steps": self.total_steps,
            },
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        controller_state = self.training_backend.load_checkpoint(path)
        self.current_epoch = int(controller_state.get("epoch", 0))
        self.total_steps = int(controller_state.get("total_steps", 0))
        self.training_backend.sync_weights_to(self.serving_backend)
