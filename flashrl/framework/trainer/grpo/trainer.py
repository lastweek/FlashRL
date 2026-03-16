"""GRPO (Group Relative Policy Optimization) trainer."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable

import torch

from flashrl.framework.config import GrpoConfig, TrainerConfig
from flashrl.framework.data_models import LearnerBatch, Prompt, RewardOutput, TrainingBatch
from flashrl.framework.observability import (
    RuntimeEvent,
    StageResult,
    dominant_stage_name,
    elapsed_seconds,
    observe_event,
    observe_event_pair,
    stage_timings,
    timed_call,
)
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.training import OptimizationResult
from flashrl.framework.training.base import assemble_loss
from flashrl.framework.training.optimization import optimize_grpo_batch
from flashrl.framework.trainer.grpo.grpo_helpers import (
    STAGE_ORDER,
    StepContext,
    accumulate_totals,
    batch_items,
    compute_advantages,
    mean_payload_metrics,
    prompt_batch_size,
    reward_rate_stats,
)
from flashrl.framework.utils import mean, summary_stats, truncate_preview

if TYPE_CHECKING:
    from flashrl.framework.metrics import MetricsSink
    from flashrl.framework.run_logger import RunLogger
    from flashrl.framework.serving import ServingBackend
    from flashrl.framework.training import ActorTrainingBackend, ReferenceTrainingBackend, TrainingBackend


class GRPOTrainer:
    """GRPO trainer implementation with detailed step logging."""

    def __init__(
        self,
        config: TrainerConfig,
        grpo_config: GrpoConfig,
        actor_backend: "ActorTrainingBackend",
        reference_backend: "ReferenceTrainingBackend | None",
        serving_backend: "ServingBackend",
        reward_fn: UserDefinedReward,
        rollout_generator: UserDefinedRollout,
        run_logger: "RunLogger | None" = None,
        metrics_sink: "MetricsSink | None" = None,
        on_step_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize GRPO trainer."""
        self.config = config
        self.run_logger = run_logger
        self.metrics_sink = metrics_sink
        self.current_epoch = 0
        self.total_steps = 0
        self.actor_backend = actor_backend
        self.reference_backend = reference_backend
        self.serving_backend = serving_backend
        self.grpo_config = grpo_config
        self.reward_fn = reward_fn
        self.rollout_generator = rollout_generator
        self.on_step_complete = on_step_complete

    @property
    def training_backend(self) -> "ActorTrainingBackend":
        """Expose the actor backend under the traditional training-backend name."""
        return self.actor_backend

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
        batches_in_epoch = math_ceil_div(len(dataset), prompts_per_step) if len(dataset) else 0

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_number = epoch + 1
            epoch_started_at = current_time()
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
                batch_items(epoch_dataset, prompts_per_step),
                start=1,
            ):
                next_step = self.total_steps + 1
                candidate_count = len(prompts) * self.grpo_config.group_size
                context = StepContext(
                    step=next_step,
                    epoch=epoch_number,
                    total_epochs=self.config.max_epochs,
                    batch_index=batch_index,
                    batches_in_epoch=batches_in_epoch,
                    batch_size=candidate_count,
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
                if self.on_step_complete is not None:
                    self.on_step_complete(
                        {
                            "epoch": epoch_number,
                            "epoch_index": self.current_epoch,
                            "step": self.total_steps,
                            "batch_index": batch_index,
                            "batches_in_epoch": batches_in_epoch,
                        }
                    )

            if self.run_logger is not None:
                epoch_summary = self._build_epoch_summary(
                    epoch=epoch_number,
                    total_epochs=self.config.max_epochs,
                    duration_seconds=elapsed_seconds(epoch_started_at),
                    step_payloads=epoch_step_payloads,
                )
                self.run_logger.log_epoch_summary(epoch_summary)

    def _run_logged_step(
        self,
        prompts: list[Prompt],
        context: StepContext,
    ) -> dict[str, Any]:
        """Run one full training step and emit detailed stage logs."""
        step_started_at = current_time()
        if self.run_logger is not None:
            self.run_logger.log_step_start(context.payload())

        rollout_started_at = current_time()
        self._install_serving_debug(context)
        try:
            grouped_prompts, rollouts, prompt_indices, candidate_indices = (
                self.rollout_generator.generate_grouped(prompts, context.group_size)
            )
        finally:
            self._clear_serving_debug()
        rollout_seconds = elapsed_seconds(rollout_started_at)
        prompt_lengths = [len(rollout.prompt_token_ids) for rollout in rollouts]
        response_lengths = [len(rollout.response_token_ids) for rollout in rollouts]
        self._log_stage(
            context,
            StageResult(
                name="rollout",
                seconds=rollout_seconds,
                metrics={
                    "sample_count": len(grouped_prompts),
                    "prompt_tokens_mean": mean(prompt_lengths),
                    "prompt_tokens_max": max(prompt_lengths, default=0),
                    "response_tokens_mean": mean(response_lengths),
                    "response_tokens_max": max(response_lengths, default=0),
                },
            ),
        )

        rewards, reward_seconds = self._compute_rewards(context, grouped_prompts, rollouts)
        reward_values = [reward.reward for reward in rewards]
        reward_stage = StageResult(
            name="reward",
            seconds=reward_seconds,
            metrics={
                **summary_stats("reward", reward_values),
                **reward_rate_stats(rewards),
                "reward_per_item_mean_seconds": (
                    reward_seconds / len(rewards) if rewards else 0.0
                ),
            },
        )
        self._log_stage(context, reward_stage)
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

        advantages, advantage_seconds = timed_call(
            lambda: self._compute_advantages(
                batch.rewards,
                prompt_count=batch.prompt_count,
                group_size=batch.group_size,
            )
        )
        advantage_values = [float(value) for value in advantages.tolist()]
        self._log_stage(
            context,
            StageResult(
                name="advantage",
                seconds=advantage_seconds,
                metrics=summary_stats("advantage", advantage_values),
            ),
        )

        learner_batch = self._build_learner_batch(batch, advantages)
        result = self._optimize_batch(learner_batch, context)
        all_stage_timings = {
            "rollout": rollout_seconds,
            "reward": reward_seconds,
            "advantage": advantage_seconds,
            **stage_timings(result.stages),
        }
        all_stage_timings.setdefault("reference_forward", 0.0)
        step_duration_seconds = elapsed_seconds(step_started_at)
        response_tokens_total = result.response_tokens_total
        tokens_per_second = (
            response_tokens_total / step_duration_seconds if step_duration_seconds > 0.0 else 0.0
        )

        visible_stage_order = [
            stage_name
            for stage_name in STAGE_ORDER
            if stage_name in all_stage_timings
            and (stage_name != "reference_forward" or result.reference_active)
        ]
        reward_summary = reward_stage.metrics
        payload = {
            **context.payload(),
            "loss": result.loss,
            "policy_loss": result.policy_loss,
            "kl_divergence": result.kl_divergence,
            "learning_rate": result.learning_rate,
            "reward_mean": mean([float(value) for value in reward_values]),
            **{
                key: value
                for key, value in reward_summary.items()
                if key.endswith("_rate")
            },
            "response_tokens_total": response_tokens_total,
            "tokens_per_second": tokens_per_second,
            "step_duration_seconds": step_duration_seconds,
            "stage_timings": all_stage_timings,
            "stage_order": visible_stage_order,
            "reference_configured": self.reference_backend is not None,
            "reference_active": result.reference_active,
            "dominant_stage": dominant_stage_name(result.stages),
            "candidate_count": len(grouped_prompts),
            "sample_count": len(grouped_prompts),
        }

        done_event = RuntimeEvent(kind="step_done", payload=payload)
        observe_event_pair(self.run_logger, self.metrics_sink, done_event)
        if self.metrics_sink is not None:
            self.metrics_sink.push()

        if self.run_logger is not None:
            self.run_logger.log_sample_preview(
                step=context.step,
                prompt=grouped_prompts[0].text if grouped_prompts else "",
                response=rollouts[0].text if rollouts else "",
                reward=reward_values[0] if reward_values else 0.0,
            )

        return payload

    def _compute_rewards(
        self,
        context: StepContext,
        prompts: list[Prompt],
        rollouts: list[Any],
    ) -> tuple[list[RewardOutput], float]:
        """Compute rewards and attach prompt/response context on failure."""
        started_at = current_time()
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
                        "prompt_preview": truncate_preview(prompts[0].text if prompts else ""),
                        "response_preview": truncate_preview(
                            rollouts[0].text if rollouts else ""
                        ),
                    },
                )
            raise
        return rewards, elapsed_seconds(started_at)

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
        result = optimize_grpo_batch(
            actor_backend=self.actor_backend,
            reference_backend=self.reference_backend,
            grpo_config=self.grpo_config,
            learner_batch=learner_batch,
        )
        for stage in result.stages:
            if stage.name == "reference_forward" and not result.reference_active:
                continue
            self._log_stage(context, stage)

        _, sync_seconds = timed_call(
            lambda: self.actor_backend.sync_weights_to(self.serving_backend)
        )
        sync_stage = StageResult(
            name="sync",
            seconds=sync_seconds,
            metrics={},
        )
        result.stages.append(sync_stage)
        if context is not None:
            partial_step_seconds = sum(stage.seconds for stage in result.stages)
            sync_stage.metrics = {
                "step_duration_seconds": partial_step_seconds,
                "tokens_per_second": (
                    result.response_tokens_total / partial_step_seconds
                    if partial_step_seconds > 0.0
                    else 0.0
                ),
            }
            self._log_stage(context, sync_stage)
        result.refresh_stage_views()
        return result

    def _prepare_inputs(
        self,
        batch: TrainingBatch | Any,
        actor: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
        """Compatibility wrapper around actor-backend input preparation."""
        del actor, device
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
        return self.actor_backend.prepare_inputs(learner_batch)

    def _assemble_loss(self, **kwargs: Any):
        """Compatibility wrapper around the shared GRPO loss helper."""
        return assemble_loss(**kwargs)

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
            accumulate_totals(stage_totals, payload["stage_timings"])

        return {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": mean([payload["loss"] for payload in step_payloads]),
            "reward": mean([payload["reward_mean"] for payload in step_payloads]),
            "kl_divergence": mean([payload["kl_divergence"] for payload in step_payloads]),
            "tokens_per_second": mean(
                [payload["tokens_per_second"] for payload in step_payloads]
            ),
            "duration_seconds": duration_seconds,
            "stage_totals": stage_totals,
            **mean_payload_metrics(
                step_payloads,
                ("accuracy_pass_rate", "format_pass_rate", "truncation_rate"),
            ),
        }

    def _log_stage(
        self,
        context: StepContext | None,
        stage: StageResult,
    ) -> None:
        """Emit one step-stage event if a run logger and context are available."""
        if context is None:
            return
        event = RuntimeEvent(
            kind="step_stage",
            payload={
                **context.payload(),
                **stage.to_payload(),
            },
        )
        observe_event_pair(self.run_logger, self.metrics_sink, event)

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
        event = RuntimeEvent(kind=f"serving_debug_{kind}", payload=payload)
        if kind == "done":
            observe_event_pair(self.run_logger, self.metrics_sink, event)
            return
        observe_event(self.run_logger, event)

    def _compute_advantages(
        self,
        rewards: list[RewardOutput],
        prompt_count: int,
        group_size: int,
    ) -> torch.Tensor:
        """Compute GRPO advantages within each prompt group."""
        return compute_advantages(
            rewards,
            prompt_count=prompt_count,
            group_size=group_size,
        )

    def _batch_prompts(self, dataset: list[Any], batch_size: int | None = None):
        """Split a dataset into prompt batches."""
        size = batch_size or self.config.batch_size
        return batch_items(dataset, size)

    def _prompts_per_step(self) -> int:
        """Return the number of unique prompts consumed by each grouped GRPO step."""
        return prompt_batch_size(self.config.batch_size, self.grpo_config.group_size)

    def _summary_stats(self, prefix: str, values: list[float]) -> dict[str, float]:
        """Compatibility wrapper around shared summary stats."""
        return summary_stats(prefix, values)

    def _reward_rate_stats(self, rewards: list[RewardOutput]) -> dict[str, float]:
        """Compatibility wrapper around shared reward rate stats."""
        return reward_rate_stats(rewards)

    def _mean_payload_metrics(
        self,
        payloads: list[dict[str, Any]],
        keys: tuple[str, ...],
    ) -> dict[str, float]:
        """Compatibility wrapper around shared payload averaging."""
        return mean_payload_metrics(payloads, keys)

    def _accumulate_totals(self, target: dict[str, float], update: dict[str, float]) -> None:
        accumulate_totals(target, update)

    def _mean(self, values: list[float] | list[int]) -> float:
        return mean(values)

    def _truncate(self, text: str, *, limit: int = 240) -> str:
        return truncate_preview(text, limit=limit)

    def save_checkpoint(
        self,
        path: str,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save model checkpoint."""
        payload = {
            "controller_state": {
                "epoch": self.current_epoch,
                "total_steps": self.total_steps,
            },
            "backend_states": {
                "actor": self.actor_backend.export_state(),
                "reference": (
                    self.reference_backend.export_state() if self.reference_backend is not None else None
                ),
            },
            "checkpoint_metadata": dict(checkpoint_metadata or {}),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        controller_state, _ = self.load_checkpoint_with_metadata(path)
        self.current_epoch = int(controller_state.get("epoch", 0))
        self.total_steps = int(controller_state.get("total_steps", 0))
        self.actor_backend.sync_weights_to(self.serving_backend)

    def load_checkpoint_with_metadata(
        self,
        path: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Load checkpoint state and expose any attached checkpoint metadata."""
        checkpoint = torch.load(path, weights_only=False)
        backend_states = checkpoint.get("backend_states")
        if not isinstance(backend_states, dict):
            raise ValueError(
                "Checkpoint is incompatible with the role-based runtime. Recreate it with the current FlashRL version."
            )

        actor_state = backend_states.get("actor")
        if not isinstance(actor_state, dict):
            raise ValueError("Checkpoint is missing backend_states.actor.")
        self.actor_backend.load_state(actor_state)

        reference_state = backend_states.get("reference")
        if self.reference_backend is None:
            if reference_state is not None:
                raise ValueError(
                    "Checkpoint contains reference backend state, but this trainer was created without a reference backend."
                )
        else:
            if not isinstance(reference_state, dict):
                raise ValueError(
                    "Checkpoint is missing backend_states.reference for a trainer with a reference backend."
                )
            self.reference_backend.load_state(reference_state)

        controller_state = dict(checkpoint.get("controller_state", {}))
        checkpoint_metadata = checkpoint.get("checkpoint_metadata")
        if not isinstance(checkpoint_metadata, dict):
            checkpoint_metadata = None
        self.current_epoch = int(controller_state.get("epoch", 0))
        self.total_steps = int(controller_state.get("total_steps", 0))
        self.actor_backend.sync_weights_to(self.serving_backend)
        return controller_state, checkpoint_metadata

    def read_checkpoint_metadata(self, path: str) -> dict[str, Any] | None:
        """Return checkpoint metadata without mutating trainer state."""
        checkpoint = torch.load(path, weights_only=False)
        checkpoint_metadata = checkpoint.get("checkpoint_metadata")
        if isinstance(checkpoint_metadata, dict):
            return dict(checkpoint_metadata)
        return None


def current_time() -> float:
    """Wrapper around perf-counter access for readability."""
    import time

    return time.perf_counter()


def math_ceil_div(size: int, divisor: int) -> int:
    """Return ``ceil(size / divisor)`` without importing ``math`` into the main flow."""
    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor}")
    return (size + divisor - 1) // divisor