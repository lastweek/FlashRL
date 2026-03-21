"""GRPO (Group Relative Policy Optimization) trainer."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable

import torch

from flashrl.framework.config import GrpoConfig, TrainerConfig
from flashrl.framework.data_models import (
    LearnerBatch,
    Prompt,
    RewardOutput,
    TrainingBatch,
    WeightVersionInfo,
)
from flashrl.framework.distributed.learner_client import LearnerClient
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    LoadCheckpointRequest,
    OptimizeStepRequest,
    RewardBatchRequest,
    RolloutBatchRequest,
    SaveCheckpointRequest,
)
from flashrl.framework.distributed.reward_client import RewardClient
from flashrl.framework.distributed.rollout_client import RolloutClient
from flashrl.framework.distributed.serving_client import ServingClient
from flashrl.framework.memory import capture_memory_snapshot, summarize_memory_window
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
from flashrl.framework.reward import RewardService
from flashrl.framework.rollout import RolloutService
from flashrl.framework.serving import ServingService
from flashrl.framework.training import LearnerService
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.base import BaseRolloutGenerator
from flashrl.framework.rollout_metrics import count_llm_call_rounds, count_tool_calls
from flashrl.framework.training import OptimizationResult
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
        actor_backend: "ActorTrainingBackend | None",
        reference_backend: "ReferenceTrainingBackend | None",
        serving_backend: "ServingBackend | None",
        reward_fn: UserDefinedReward | None,
        rollout_generator: BaseRolloutGenerator | None,
        run_logger: "RunLogger | None" = None,
        metrics_sink: "MetricsSink | None" = None,
        on_step_complete: Callable[[dict[str, Any]], None] | None = None,
        rollout: RolloutService | RolloutClient | None = None,
        reward: RewardService | RewardClient | None = None,
        learner: LearnerService | LearnerClient | None = None,
        serving: ServingService | ServingClient | None = None,
        reference_configured: bool | None = None,
    ) -> None:
        """Initialize GRPO trainer."""
        self.config = config
        self.run_logger = run_logger
        self.metrics_sink = metrics_sink
        self.current_epoch = 0
        self.total_steps = 0
        self._active_step_context: StepContext | None = None
        self.actor_backend = actor_backend
        self.reference_backend = reference_backend
        self.serving_backend = serving_backend
        self.grpo_config = grpo_config
        self.reward_fn = reward_fn
        self.rollout_generator = rollout_generator
        self.on_step_complete = on_step_complete
        self.reference_configured = (
            bool(reference_backend is not None)
            if reference_configured is None
            else bool(reference_configured)
        )
        self.rollout = rollout or self._build_default_rollout_service(
            rollout_generator
        )
        self.reward = reward or self._build_default_reward_service(reward_fn)
        self.learner = learner or self._build_default_learner_service(
            actor_backend=actor_backend,
            reference_backend=reference_backend,
            serving_backend=serving_backend,
        )
        self.serving = serving or self._build_default_serving_service(
            serving_backend
        )

    @property
    def training_backend(self) -> "ActorTrainingBackend":
        """Expose the actor backend under the traditional training-backend name."""
        if self.actor_backend is None:
            raise RuntimeError("training_backend is unavailable without a local actor backend.")
        return self.actor_backend

    def attach_run_logger(self, run_logger: "RunLogger | None") -> None:
        """Attach or clear the current run-scoped logger."""
        self.run_logger = run_logger

    def reset_state(self) -> None:
        """Reset per-run trainer state."""
        self.current_epoch = 0
        self.total_steps = 0
        self._active_step_context = None
        for dependency in (
            self.rollout,
            self.reward,
            self.learner,
            self.serving,
        ):
            reset = getattr(dependency, "reset_state", None)
            if callable(reset):
                reset()

    @property
    def active_step_context(self) -> StepContext | None:
        """Return the currently running step context when a step is in progress."""
        return self._active_step_context

    def _build_default_rollout_service(
        self,
        rollout_generator: BaseRolloutGenerator | None,
    ) -> RolloutService:
        if rollout_generator is None:
            raise ValueError("rollout_generator is required when rollout is not provided.")
        return RolloutService(rollout_generator)

    def _build_default_reward_service(
        self,
        reward_fn: UserDefinedReward | None,
    ) -> RewardService:
        if reward_fn is None:
            raise ValueError("reward_fn is required when reward is not provided.")
        return RewardService(reward_fn)

    def _build_default_learner_service(
        self,
        *,
        actor_backend: "ActorTrainingBackend | None",
        reference_backend: "ReferenceTrainingBackend | None",
        serving_backend: "ServingBackend | None",
    ) -> LearnerService:
        if actor_backend is None:
            raise ValueError("actor_backend is required when learner is not provided.")
        return LearnerService(
            actor_backend,
            reference_backend,
            grpo_config=self.grpo_config,
            serving_backend=serving_backend,
            synchronize_serving=True,
        )

    def _build_default_serving_service(
        self,
        serving_backend: "ServingBackend | None",
    ) -> ServingService:
        if serving_backend is None:
            raise ValueError("serving_backend is required when serving is not provided.")
        return ServingService(serving_backend)

    def train(self, dataset: Any) -> None:
        """Train on the given dataset."""
        prompts_per_step = prompt_batch_size(self.config.batch_size, self.grpo_config.group_size)
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
                self._active_step_context = context
                step_payload = self._run_logged_step(prompts, context)
                epoch_step_payloads.append(step_payload)
                self.total_steps = context.step
                self._active_step_context = None
                if self.on_step_complete is not None:
                    self.on_step_complete(
                        {
                            "epoch": epoch_number,
                            "epoch_index": self.current_epoch,
                            "step": self.total_steps,
                            "batch_index": batch_index,
                            "batches_in_epoch": batches_in_epoch,
                            "step_payload": step_payload,
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
                # Clear step payloads and dataset to free memory before next epoch
                epoch_step_payloads.clear()
                del epoch_dataset
        self._active_step_context = None

    def _run_logged_step(
        self,
        prompts: list[Prompt],
        context: StepContext,
    ) -> dict[str, Any]:
        """Run one full training step and emit detailed stage logs."""
        step_started_at = current_time()
        step_memory_start = capture_memory_snapshot(self._actor_device_for_memory())
        if self.run_logger is not None:
            self.run_logger.log_step_start(context.payload())

        rollout_started_at = current_time()
        self._install_serving_debug(context)
        try:
            rollout_response = self.rollout.rollout_batch(
                RolloutBatchRequest(
                    step_id=context.step,
                    prompts=prompts,
                    group_size=context.group_size,
                )
            )
        finally:
            self._clear_serving_debug()
        rollout_seconds = elapsed_seconds(rollout_started_at)
        rollouts = list(rollout_response.rollouts)
        prompt_indices = list(rollout_response.prompt_indices)
        candidate_indices = list(rollout_response.candidate_indices)
        if not prompt_indices and rollouts:
            prompt_indices = [index // context.group_size for index in range(len(rollouts))]
        if not candidate_indices and rollouts:
            candidate_indices = [index % context.group_size for index in range(len(rollouts))]
        grouped_prompts = [
            prompts[prompt_index]
            for prompt_index in prompt_indices
        ] if prompt_indices else []
        grouped_prompts_count = len(grouped_prompts)
        prompt_lengths = []
        response_lengths = []
        for rollout in rollouts:
            prompt_lengths.append(len(rollout.prompt_token_ids))
            response_lengths.append(len(rollout.response_token_ids))

        # Compute additional rollout metrics
        llm_call_rounds = count_llm_call_rounds(rollouts)
        tool_calls_total = count_tool_calls(rollouts)

        self._log_stage(
            context,
            StageResult(
                name="rollout",
                seconds=rollout_seconds,
                metrics={
                    "sample_count": grouped_prompts_count,
                    "prompt_tokens_mean": mean(prompt_lengths),
                    "prompt_tokens_max": max(prompt_lengths, default=0),
                    "response_tokens_mean": mean(response_lengths),
                    "response_tokens_max": max(response_lengths, default=0),
                    "llm_call_rounds": llm_call_rounds,
                    "tool_calls_total": tool_calls_total,
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
            metadata=self._build_batch_metadata(rollouts),
        )
        rollout_weight_version = dict(batch.metadata.get("weight_version", {}))

        advantages, advantage_seconds = timed_call(
            lambda: compute_advantages(
                batch.rewards,
                prompt_count=batch.prompt_count,
                group_size=batch.group_size,
            )
        )
        advantage_values = advantages.cpu().numpy().tolist()
        self._log_stage(
            context,
            StageResult(
                name="advantage",
                seconds=advantage_seconds,
                metrics=summary_stats("advantage", advantage_values),
            ),
        )

        learner_batch = self._build_learner_batch(batch, advantages)
        # Explicitly delete the advantages tensor to free memory early
        del advantages
        result = self._optimize_batch(
            learner_batch,
            context,
            rollout_weight_version=rollout_response.weight_version,
        )
        all_stage_timings = {
            "rollout": rollout_seconds,
            "reward": reward_seconds,
            "advantage": advantage_seconds,
            **stage_timings(result.stages),
        }
        all_stage_timings.setdefault("reference_forward", 0.0)
        step_duration_seconds = elapsed_seconds(step_started_at)
        step_memory_end = capture_memory_snapshot(self._actor_device_for_memory())
        response_tokens_total = result.response_tokens_total
        tokens_per_second = (
            response_tokens_total / step_duration_seconds if step_duration_seconds > 0.0 else 0.0
        )

        ref_active = result.reference_active
        visible_stage_order = [
            stage_name
            for stage_name in STAGE_ORDER
            if stage_name in all_stage_timings
            and (stage_name != "reference_forward" or ref_active)
        ]
        reward_summary = reward_stage.metrics
        payload = {
            **context.payload(),
            "loss": result.loss,
            "policy_loss": result.policy_loss,
            "kl_divergence": result.kl_divergence,
            "learning_rate": result.learning_rate,
            "reward_mean": mean(reward_values),
            **{
                key: value
                for key, value in reward_summary.items()
                if key.endswith("_rate")
            },
            "response_tokens_total": response_tokens_total,
            "tokens_per_second": tokens_per_second,
            "step_duration_seconds": step_duration_seconds,
            "learner_total_seconds": result.learner_total_seconds,
            "learner_unaccounted_seconds": result.learner_unaccounted_seconds,
            "memory_summary": summarize_memory_window(
                step_memory_start,
                *self._stage_memory_snapshots(result.stages),
                step_memory_end,
                start=step_memory_start,
                end=step_memory_end,
            ),
            "stage_timings": all_stage_timings,
            "stage_order": visible_stage_order,
            "reference_configured": self.reference_configured,
            "reference_active": ref_active,
            "dominant_stage": dominant_stage_name(result.stages),
            "candidate_count": grouped_prompts_count,
            "sample_count": grouped_prompts_count,
        }
        if rollout_weight_version:
            payload["rollout_weight_version"] = rollout_weight_version

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
            rewards = self.reward.reward_batch(
                RewardBatchRequest(
                    step_id=context.step,
                    rollouts=list(rollouts),
                )
            ).rewards
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
        prompt_token_ids = []
        response_token_ids = []
        response_token_logprobs = []
        expanded_advantages: list[float] = []
        expanded_prompt_indices: list[int] = []
        expanded_candidate_indices: list[int] = []
        advantage_values = advantages.cpu().numpy().tolist()
        for rollout, advantage, prompt_index, candidate_index in zip(
            batch.rollouts,
            advantage_values,
            batch.prompt_indices,
            batch.candidate_indices,
            strict=True,
        ):
            turns = getattr(rollout, "assistant_turns", [])
            if turns:
                for turn in turns:
                    prompt_token_ids.append([int(token_id) for token_id in turn.prompt_token_ids])
                    response_token_ids.append([int(token_id) for token_id in turn.response_token_ids])
                    response_token_logprobs.append(
                        [float(value) for value in turn.response_token_logprobs]
                    )
                    expanded_advantages.append(float(advantage))
                    expanded_prompt_indices.append(int(prompt_index))
                    expanded_candidate_indices.append(int(candidate_index))
                continue

            prompt_token_ids.append([int(token_id) for token_id in rollout.prompt_token_ids])
            response_token_ids.append([int(token_id) for token_id in rollout.response_token_ids])
            response_token_logprobs.append([float(value) for value in rollout.response_token_logprobs])
            expanded_advantages.append(float(advantage))
            expanded_prompt_indices.append(int(prompt_index))
            expanded_candidate_indices.append(int(candidate_index))

        return LearnerBatch(
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            response_token_logprobs=response_token_logprobs,
            advantages=expanded_advantages,
            group_size=batch.group_size,
            prompt_count=batch.prompt_count,
            prompt_indices=expanded_prompt_indices,
            candidate_indices=expanded_candidate_indices,
            metadata=dict(batch.metadata),
        )

    def _optimize_batch(
        self,
        learner_batch: LearnerBatch,
        context: StepContext | None,
        *,
        rollout_weight_version: WeightVersionInfo | None,
    ) -> OptimizationResult:
        """Delegate learner-side tensor work to the training backend."""
        learner_started_at = current_time()
        try:
            optimize_response = self.learner.optimize_step(
                OptimizeStepRequest(
                    step_id=(context.step if context is not None else None),
                    epoch=(context.epoch if context is not None else 0),
                    learner_batch=learner_batch,
                    rollout_weight_version=rollout_weight_version,
                )
            )
        except Exception as exc:
            setattr(exc, "learner_total_seconds", elapsed_seconds(learner_started_at))
            raise
        learner_total_seconds = elapsed_seconds(learner_started_at)
        learner_stage_seconds = sum(float(stage.seconds) for stage in optimize_response.stages)
        result = OptimizationResult(
            loss=optimize_response.loss,
            policy_loss=optimize_response.policy_loss,
            kl_divergence=optimize_response.kl_divergence,
            learning_rate=optimize_response.learning_rate,
            response_tokens_total=optimize_response.response_tokens_total,
            reference_active=optimize_response.reference_active,
            stages=[
                StageResult(
                    name=stage.name,
                    seconds=float(stage.seconds),
                    metrics=dict(stage.metrics),
                )
                for stage in optimize_response.stages
            ],
            learner_total_seconds=learner_total_seconds,
            learner_unaccounted_seconds=max(0.0, learner_total_seconds - learner_stage_seconds),
        )
        for stage in result.stages:
            if stage.name == "reference_forward" and not result.reference_active:
                continue
            self._log_stage(context, stage)

        activation_response, sync_seconds = timed_call(
            lambda: self.serving.activate_weight_version(
                ActivateWeightVersionRequest(
                    step_id=(context.step if context is not None else None),
                    weight_version=optimize_response.weight_version,
                )
            )
        )
        active_weight_version = activation_response.active_weight_version
        sync_stage = StageResult(
            name="sync",
            seconds=sync_seconds,
            metrics={
                "weight_version_id": (
                    active_weight_version.version_id
                    if active_weight_version is not None
                    else optimize_response.weight_version.version_id
                ),
            },
        )
        result.stages.append(sync_stage)
        if context is not None:
            partial_step_seconds = sum(stage.seconds for stage in result.stages)
            sync_stage.metrics = {
                "weight_version_id": (
                    active_weight_version.version_id
                    if active_weight_version is not None
                    else optimize_response.weight_version.version_id
                ),
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

    def _actor_device_for_memory(self) -> Any | None:
        """Return the best local device handle for step-level memory snapshots."""
        if self.actor_backend is not None:
            return self.actor_backend.device
        if self.serving_backend is not None:
            return getattr(self.serving_backend, "device", None)
        return None

    def _stage_memory_snapshots(self, stages: list[StageResult]) -> list[dict[str, Any]]:
        """Collect before/after memory snapshots from measured learner stages."""
        snapshots: list[dict[str, Any]] = []
        for stage in stages:
            memory_payload = (
                stage.metrics.get("memory") if isinstance(stage.metrics.get("memory"), dict) else None
            )
            if not isinstance(memory_payload, dict):
                continue
            before = memory_payload.get("before")
            after = memory_payload.get("after")
            if isinstance(before, dict):
                snapshots.append(before)
            if isinstance(after, dict):
                snapshots.append(after)
        return snapshots

    def _build_batch_metadata(self, rollouts: list[Any]) -> dict[str, Any]:
        """Capture batch-wide rollout provenance and reject mixed serving versions."""
        normalized_versions = []
        for rollout in rollouts:
            payload = _weight_version_payload(rollout)
            if payload:
                normalized_versions.append(payload)

        if not normalized_versions:
            return {}

        canonical = normalized_versions[0]
        if any(payload != canonical for payload in normalized_versions[1:]):
            raise RuntimeError("Rollout batch mixed multiple serving weight versions.")
        return {
            "weight_version": canonical,
        }

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
        if self.serving_backend is None:
            return
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
        if self.serving_backend is None:
            return
        if hasattr(self.serving_backend, "clear_live_rollout_debug"):
            self.serving_backend.clear_live_rollout_debug()

    def _handle_serving_debug_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Dispatch actor-level live-rollout debug events to logging and metrics."""
        event = RuntimeEvent(kind=f"serving_debug_{kind}", payload=payload)
        if kind == "done":
            observe_event_pair(self.run_logger, self.metrics_sink, event)
            return
        observe_event(self.run_logger, event)

    def save_checkpoint(
        self,
        path: str,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save model checkpoint."""
        self.learner.save_checkpoint(
            SaveCheckpointRequest(
                path=path,
                controller_state={
                    "epoch": self.current_epoch,
                    "total_steps": self.total_steps,
                },
                checkpoint_metadata=dict(checkpoint_metadata or {}),
            )
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        controller_state, _ = self.load_checkpoint_with_metadata(path)
        self.current_epoch = int(controller_state.get("epoch", 0))
        self.total_steps = int(controller_state.get("total_steps", 0))

    def load_checkpoint_with_metadata(
        self,
        path: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Load checkpoint state and expose any attached checkpoint metadata."""
        response = self.learner.load_checkpoint(LoadCheckpointRequest(path=path))
        controller_state = dict(response.controller_state)
        checkpoint_metadata = response.checkpoint_metadata
        self.current_epoch = int(controller_state.get("epoch", 0))
        self.total_steps = int(controller_state.get("total_steps", 0))
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


def _weight_version_payload(rollout: Any) -> dict[str, Any]:
    metadata = getattr(rollout, "metadata", {})
    if not isinstance(metadata, dict):
        return {}
    weight_version = metadata.get("weight_version")
    if isinstance(weight_version, WeightVersionInfo):
        return weight_version.model_dump()
    if isinstance(weight_version, dict):
        return dict(weight_version)
    return {}


def math_ceil_div(size: int, divisor: int) -> int:
    """Return ``ceil(size / divisor)`` without importing ``math`` into the main flow."""
    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor}")
    return (size + divisor - 1) // divisor
