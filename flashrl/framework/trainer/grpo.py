"""GRPO (Group Relative Policy Optimization) trainer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F

from flashrl.framework.config import GrpoConfig, TrainerConfig
from flashrl.framework.data_models import Prompt, RewardOutput, TrainingBatch
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout

if TYPE_CHECKING:
    from flashrl.framework.backends.serving import ServingBackend
    from flashrl.framework.backends.training import TrainingBackend
    from flashrl.framework.metrics import PrometheusMetricsSink
    from flashrl.framework.run_logger import RunLogger


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
        }


@dataclass
class OptimizationResult:
    """Result of the optimize portion of a GRPO step."""

    loss: float
    policy_loss: float
    kl_divergence: float
    stage_timings: dict[str, float]
    response_tokens_total: int
    reference_active: bool


class GRPOTrainer:
    """GRPO trainer implementation with detailed step logging."""

    def __init__(
        self,
        config: TrainerConfig,
        grpo_config: GrpoConfig,
        training_backend: "TrainingBackend",
        serving_backend: "ServingBackend",
        reference: ReferenceModel | None,
        reward_fn: UserDefinedReward,
        rollout_generator: UserDefinedRollout,
        run_logger: "RunLogger | None" = None,
        metrics_sink: "PrometheusMetricsSink | None" = None,
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
        prompts_per_step = self._prompts_per_step()
        batches_in_epoch = math.ceil(len(dataset) / prompts_per_step) if len(dataset) else 0

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_number = epoch + 1
            epoch_started_at = time.perf_counter()
            epoch_step_payloads: list[dict[str, Any]] = []

            if self.run_logger is not None:
                self.run_logger.log_epoch_start(
                    epoch_number,
                    self.config.max_epochs,
                    batches_in_epoch,
                )

            for batch_index, prompts in enumerate(
                self._batch_prompts(dataset, prompts_per_step),
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
        reward_per_item_mean_seconds = reward_seconds / len(rewards) if rewards else 0.0
        self._log_stage(
            context,
            "reward",
            reward_seconds,
            {
                **reward_stats,
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

        result = self._optimize_batch(batch, advantages, context)
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
        reference_enabled = self.reference is not None

        payload = {
            **context.payload(),
            "loss": result.loss,
            "policy_loss": result.policy_loss,
            "kl_divergence": result.kl_divergence,
            "reward_mean": reward_mean,
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
            self.metrics_sink.observe_step(
                loss=result.loss,
                reward_mean=reward_mean,
                kl_mean=result.kl_divergence,
                step_duration_seconds=step_duration_seconds,
            )
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

    def _optimize_batch(
        self,
        batch: TrainingBatch,
        advantages: torch.Tensor,
        context: StepContext | None,
    ) -> OptimizationResult:
        """Run input preparation, forward, loss, backward, step, and sync."""
        actor = self.training_backend.actor
        device = actor.device
        (
            input_ids,
            attention_mask,
            prompt_lengths,
            full_lengths,
            rollout_response_log_probs,
        ), full_seconds = self._measure(lambda: self._prepare_inputs(batch, actor, device))
        full_tokens_total = int(sum(full_lengths))
        self._log_stage(
            context,
            "prepare_inputs",
            full_seconds,
            {
                "full_tokens_mean": self._mean(full_lengths),
                "full_tokens_max": max(full_lengths, default=0),
                "response_tokens_total": int(sum(len(rollout.response_token_ids) for rollout in batch.rollouts)),
            },
        )

        actor_logits, actor_forward_seconds = self._measure(
            lambda: actor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        )
        self._log_stage(
            context,
            "actor_forward",
            actor_forward_seconds,
            {"full_tokens_total": full_tokens_total},
        )

        reference_active = self.reference is not None and self.grpo_config.kl_coefficient > 0.0
        if reference_active:
            ref_logits, reference_forward_seconds = self._measure(
                lambda: self._reference_logits(input_ids, attention_mask)
            )
            self._log_stage(
                context,
                "reference_forward",
                reference_forward_seconds,
                {"full_tokens_total": full_tokens_total},
            )
        else:
            ref_logits = None
            reference_forward_seconds = 0.0

        loss_result, loss_seconds = self._measure(
            lambda: self._assemble_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
                actor_logits=actor_logits,
                ref_logits=ref_logits,
                rollout_response_log_probs=rollout_response_log_probs,
                advantages=advantages,
                kl_coefficient=self.grpo_config.kl_coefficient,
                clip_ratio=self.grpo_config.clip_ratio,
            )
        )
        loss, policy_loss, kl_divergence, response_tokens_total = loss_result
        self._log_stage(
            context,
            "loss_assembly",
            loss_seconds,
            {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "kl_divergence": float(kl_divergence.item()),
                "response_tokens_total": response_tokens_total,
            },
        )

        _, backward_seconds = self._measure(self._backward_step(loss))
        self._log_stage(
            context,
            "backward",
            backward_seconds,
            {"loss": float(loss.item())},
        )

        _, optimizer_seconds = self._measure(self.training_backend.optimizer.step)
        self._log_stage(
            context,
            "optimizer",
            optimizer_seconds,
            {"learning_rate": float(self.training_backend.optimizer.param_groups[0]["lr"])},
        )

        _, sync_seconds = self._measure(
            lambda: self.training_backend.sync_weights_to(self.serving_backend)
        )
        stage_timings = {
            "prepare_inputs": full_seconds,
            "actor_forward": actor_forward_seconds,
            "reference_forward": reference_forward_seconds,
            "loss_assembly": loss_seconds,
            "backward": backward_seconds,
            "optimizer": optimizer_seconds,
            "sync": sync_seconds,
        }
        if context is not None:
            partial_step_seconds = sum(stage_timings.values())
            tokens_per_second = (
                response_tokens_total / partial_step_seconds
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

        return OptimizationResult(
            loss=float(loss.item()),
            policy_loss=float(policy_loss.item()),
            kl_divergence=float(kl_divergence.item()),
            stage_timings=stage_timings,
            response_tokens_total=response_tokens_total,
            reference_active=reference_active,
        )

    def _reference_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the frozen reference model without gradients."""
        assert self.reference is not None
        with torch.no_grad():
            return self.reference.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

    def _assemble_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        actor_logits: torch.Tensor,
        ref_logits: torch.Tensor | None,
        rollout_response_log_probs: list[list[float]],
        advantages: torch.Tensor,
        kl_coefficient: float,
        clip_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Assemble the response-only GRPO objective and count response tokens.

        We optimize the sampled response tokens with the paper-style GRPO split:

            L(theta) = L_policy(theta) + beta * L_KL(theta)

            L_policy(theta)
              = -mean_t min(r_t * A_i, clip(r_t, 1 - clip_ratio, 1 + clip_ratio) * A_i)
            r_t = exp(log_pi_theta - log_pi_old)

            L_KL(theta)
              = mean_t [exp(log_pi_ref - log_pi_theta) - (log_pi_ref - log_pi_theta) - 1]

        where:
        - log_pi_theta is the current actor log-prob of the sampled response token
        - log_pi_old is the rollout-policy log-prob captured at sample time
        - log_pi_ref is the frozen reference log-prob of that sampled token
        - A_i is the group-normalized advantage for sample i, reused for every
          response token in that sample
        - beta is `kl_coefficient`

        Prompt tokens only provide context. All reductions are over response
        tokens selected by `response_mask`.
        """
        shift_ids = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        # 1. Current policy log-probs for the sampled next tokens.
        actor_token_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
        log_pi_theta = torch.gather(
            actor_token_log_probs,
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)

        # 2. Response-only mask. Prompt tokens stay out of the objective.
        response_mask = torch.zeros_like(shift_mask)
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            prompt_start = max(int(prompt_length) - 1, 0)
            response_mask[index, prompt_start:] = shift_mask[index, prompt_start:]
        response_token_count = response_mask.sum().clamp(min=1)
        response_tokens_total = int(response_mask.sum().item())

        # 3. Fixed rollout-policy log-probs captured when the samples were generated.
        if len(rollout_response_log_probs) != response_mask.shape[0]:
            raise ValueError(
                "rollout_response_log_probs must match the batch size."
            )

        log_pi_old = torch.zeros(
            response_mask.shape,
            dtype=log_pi_theta.dtype,
            device=log_pi_theta.device,
        )
        response_mask_bool = response_mask.to(dtype=torch.bool)
        for index, sample_log_probs in enumerate(rollout_response_log_probs):
            expected_tokens = int(response_mask_bool[index].sum().item())
            if len(sample_log_probs) != expected_tokens:
                raise ValueError(
                    "Each rollout_response_log_probs entry must match that sample's "
                    "response-token count."
                )
            if expected_tokens == 0:
                continue
            sample_tensor = torch.tensor(
                sample_log_probs,
                dtype=log_pi_theta.dtype,
                device=log_pi_theta.device,
            )
            log_pi_old[index, response_mask_bool[index]] = sample_tensor

        # 4. Clipped GRPO surrogate. Each sample-level A_i is reused for every
        # response token in that sample because this trainer uses outcome-level rewards.
        sample_advantages = advantages.to(
            device=log_pi_theta.device,
            dtype=log_pi_theta.dtype,
        ).unsqueeze(-1)
        expanded_advantages = sample_advantages.expand_as(response_mask)
        ratio = torch.exp(log_pi_theta - log_pi_old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surrogate_unclipped = ratio * expanded_advantages
        surrogate_clipped = clipped_ratio * expanded_advantages
        surrogate_objective = torch.minimum(surrogate_unclipped, surrogate_clipped)
        masked_surrogate = surrogate_objective * response_mask
        policy_loss = -masked_surrogate.sum() / response_token_count

        # 5. Separate reference regularization term.
        if ref_logits is not None:
            reference_token_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            log_pi_ref = torch.gather(
                reference_token_log_probs,
                dim=-1,
                index=shift_ids.unsqueeze(-1),
            ).squeeze(-1)
            reference_log_gap = log_pi_ref - log_pi_theta
            kl_terms = torch.exp(reference_log_gap) - reference_log_gap - 1.0
            masked_kl = kl_terms * response_mask
            kl_divergence = masked_kl.sum() / response_token_count
        else:
            kl_divergence = torch.zeros(
                (),
                device=log_pi_theta.device,
                dtype=log_pi_theta.dtype,
            )

        # 6. Final objective used for backprop.
        loss = policy_loss + kl_coefficient * kl_divergence
        return loss, policy_loss, kl_divergence, response_tokens_total

    def _backward_step(self, loss: torch.Tensor) -> Callable[[], None]:
        """Return the backward closure used in timing."""

        def run() -> None:
            self.training_backend.optimizer.zero_grad()
            loss.backward()

        return run

    def _prepare_inputs(
        self,
        batch: TrainingBatch,
        actor: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
        """Build padded training tensors directly from rollout-provided token ids."""
        pad_token_id = getattr(actor.tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        full_sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        full_lengths: list[int] = []
        rollout_response_log_probs: list[list[float]] = []
        for rollout in batch.rollouts:
            prompt_ids = [int(token_id) for token_id in rollout.prompt_token_ids]
            response_ids = [int(token_id) for token_id in rollout.response_token_ids]
            response_logprobs = [float(value) for value in rollout.response_token_logprobs]
            if len(response_ids) != len(response_logprobs):
                raise ValueError(
                    "RolloutOutput.response_token_logprobs must match RolloutOutput.response_token_ids."
                )
            full_sequence = prompt_ids + response_ids
            if not full_sequence:
                raise ValueError("GRPO training requires at least one prompt token per rollout.")
            full_sequences.append(full_sequence)
            prompt_lengths.append(len(prompt_ids))
            full_lengths.append(len(full_sequence))
            rollout_response_log_probs.append(response_logprobs)

        batch_size = len(full_sequences)
        max_length = max(full_lengths, default=0)
        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        for index, token_ids in enumerate(full_sequences):
            length = len(token_ids)
            input_ids[index, :length] = torch.tensor(token_ids, dtype=torch.long, device=device)
            attention_mask[index, :length] = 1

        prompt_length_tensor = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
        return input_ids, attention_mask, prompt_length_tensor, full_lengths, rollout_response_log_probs

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
            self.metrics_sink.observe_stage(stage, latency_seconds)

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
                self.metrics_sink.observe_serving_debug(
                    ttft_seconds=float(payload["ttft_seconds"]),
                    tpot_seconds=float(payload["tpot_seconds"]),
                )

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
