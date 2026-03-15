"""Shared training backend abstraction and learner-side optimization flow."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import torch

from flashrl.framework.config import GrpoConfig, TrainingConfig
from flashrl.framework.data_models import LearnerBatch
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.observability import StageResult, timed_call
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.training.optimization import (
    LossAssemblyResult,
    assemble_grpo_loss,
    backward_step,
    compute_rollout_log_probs_from_actor,
    prepare_learner_inputs,
    reference_logits,
)


@dataclass
class OptimizationResult:
    """Result of one learner-side optimization step."""

    loss: float
    policy_loss: float
    kl_divergence: float
    learning_rate: float
    response_tokens_total: int
    reference_active: bool
    stages: list[StageResult] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def stage(self, name: str) -> StageResult | None:
        """Return one named stage result when present."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def __post_init__(self) -> None:
        """Keep legacy dict views and the new stage list in sync."""
        if self.stages:
            self.refresh_stage_views()
            return

        if not self.stage_timings:
            return

        self.stages = [
            StageResult(
                name=stage_name,
                seconds=float(seconds),
                metrics=dict(self.stage_metrics.get(stage_name, {})),
            )
            for stage_name, seconds in self.stage_timings.items()
        ]
        self.refresh_stage_views()

    def refresh_stage_views(self) -> None:
        """Refresh compatibility dicts from the canonical stage list."""
        self.stage_timings = {stage.name: float(stage.seconds) for stage in self.stages}
        self.stage_metrics = {stage.name: dict(stage.metrics) for stage in self.stages}


class TrainingBackend(ABC):
    """Shared training surface used by controller code."""

    config: TrainingConfig
    actor: ActorModel
    optimizer: torch.optim.Optimizer
    device: Any
    rank: int
    world_size: int
    is_primary: bool
    startup_events: list[dict[str, Any]]

    def __init__(
        self,
        config: TrainingConfig,
        *,
        learning_rate: float,
        grpo_config: GrpoConfig,
        reference_enabled: bool = False,
        reference_device: str | None = None,
    ) -> None:
        self.config = config
        self.learning_rate = learning_rate
        self.grpo_config = grpo_config
        self.reference_enabled = bool(reference_enabled)
        self.reference_device = reference_device
        self.reference: ReferenceModel | None = None
        self.rank = 0
        self.world_size = int(config.dp_size)
        self.is_primary = True
        self.startup_events: list[dict[str, Any]] = []

    @property
    def backend_name(self) -> str:
        """Return the configured backend name."""
        return self.config.backend

    @property
    def optimizer_name(self) -> str:
        """Return the optimizer class name for admin/logging."""
        return type(self.optimizer).__name__

    @property
    def reference_loaded(self) -> bool:
        """Return whether this backend owns a frozen reference model."""
        return self.reference is not None

    @property
    def resolved_reference_device(self) -> str:
        """Return the user-facing reference device label."""
        if self.reference is not None:
            return str(self.reference.device)
        return self.reference_device or self.config.device or "auto"

    def optimize_batch(self, learner_batch: LearnerBatch) -> OptimizationResult:
        """Run learner-side tensor prep, forward, loss, backward, and step."""
        actor = self.actor
        device = actor.device
        (
            input_ids,
            attention_mask,
            prompt_lengths,
            full_lengths,
            rollout_response_log_probs,
        ), prepare_seconds = timed_call(
            lambda: self._prepare_inputs(learner_batch, actor, device)
        )
        full_tokens_total = int(sum(full_lengths))
        response_tokens_total = int(sum(len(tokens) for tokens in learner_batch.response_token_ids))

        actor_logits, actor_forward_seconds = timed_call(
            lambda: actor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        )

        reference_active = self.reference is not None and self.grpo_config.kl_coefficient > 0.0
        if reference_active:
            ref_logits, reference_forward_seconds = timed_call(
                lambda: self._reference_logits(input_ids, attention_mask)
            )
        else:
            ref_logits = None
            reference_forward_seconds = 0.0

        advantages = torch.tensor(
            learner_batch.advantages,
            dtype=torch.float32,
            device=device,
        )
        loss_result, loss_seconds = timed_call(
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

        _, backward_seconds = timed_call(self._backward_step(loss_result.loss))
        _, optimizer_seconds = timed_call(self.optimizer.step)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        stages = [
            StageResult(
                name="prepare_inputs",
                seconds=prepare_seconds,
                metrics={
                    "full_tokens_mean": self._mean(full_lengths),
                    "full_tokens_max": max(full_lengths, default=0),
                    "response_tokens_total": response_tokens_total,
                },
            ),
            StageResult(
                name="actor_forward",
                seconds=actor_forward_seconds,
                metrics={"full_tokens_total": full_tokens_total},
            ),
        ]
        if reference_active:
            stages.append(
                StageResult(
                    name="reference_forward",
                    seconds=reference_forward_seconds,
                    metrics={"full_tokens_total": full_tokens_total},
                )
            )
        stages.extend(
            [
                StageResult(
                    name="loss_assembly",
                    seconds=loss_seconds,
                    metrics={
                        "loss": float(loss_result.loss.item()),
                        "policy_loss": float(loss_result.policy_loss.item()),
                        "kl_divergence": float(loss_result.kl_divergence.item()),
                        "response_tokens_total": loss_result.response_tokens_total,
                        "importance_sampling_ratio_mean": (
                            loss_result.importance_sampling_ratio_mean
                        ),
                        "importance_sampling_ratio_std": loss_result.importance_sampling_ratio_std,
                        "importance_sampling_ratio_min": loss_result.importance_sampling_ratio_min,
                        "importance_sampling_ratio_max": loss_result.importance_sampling_ratio_max,
                        "clip_fraction": loss_result.clip_fraction,
                    },
                ),
                StageResult(
                    name="backward",
                    seconds=backward_seconds,
                    metrics={"loss": float(loss_result.loss.item())},
                ),
                StageResult(
                    name="optimizer",
                    seconds=optimizer_seconds,
                    metrics={"learning_rate": learning_rate},
                ),
            ]
        )

        return OptimizationResult(
            loss=float(loss_result.loss.item()),
            policy_loss=float(loss_result.policy_loss.item()),
            kl_divergence=float(loss_result.kl_divergence.item()),
            learning_rate=learning_rate,
            response_tokens_total=loss_result.response_tokens_total,
            reference_active=reference_active,
            stages=stages,
        )

    def sync_weights_to(self, serving_backend: ServingBackend) -> None:
        """Sync the backend-owned actor weights into the serving backend."""
        serving_backend.sync_from_training_actor(self.actor)

    def save_checkpoint(
        self,
        path: str,
        controller_state: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save controller plus backend-owned training state into one envelope."""
        payload = {
            "controller_state": dict(controller_state or {}),
            "backend_state": {
                "actor_state_dict": self.actor.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "reference_state_dict": (
                    self.reference.model.state_dict() if self.reference is not None else None
                ),
                "backend": self.backend_name,
                "world_size": self.world_size,
            },
            "checkpoint_metadata": dict(checkpoint_metadata or {}),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load a checkpoint envelope and return the controller-owned state."""
        controller_state, _ = self.load_checkpoint_with_metadata(path)
        return controller_state

    def load_checkpoint_with_metadata(
        self,
        path: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Load checkpoint weights and return controller state plus metadata."""
        checkpoint = torch.load(path, weights_only=False)
        backend_state, controller_state, checkpoint_metadata = self._split_checkpoint_payload(
            checkpoint
        )

        self.actor.model.load_state_dict(backend_state["actor_state_dict"])
        self.optimizer.load_state_dict(backend_state["optimizer_state_dict"])
        reference_state = backend_state.get("reference_state_dict")
        if reference_state is not None and self.reference is not None:
            self.reference.model.load_state_dict(reference_state)
        return dict(controller_state), checkpoint_metadata

    def read_checkpoint_metadata(self, path: str) -> dict[str, Any] | None:
        """Return checkpoint metadata without mutating backend state."""
        checkpoint = torch.load(path, weights_only=False)
        _, _, checkpoint_metadata = self._split_checkpoint_payload(checkpoint)
        return checkpoint_metadata

    def _split_checkpoint_payload(
        self,
        checkpoint: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Normalize legacy and current checkpoint payloads into one shape."""
        if "backend_state" in checkpoint:
            backend_state = checkpoint["backend_state"]
            controller_state = checkpoint.get("controller_state", {})
            checkpoint_metadata = checkpoint.get("checkpoint_metadata")
            if isinstance(checkpoint_metadata, dict):
                return backend_state, controller_state, dict(checkpoint_metadata)
            return backend_state, controller_state, None

        backend_state = checkpoint
        controller_state = {
            "epoch": checkpoint.get("epoch", 0),
            "total_steps": checkpoint.get("total_steps", 0),
        }
        return backend_state, controller_state, None

    def barrier(self) -> None:
        """Backend-local barrier hook."""
        return None

    def broadcast_object(self, obj: Any) -> Any:
        """Backend-local object broadcast hook."""
        return obj

    def list_admin_objects(self) -> list[dict[str, Any]]:
        """Return backend-owned admin child objects when available."""
        return []

    def close(self) -> None:
        """Release any backend-owned resources."""
        return None

    def _reference_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the frozen reference model without gradients."""
        assert self.reference is not None
        return reference_logits(self.reference, input_ids, attention_mask)

    def _backward_step(self, loss: torch.Tensor):
        """Return the backward closure used in timing."""
        return backward_step(self.optimizer, loss)

    def _prepare_inputs(
        self,
        learner_batch: LearnerBatch,
        actor: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
        """Build padded learner tensors directly from rollout-provided token ids."""
        return prepare_learner_inputs(learner_batch, actor, device)

    def _compute_rollout_log_probs_from_actor(
        self,
        actor: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> list[list[float]]:
        """Derive rollout-policy token logprobs from the synced training actor."""
        return compute_rollout_log_probs_from_actor(
            actor=actor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )

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
    ) -> LossAssemblyResult:
        """Assemble the response-only GRPO objective and count response tokens."""
        return assemble_grpo_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            actor_logits=actor_logits,
            ref_logits=ref_logits,
            rollout_response_log_probs=rollout_response_log_probs,
            advantages=advantages,
            kl_coefficient=kl_coefficient,
            clip_ratio=clip_ratio,
        )

    def _mean(self, values: list[float] | list[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))
