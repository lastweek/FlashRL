"""Shared training backend abstractions and optimization result types."""

from __future__ import annotations

from abc import ABC
import copy
from dataclasses import dataclass, field
import inspect
from typing import Any, Literal

import torch

from flashrl.framework.config import TrainingConfig
from flashrl.framework.data_models import LearnerBatch
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.observability import StageResult
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.serving.base import ServingBackend
from flashrl.framework.training.optimization import (
    LossAssemblyResult,
    backward_step,
    compute_rollout_log_probs_from_actor,
    prepare_learner_inputs,
)


BackendRole = Literal["actor", "reference"]


def _model_accepts_use_cache(model: Any) -> bool:
    """Return whether the target module forward accepts a ``use_cache`` kwarg."""
    cached = getattr(model, "_flashrl_accepts_use_cache", None)
    if isinstance(cached, bool):
        return cached

    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        supports_use_cache = True
    else:
        supports_use_cache = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD or name == "use_cache"
            for name, parameter in signature.parameters.items()
        )
    setattr(model, "_flashrl_accepts_use_cache", supports_use_cache)
    return supports_use_cache


def _forward_model_logits(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward a causal LM while disabling KV cache only when supported."""
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if _model_accepts_use_cache(model):
        forward_kwargs["use_cache"] = False
    return model(**forward_kwargs).logits


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
        """Keep legacy dict views and the canonical stage list in sync."""
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
    """Single-engine training backend used by controller code."""

    config: TrainingConfig
    role: BackendRole
    model_copy: ActorModel
    device: Any
    rank: int
    world_size: int
    is_primary: bool
    startup_events: list[dict[str, Any]]

    def __init__(
        self,
        config: TrainingConfig,
        *,
        role: BackendRole,
    ) -> None:
        self.config = config
        self.role = role
        self.rank = 0
        self.world_size = int(config.dp_size)
        self.is_primary = True
        self.startup_events = []

    @property
    def backend_name(self) -> str:
        """Return the configured backend name."""
        return self.config.backend

    @property
    def component_name(self) -> str:
        """Return the stable component label used in logs and admin views."""
        return f"{self.role}_backend"

    @property
    def model_name(self) -> str:
        """Return the user-facing model name for this backend."""
        return self.config.model_name

    @property
    def model(self):
        """Expose the wrapped causal LM module directly when needed."""
        return self.model_copy.model

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one backend-owned forward pass and return logits."""
        return _forward_model_logits(
            self.model_copy.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def export_state(self) -> dict[str, Any]:
        """Return the backend-owned checkpoint payload for this role."""
        return {
            "role": self.role,
            "backend": self.backend_name,
            "world_size": self.world_size,
            "model_state_dict": copy.deepcopy(self.model_copy.model.state_dict()),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load backend-owned checkpoint state for this role."""
        self.model_copy.model.load_state_dict(state["model_state_dict"])

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
        """Release backend-owned resources."""
        return None

    def prepare_inputs(
        self,
        learner_batch: LearnerBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
        """Build padded learner tensors directly from rollout-provided token ids."""
        return prepare_learner_inputs(learner_batch, self.model_copy, self.device)

    def compute_rollout_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> list[list[float]]:
        """Derive rollout-policy token logprobs from the synced actor backend."""
        return compute_rollout_log_probs_from_actor(
            actor=self.model_copy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )


class ActorTrainingBackend(TrainingBackend):
    """Mutable learner backend for the actor role."""

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        config: TrainingConfig,
        *,
        learning_rate: float,
    ) -> None:
        super().__init__(config, role="actor")
        self.learning_rate = learning_rate

    @property
    def optimizer_name(self) -> str:
        """Return the optimizer class name for admin/logging."""
        return type(self.optimizer).__name__

    @property
    def actor(self) -> ActorModel:
        """Expose the actor model wrapper under the traditional actor name."""
        return self.model_copy

    def backward_step(self, loss: torch.Tensor):
        """Return the timed backward closure for the actor backend."""
        return backward_step(self.optimizer, loss)

    def optimizer_step(self) -> None:
        """Advance the actor optimizer once."""
        self.optimizer.step()
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()

    def sync_weights_to(
        self,
        serving_backend: ServingBackend,
        *,
        source_training_step: int | None = None,
        source_epoch: int | None = None,
        origin: str = "sync",
    ) -> WeightVersionInfo:
        """Sync the backend-owned actor weights into the serving backend."""
        return serving_backend.sync_from_training_actor(
            self.model_copy,
            source_training_step=source_training_step,
            source_epoch=source_epoch,
            origin=origin,
        )

    def export_state(self) -> dict[str, Any]:
        """Return the actor checkpoint payload including optimizer state."""
        state = super().export_state()
        state["optimizer_state_dict"] = copy.deepcopy(self.optimizer.state_dict())
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Load actor model weights plus optimizer state."""
        super().load_state(state)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])


class ReferenceTrainingBackend(TrainingBackend):
    """Frozen reference backend for stable policy evaluation.

    The reference model is kept frozen during training and provides
    stable log-probability targets for KL divergence computation.

    This backend is optimized for inference-only operations and does
    not support training methods like backward_step() or optimizer_step().
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config, role="reference")

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass with torch.no_grad() for efficiency."""
        with torch.no_grad():
            return _forward_model_logits(
                self.model_copy.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    def backward_step(self, loss: torch.Tensor):
        """Reference models do not support backward passes."""
        raise NotImplementedError(
            "Reference models are frozen and cannot perform backward passes"
        )

    def optimizer_step(self) -> None:
        """Reference models do not support optimizer steps."""
        raise NotImplementedError(
            "Reference models are frozen and cannot perform optimizer steps"
        )

    def sync_weights_to(self, serving_backend: ServingBackend) -> None:
        """Reference models do not sync weights to serving backends."""
        raise NotImplementedError(
            "Reference models do not sync weights to serving backends"
        )


def assemble_loss(
    *,
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
    """Compatibility wrapper for the shared GRPO loss helper."""
    from flashrl.framework.training.optimization import assemble_grpo_loss

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
