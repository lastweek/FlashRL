"""Shared training backend abstraction and learner-side optimization helpers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
import time
from typing import Any

import torch
import torch.nn.functional as F

from flashrl.framework.config import GrpoConfig, TrainingConfig
from flashrl.framework.data_models import LearnerBatch
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.serving.base import ServingBackend


@dataclass
class LossAssemblyResult:
    """Full response-only GRPO loss assembly output."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_divergence: torch.Tensor
    response_tokens_total: int
    importance_sampling_ratio_mean: float
    importance_sampling_ratio_std: float
    importance_sampling_ratio_min: float
    importance_sampling_ratio_max: float
    clip_fraction: float

    def __iter__(self):
        """Preserve tuple-style unpacking in tests."""
        yield self.loss
        yield self.policy_loss
        yield self.kl_divergence
        yield self.response_tokens_total


@dataclass
class OptimizationResult:
    """Result of one learner-side optimization step."""

    loss: float
    policy_loss: float
    kl_divergence: float
    learning_rate: float
    stage_timings: dict[str, float]
    response_tokens_total: int
    reference_active: bool
    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        ), prepare_seconds = self._measure(lambda: self._prepare_inputs(learner_batch, actor, device))
        full_tokens_total = int(sum(full_lengths))
        response_tokens_total = int(sum(len(tokens) for tokens in learner_batch.response_token_ids))

        actor_logits, actor_forward_seconds = self._measure(
            lambda: actor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        )

        reference_active = self.reference is not None and self.grpo_config.kl_coefficient > 0.0
        if reference_active:
            ref_logits, reference_forward_seconds = self._measure(
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

        _, backward_seconds = self._measure(self._backward_step(loss_result.loss))
        _, optimizer_seconds = self._measure(self.optimizer.step)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        stage_timings = {
            "prepare_inputs": prepare_seconds,
            "actor_forward": actor_forward_seconds,
            "reference_forward": reference_forward_seconds,
            "loss_assembly": loss_seconds,
            "backward": backward_seconds,
            "optimizer": optimizer_seconds,
        }
        stage_metrics = {
            "prepare_inputs": {
                "full_tokens_mean": self._mean(full_lengths),
                "full_tokens_max": max(full_lengths, default=0),
                "response_tokens_total": response_tokens_total,
            },
            "actor_forward": {"full_tokens_total": full_tokens_total},
            "reference_forward": {"full_tokens_total": full_tokens_total},
            "loss_assembly": {
                "loss": float(loss_result.loss.item()),
                "policy_loss": float(loss_result.policy_loss.item()),
                "kl_divergence": float(loss_result.kl_divergence.item()),
                "response_tokens_total": loss_result.response_tokens_total,
                "importance_sampling_ratio_mean": loss_result.importance_sampling_ratio_mean,
                "importance_sampling_ratio_std": loss_result.importance_sampling_ratio_std,
                "importance_sampling_ratio_min": loss_result.importance_sampling_ratio_min,
                "importance_sampling_ratio_max": loss_result.importance_sampling_ratio_max,
                "clip_fraction": loss_result.clip_fraction,
            },
            "backward": {"loss": float(loss_result.loss.item())},
            "optimizer": {"learning_rate": learning_rate},
        }

        return OptimizationResult(
            loss=float(loss_result.loss.item()),
            policy_loss=float(loss_result.policy_loss.item()),
            kl_divergence=float(loss_result.kl_divergence.item()),
            learning_rate=learning_rate,
            stage_timings=stage_timings,
            response_tokens_total=loss_result.response_tokens_total,
            reference_active=reference_active,
            stage_metrics=stage_metrics,
        )

    def sync_weights_to(self, serving_backend: ServingBackend) -> None:
        """Sync the backend-owned actor weights into the serving backend."""
        serving_backend.sync_from_training_actor(self.actor)

    def save_checkpoint(
        self,
        path: str,
        controller_state: dict[str, Any] | None = None,
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
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load a checkpoint envelope and return the controller-owned state."""
        checkpoint = torch.load(path, weights_only=False)
        if "backend_state" in checkpoint:
            backend_state = checkpoint["backend_state"]
            controller_state = checkpoint.get("controller_state", {})
        else:
            backend_state = checkpoint
            controller_state = {
                "epoch": checkpoint.get("epoch", 0),
                "total_steps": checkpoint.get("total_steps", 0),
            }

        self.actor.model.load_state_dict(backend_state["actor_state_dict"])
        self.optimizer.load_state_dict(backend_state["optimizer_state_dict"])
        reference_state = backend_state.get("reference_state_dict")
        if reference_state is not None and self.reference is not None:
            self.reference.model.load_state_dict(reference_state)
        return dict(controller_state)

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

    def _measure(self, operation):
        started_at = time.perf_counter()
        result = operation()
        return result, time.perf_counter() - started_at

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

    def _backward_step(self, loss: torch.Tensor):
        """Return the backward closure used in timing."""

        def run() -> None:
            self.optimizer.zero_grad()
            loss.backward()

        return run

    def _prepare_inputs(
        self,
        learner_batch: LearnerBatch,
        actor: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
        """Build padded learner tensors directly from rollout-provided token ids."""
        pad_token_id = getattr(actor.tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        full_sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        full_lengths: list[int] = []
        rollout_response_log_probs: list[list[float] | None] = []
        for prompt_ids, response_ids, response_logprobs in zip(
            learner_batch.prompt_token_ids,
            learner_batch.response_token_ids,
            learner_batch.response_token_logprobs,
            strict=True,
        ):
            prompt_ids = [int(token_id) for token_id in prompt_ids]
            response_ids = [int(token_id) for token_id in response_ids]
            response_logprobs = [float(value) for value in response_logprobs]
            full_sequence = prompt_ids + response_ids
            if not full_sequence:
                raise ValueError("GRPO training requires at least one prompt token per rollout.")
            full_sequences.append(full_sequence)
            prompt_lengths.append(len(prompt_ids))
            full_lengths.append(len(full_sequence))
            if response_ids and len(response_ids) != len(response_logprobs):
                rollout_response_log_probs.append(None)
            else:
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
        if any(sample_log_probs is None for sample_log_probs in rollout_response_log_probs):
            computed_log_probs = self._compute_rollout_log_probs_from_actor(
                actor=actor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
            )
            rollout_response_log_probs = [
                computed if sample_log_probs is None else sample_log_probs
                for sample_log_probs, computed in zip(
                    rollout_response_log_probs,
                    computed_log_probs,
                    strict=True,
                )
            ]
        return (
            input_ids,
            attention_mask,
            prompt_length_tensor,
            full_lengths,
            [list(sample_log_probs) for sample_log_probs in rollout_response_log_probs],
        )

    def _compute_rollout_log_probs_from_actor(
        self,
        actor: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> list[list[float]]:
        """Derive rollout-policy token logprobs from the synced training actor."""
        with torch.no_grad():
            if hasattr(actor, "compute_log_probs"):
                actor_logits = actor.compute_log_probs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
            else:
                outputs = actor.model(
                    input_ids=input_ids.to(actor.device),
                    attention_mask=attention_mask.to(actor.device),
                    labels=input_ids.to(actor.device),
                )
                actor_logits = outputs.logits

        token_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
        shift_ids = input_ids[:, 1:]
        gathered = torch.gather(
            token_log_probs,
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)
        shift_mask = attention_mask[:, 1:].to(dtype=torch.bool)

        computed: list[list[float]] = []
        for index, prompt_length in enumerate(prompt_lengths):
            response_mask = torch.zeros_like(shift_mask[index], dtype=torch.bool)
            prompt_start = max(int(prompt_length) - 1, 0)
            response_mask[prompt_start:] = shift_mask[index, prompt_start:]
            computed.append(gathered[index, response_mask].detach().cpu().tolist())
        return computed

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
        shift_ids = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        actor_token_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
        log_pi_theta = torch.gather(
            actor_token_log_probs,
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)

        response_mask = torch.zeros_like(shift_mask)
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            prompt_start = max(int(prompt_length) - 1, 0)
            response_mask[index, prompt_start:] = shift_mask[index, prompt_start:]
        response_token_count = response_mask.sum().clamp(min=1)
        response_tokens_total = int(response_mask.sum().item())

        if len(rollout_response_log_probs) != response_mask.shape[0]:
            raise ValueError("rollout_response_log_probs must match the batch size.")

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

        ratio_values = ratio[response_mask_bool]
        if ratio_values.numel() > 0:
            importance_sampling_ratio_mean = float(ratio_values.mean().item())
            importance_sampling_ratio_std = float(ratio_values.std(unbiased=False).item())
            importance_sampling_ratio_min = float(ratio_values.min().item())
            importance_sampling_ratio_max = float(ratio_values.max().item())
            clip_fraction = float(
                ((ratio_values < (1.0 - clip_ratio)) | (ratio_values > (1.0 + clip_ratio)))
                .float()
                .mean()
                .item()
            )
        else:
            importance_sampling_ratio_mean = 0.0
            importance_sampling_ratio_std = 0.0
            importance_sampling_ratio_min = 0.0
            importance_sampling_ratio_max = 0.0
            clip_fraction = 0.0

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

        loss = policy_loss + kl_coefficient * kl_divergence
        return LossAssemblyResult(
            loss=loss,
            policy_loss=policy_loss,
            kl_divergence=kl_divergence,
            response_tokens_total=response_tokens_total,
            importance_sampling_ratio_mean=importance_sampling_ratio_mean,
            importance_sampling_ratio_std=importance_sampling_ratio_std,
            importance_sampling_ratio_min=importance_sampling_ratio_min,
            importance_sampling_ratio_max=importance_sampling_ratio_max,
            clip_fraction=clip_fraction,
        )

    def _mean(self, values: list[float] | list[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))
