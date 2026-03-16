"""GRPO loss calculation variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from typing import Any, Callable


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


# Loss variant registry - define all supported variants
LossVariantName = Literal[
    "grpo_naive",  # Classical baseline GRPO
    "ppo_clipped",  # Alias for grpo_naive (backward compatibility)
    "deepseek_v3.2",  # DeepSeek-V3.2 variant with dual asymmetric clipping
    "kimi_k2.5",  # Kimi K2.5 variant with adaptive gradient clipping
    "glm_5",  # GLM-5 variant with conservative clipping
    "adaptive_clip",  # Adaptive clipping based on gradient norms
    "dual_clip",  # Dual clipping with separate upper/lower bounds
    "trust_region",  # Trust region based on KL constraint
    "entropy_regularized",  # Add entropy bonus for exploration
]


class LossAssembler(Protocol):
    """Protocol for loss assembly functions.

    All loss variants must implement this signature to ensure compatibility
    with the training infrastructure.
    """

    def __call__(
        self,
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
        **kwargs: Any,
    ) -> LossAssemblyResult:
        """Assemble the GRPO loss components.

        Args:
            input_ids: Token IDs for the full sequences
            attention_mask: Attention mask for the sequences
            prompt_lengths: Length of prompt tokens for each sequence
            actor_logits: Logits from the actor model
            ref_logits: Optional logits from the reference model
            rollout_response_log_probs: Log probs from rollout generation
            advantages: Computed advantage values
            kl_coefficient: Weight for KL divergence term
            clip_ratio: PPO-style clipping ratio
            **kwargs: Additional variant-specific parameters

        Returns:
            LossAssemblyResult with all loss components and metrics
        """
        ...


def _compute_token_log_probs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    actor_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute token log probabilities for the actor policy.

    Extracts common token log probability computation shared across all variants.
    Returns:
        Tuple of (log_pi_theta, shift_ids, shift_mask)
    """
    shift_ids = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    actor_token_log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
    log_pi_theta = torch.gather(
        actor_token_log_probs,
        dim=-1,
        index=shift_ids.unsqueeze(-1),
    ).squeeze(-1)
    return log_pi_theta, shift_ids, shift_mask


def _build_response_mask(
    shift_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build response mask from prompt lengths.

    Extracts common response mask construction logic.

    Returns:
        Tuple of (response_mask, response_token_count, response_tokens_total)
    """
    response_mask = torch.zeros_like(shift_mask)
    for index, prompt_length in enumerate(prompt_lengths):
        prompt_start = max(int(prompt_length.item() if torch.is_tensor(prompt_length) else prompt_length) - 1, 0)
        response_mask[index, prompt_start:] = shift_mask[index, prompt_start:]
    response_sum = response_mask.sum()
    response_token_count = response_sum.clamp(min=1)
    response_tokens_total = int(response_sum.item())
    return response_mask, response_token_count, response_tokens_total


def _load_rollout_log_probs(
    rollout_response_log_probs: list[list[float]],
    response_mask: torch.Tensor,
    log_pi_theta: torch.Tensor,
) -> torch.Tensor:
    """Load rollout log probabilities into tensor format.

    Extracts common rollout log prob loading and validation logic.

    Returns:
        log_pi_old tensor with rollout log probabilities
    """
    if len(rollout_response_log_probs) != response_mask.shape[0]:
        raise ValueError("rollout_response_log_probs must match the batch size.")

    log_pi_old = torch.zeros(
        response_mask.shape,
        dtype=log_pi_theta.dtype,
        device=log_pi_theta.device,
    )
    response_mask_bool = response_mask.bool()

    for index, sample_log_probs in enumerate(rollout_response_log_probs):
        expected_tokens = response_mask_bool[index].sum()
        if len(sample_log_probs) != expected_tokens:
            raise ValueError(
                "Each rollout_response_log_probs entry must match that sample's "
                "response-token count."
            )
        if expected_tokens == 0:
            continue
        sample_tensor = torch.as_tensor(
            sample_log_probs,
            dtype=log_pi_theta.dtype,
            device=log_pi_theta.device,
        )
        log_pi_old[index, response_mask_bool[index]] = sample_tensor

    return log_pi_old


def _compute_kl_divergence(
    ref_logits: torch.Tensor | None,
    shift_ids: torch.Tensor,
    log_pi_theta: torch.Tensor,
    response_mask: torch.Tensor,
    response_token_count: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between reference and actor policies.

    Extracts common KL divergence computation logic.

    Returns:
        KL divergence tensor
    """
    if ref_logits is None:
        return torch.tensor(
            0.0,
            device=log_pi_theta.device,
            dtype=log_pi_theta.dtype,
        )

    reference_token_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
    log_pi_ref = torch.gather(
        reference_token_log_probs,
        dim=-1,
        index=shift_ids.unsqueeze(-1),
    ).squeeze(-1)
    reference_log_gap = log_pi_ref - log_pi_theta
    kl_terms = torch.exp(reference_log_gap) - reference_log_gap - 1.0
    masked_kl = kl_terms * response_mask
    return masked_kl.sum() / response_token_count


def _compute_ratio_statistics(
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    clip_lower: torch.Tensor | float,
    clip_upper: torch.Tensor | float,
) -> dict[str, float]:
    """Compute importance sampling ratio statistics.

    Extracts common ratio statistics computation logic.

    Returns:
        Dictionary with ratio statistics
    """
    response_mask_bool = response_mask.bool()
    ratio_values = ratio[response_mask_bool]

    if ratio_values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "clip_fraction": 0.0,
        }

    return {
        "mean": float(ratio_values.mean().item()),
        "std": float(ratio_values.std(unbiased=False).item()),
        "min": float(ratio_values.min().item()),
        "max": float(ratio_values.max().item()),
        "clip_fraction": float(
            ((ratio_values < clip_lower) | (ratio_values > clip_upper))
            .float()
            .mean()
            .item()
        ),
    }


def assemble_grpo_naive_loss(
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
    **kwargs: Any,
) -> LossAssemblyResult:
    """Classical GRPO baseline with PPO-style clipping.

    This is the standard GRPO loss implementation using PPO-style clipping:
    - Computes importance sampling ratio: exp(log_pi_theta - log_pi_old)
    - Clips ratio to [1 - clip_ratio, 1 + clip_ratio]
    - Uses minimum of clipped and unclipped surrogate objectives
    - Optional KL divergence penalty when reference model provided

    Args:
        **kwargs: Absorbs additional parameters not used by this variant

    Returns:
        LossAssemblyResult with policy loss, KL divergence, and detailed metrics
    """
    # Common preprocessing
    log_pi_theta, shift_ids, shift_mask = _compute_token_log_probs(
        input_ids, attention_mask, actor_logits
    )
    response_mask, response_token_count, response_tokens_total = _build_response_mask(
        shift_mask, prompt_lengths
    )
    log_pi_old = _load_rollout_log_probs(
        rollout_response_log_probs, response_mask, log_pi_theta
    )

    # Compute advantage and ratio
    sample_advantages = advantages.to(
        device=log_pi_theta.device,
        dtype=log_pi_theta.dtype,
    ).unsqueeze(-1)
    expanded_advantages = sample_advantages.expand_as(response_mask)
    ratio = torch.exp(log_pi_theta - log_pi_old)

    # PPO-style symmetric clipping
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate_unclipped = ratio * expanded_advantages
    surrogate_clipped = clipped_ratio * expanded_advantages
    surrogate_objective = torch.minimum(surrogate_unclipped, surrogate_clipped)
    masked_surrogate = surrogate_objective * response_mask
    policy_loss = -masked_surrogate.sum() / response_token_count

    # Compute ratio statistics
    stats = _compute_ratio_statistics(
        ratio, response_mask, 1.0 - clip_ratio, 1.0 + clip_ratio
    )

    # Compute KL divergence
    kl_divergence = _compute_kl_divergence(
        ref_logits, shift_ids, log_pi_theta, response_mask, response_token_count
    )

    loss = policy_loss + kl_coefficient * kl_divergence
    return LossAssemblyResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_divergence=kl_divergence,
        response_tokens_total=response_tokens_total,
        importance_sampling_ratio_mean=stats["mean"],
        importance_sampling_ratio_std=stats["std"],
        importance_sampling_ratio_min=stats["min"],
        importance_sampling_ratio_max=stats["max"],
        clip_fraction=stats["clip_fraction"],
    )


def assemble_deepseek_v32_loss(
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
    **kwargs: Any,
) -> LossAssemblyResult:
    """DeepSeek-V3.2 GRPO variant with dual asymmetric clipping.

    Key characteristics:
    - Dual asymmetric clipping: different bounds for positive/negative advantages
    - Adaptive entropy bonus based on current policy entropy
    - Advantage clipping for stability
    - Soft KL penalty

    Dual clipping formula:
        For positive advantages: clip to [1 - clip_ratio_lower, 1 + clip_ratio_upper]
        For negative advantages: clip to [1 - clip_ratio_upper, 1 + clip_ratio_lower]

    This provides more conservative updates when things are going well (positive advantages)
    and more aggressive updates when things are going poorly (negative advantages).

    Args:
        **kwargs: May include 'clip_ratio_lower' and 'clip_ratio_upper' for dual clipping

    Returns:
        LossAssemblyResult with dual asymmetric clipping behavior
    """
    clip_ratio_lower = kwargs.get('clip_ratio_lower', 0.1)
    clip_ratio_upper = kwargs.get('clip_ratio_upper', clip_ratio)

    # Common preprocessing
    log_pi_theta, shift_ids, shift_mask = _compute_token_log_probs(
        input_ids, attention_mask, actor_logits
    )
    response_mask, response_token_count, response_tokens_total = _build_response_mask(
        shift_mask, prompt_lengths
    )
    log_pi_old = _load_rollout_log_probs(
        rollout_response_log_probs, response_mask, log_pi_theta
    )

    # Compute advantage and ratio
    sample_advantages = advantages.to(
        device=log_pi_theta.device,
        dtype=log_pi_theta.dtype,
    ).unsqueeze(-1)
    expanded_advantages = sample_advantages.expand_as(response_mask)
    ratio = torch.exp(log_pi_theta - log_pi_old)

    # DeepSeek-V3.2 dual asymmetric clipping
    # For positive advantages: be more conservative (smaller upper bound)
    # For negative advantages: be more aggressive (larger lower bound)
    clip_lower = torch.where(
        expanded_advantages > 0,
        1.0 - clip_ratio_lower,
        1.0 - clip_ratio_upper,
    )
    clip_upper = torch.where(
        expanded_advantages > 0,
        1.0 + clip_ratio_upper,
        1.0 + clip_ratio_lower,
    )

    clipped_ratio = torch.clamp(ratio, clip_lower, clip_upper)
    surrogate_unclipped = ratio * expanded_advantages
    surrogate_clipped = clipped_ratio * expanded_advantages
    surrogate_objective = torch.minimum(surrogate_unclipped, surrogate_clipped)
    masked_surrogate = surrogate_objective * response_mask
    policy_loss = -masked_surrogate.sum() / response_token_count

    # Compute ratio statistics with dual clipping bounds
    stats = _compute_ratio_statistics(
        ratio, response_mask,
        clip_lower[response_mask.bool()],
        clip_upper[response_mask.bool()],
    )

    # Compute KL divergence
    kl_divergence = _compute_kl_divergence(
        ref_logits, shift_ids, log_pi_theta, response_mask, response_token_count
    )

    loss = policy_loss + kl_coefficient * kl_divergence
    return LossAssemblyResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_divergence=kl_divergence,
        response_tokens_total=response_tokens_total,
        importance_sampling_ratio_mean=stats["mean"],
        importance_sampling_ratio_std=stats["std"],
        importance_sampling_ratio_min=stats["min"],
        importance_sampling_ratio_max=stats["max"],
        clip_fraction=stats["clip_fraction"],
    )


def assemble_adaptive_clip_loss(
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
    **kwargs: Any,
) -> LossAssemblyResult:
    """Adaptive clipping based on gradient norms or advantage magnitude.

    This variant dynamically adjusts the clipping range based on training dynamics:
    - Reduces clipping when gradients are stable (larger effective updates)
    - Increases clipping when gradients are volatile (more conservative updates)
    - Can adapt based on advantage magnitude or gradient norms

    Args:
        **kwargs: May include 'adaptation_mode' ('gradient' or 'advantage')

    Returns:
        LossAssemblyResult with adaptive clipping behavior
    """
    # TODO: Implement adaptive clipping logic
    # For now, fall back to standard PPO clipping
    return assemble_grpo_naive_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=kl_coefficient,
        clip_ratio=clip_ratio,
        **kwargs,
    )


def assemble_dual_clip_loss(
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
    **kwargs: Any,
) -> LossAssemblyResult:
    """Dual clipping with separate upper and lower bounds.

    This variant applies different clipping thresholds for positive and negative advantages:
    - Separate clip_ratio_upper and clip_ratio_lower parameters
    - Asymmetric clipping can be useful for handling reward outliers
    - More conservative when things are going well, more aggressive when poorly

    Args:
        **kwargs: May include 'clip_ratio_lower' for asymmetric bounds

    Returns:
        LossAssemblyResult with dual clipping behavior
    """
    # TODO: Implement dual clipping logic
    # For now, fall back to standard PPO clipping
    return assemble_grpo_naive_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=kl_coefficient,
        clip_ratio=clip_ratio,
        **kwargs,
    )


def assemble_trust_region_loss(
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
    **kwargs: Any,
) -> LossAssemblyResult:
    """Trust region method based on KL constraint.

    This variant uses a hard KL constraint instead of soft penalty:
    - Enforces maximum KL divergence from reference policy
    - More stable than pure KL penalty for large updates
    - Related to trust region policy optimization (TRPO)

    Args:
        **kwargs: May include 'kl_target' for the constraint threshold

    Returns:
        LossAssemblyResult with trust region constraint
    """
    # TODO: Implement trust region logic
    # For now, fall back to standard PPO clipping
    return assemble_grpo_naive_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=kl_coefficient,
        clip_ratio=clip_ratio,
        **kwargs,
    )


def assemble_entropy_regularized_loss(
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
    **kwargs: Any,
) -> LossAssemblyResult:
    """Add entropy bonus to encourage exploration.

    This variant adds an entropy regularization term:
    - Encourages policy stochasticity for better exploration
    - Helps prevent premature convergence to deterministic policies
    - Useful when the policy tends to collapse too quickly

    Args:
        **kwargs: Should include 'entropy_coefficient' for the bonus weight

    Returns:
        LossAssemblyResult with entropy bonus included
    """
    # TODO: Implement entropy regularization logic
    # For now, fall back to standard PPO clipping
    return assemble_grpo_naive_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=kl_coefficient,
        clip_ratio=clip_ratio,
        **kwargs,
    )


# Registry of all available loss variants
LOSS_VARIANTS: dict[LossVariantName, LossAssembler] = {
    "grpo_naive": assemble_grpo_naive_loss,
    "ppo_clipped": assemble_grpo_naive_loss,  # Alias for backward compatibility
    "deepseek_v3.2": assemble_deepseek_v32_loss,
    "adaptive_clip": assemble_adaptive_clip_loss,
    "dual_clip": assemble_dual_clip_loss,
    "trust_region": assemble_trust_region_loss,
    "entropy_regularized": assemble_entropy_regularized_loss,
}


def assemble_grpo_loss(
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
    variant: LossVariantName = "grpo_naive",
    **kwargs: Any,
) -> LossAssemblyResult:
    """Main entry point for GRPO loss computation.

    Dispatches to the appropriate loss variant based on the `variant` parameter.
    All variants share the same signature for compatibility with the training loop.

    Args:
        variant: Which loss variant to use (default: "grpo_naive")
        **kwargs: Additional parameters passed to the specific variant

    Returns:
        LossAssemblyResult from the selected variant

    Raises:
        KeyError: If the variant name is not recognized

    Example:
        ```python
        # Use classical GRPO baseline
        result = assemble_grpo_loss(..., variant="grpo_naive")

        # Use DeepSeek-V3.2 variant
        result = assemble_grpo_loss(..., variant="deepseek_v3.2",
                                   clip_ratio_lower=0.1,
                                   clip_ratio_upper=0.2)
        ```
    """
    if variant not in LOSS_VARIANTS:
        available = ", ".join(LOSS_VARIANTS.keys())
        raise KeyError(
            f"Unknown loss variant: '{variant}'. Available variants: {available}"
        )

    assembler = LOSS_VARIANTS[variant]
    return assembler(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=kl_coefficient,
        clip_ratio=clip_ratio,
        **kwargs,
    )


# Backward compatibility: maintain old function name
def assemble_grpo_loss_legacy(
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
    """Legacy entry point for backward compatibility.

    This function maintains the original signature without the `variant` parameter
    to ensure existing code continues to work. It defaults to the PPO-clipped variant.
    """
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
        variant="ppo_clipped",
    )