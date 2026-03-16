"""GRPO loss calculation variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from typing import Any

    from flashrl.framework.config import GrpoConfig


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
class GrpoPreset:
    """Loss preset configuration."""

    clipping_mode: str
    clip_ratio: float | None = None
    clip_ratio_lower: float | None = None
    clip_ratio_upper: float | None = None
    clip_log_ratio_alpha: float | None = None
    clip_log_ratio_beta: float | None = None
    kl_mode: str = "k3"
    log_ratio_penalty_coefficient: float = 0.0
    enable_icepop_token_gate: bool = False
    icepop_token_gate_beta: float = 2.0
    enable_off_policy_sequence_masking: bool = False
    off_policy_sequence_masking_delta: float = 2.0
    enable_importance_gating: bool = False
    importance_epsilon_low: float | None = None
    importance_epsilon_high: float | None = None
    advantage_normalization: bool = True
    advantage_mode: str = "group_centered"
    entropy_coefficient: float = 0.0


def _check_preset_conflicts(config: GrpoConfig, preset: GrpoPreset) -> list[str]:
    """Check if explicit config parameters conflict with preset values.

    Returns list of conflicting parameter names.
    """
    conflicts = []

    # Check clipping parameters
    if config.clipping_mode != "symmetric" and config.clipping_mode != preset.clipping_mode:
        conflicts.append("clipping_mode")

    if config.clip_ratio != 0.2 and preset.clip_ratio is not None:
        conflicts.append("clip_ratio")

    if config.clip_ratio_lower is not None and preset.clip_ratio_lower is not None:
        if config.clip_ratio_lower != preset.clip_ratio_lower:
            conflicts.append("clip_ratio_lower")

    if config.clip_ratio_upper is not None and preset.clip_ratio_upper is not None:
        if config.clip_ratio_upper != preset.clip_ratio_upper:
            conflicts.append("clip_ratio_upper")

    # Check KL parameters
    if config.kl_mode != "k3" and config.kl_mode != preset.kl_mode:
        conflicts.append("kl_mode")

    if config.log_ratio_penalty_coefficient != 0.0:
        conflicts.append("log_ratio_penalty_coefficient")

    # Check gate parameters
    if config.enable_icepop_token_gate != preset.enable_icepop_token_gate:
        conflicts.append("enable_icepop_token_gate")

    if config.enable_off_policy_sequence_masking != preset.enable_off_policy_sequence_masking:
        conflicts.append("enable_off_policy_sequence_masking")

    if config.enable_importance_gating != preset.enable_importance_gating:
        conflicts.append("enable_importance_gating")

    # Check advantage parameters
    if config.advantage_normalization != preset.advantage_normalization:
        conflicts.append("advantage_normalization")

    if config.advantage_mode != "group_centered" and config.advantage_mode != preset.advantage_mode:
        conflicts.append("advantage_mode")

    # Check entropy parameters
    if config.entropy_coefficient != 0.0 and preset.entropy_coefficient != 0.0:
        if config.entropy_coefficient != preset.entropy_coefficient:
            conflicts.append("entropy_coefficient")

    return conflicts


def _merge_preset_with_config(preset: GrpoPreset, config: GrpoConfig) -> GrpoConfig:
    """Merge preset values with config (preset fills in unspecified parameters).

    This creates a new config with preset values applied where the user hasn't
    explicitly set a value.
    """
    # Convert config to dict for easier manipulation
    config_dict = config.model_dump()

    # Apply preset values for fields that weren't explicitly set
    # (i.e., fields that still have default values)

    # Clipping mode
    if config_dict["clipping_mode"] == "symmetric" and preset.clipping_mode != "symmetric":
        config_dict["clipping_mode"] = preset.clipping_mode

    # Clip ratios
    if preset.clip_ratio is not None and config_dict["clip_ratio"] == 0.2:
        config_dict["clip_ratio"] = preset.clip_ratio

    if (
        preset.clip_ratio_lower is not None
        and config_dict["clip_ratio_lower"] is None
    ):
        config_dict["clip_ratio_lower"] = preset.clip_ratio_lower

    if (
        preset.clip_ratio_upper is not None
        and config_dict["clip_ratio_upper"] is None
    ):
        config_dict["clip_ratio_upper"] = preset.clip_ratio_upper

    # Kimi hard mask parameters
    if (
        preset.clip_log_ratio_alpha is not None
        and config_dict["clip_log_ratio_alpha"] is None
    ):
        config_dict["clip_log_ratio_alpha"] = preset.clip_log_ratio_alpha

    if (
        preset.clip_log_ratio_beta is not None
        and config_dict["clip_log_ratio_beta"] is None
    ):
        config_dict["clip_log_ratio_beta"] = preset.clip_log_ratio_beta

    # KL mode
    if config_dict["kl_mode"] == "k3" and preset.kl_mode != "k3":
        config_dict["kl_mode"] = preset.kl_mode

    # Log-ratio penalty
    if (
        preset.log_ratio_penalty_coefficient != 0.0
        and config_dict["log_ratio_penalty_coefficient"] == 0.0
    ):
        config_dict["log_ratio_penalty_coefficient"] = (
            preset.log_ratio_penalty_coefficient
        )

    # Gates
    if (
        preset.enable_icepop_token_gate
        and not config_dict["enable_icepop_token_gate"]
    ):
        config_dict["enable_icepop_token_gate"] = True
        config_dict["icepop_token_gate_beta"] = preset.icepop_token_gate_beta

    if (
        preset.enable_off_policy_sequence_masking
        and not config_dict["enable_off_policy_sequence_masking"]
    ):
        config_dict["enable_off_policy_sequence_masking"] = True
        config_dict["off_policy_sequence_masking_delta"] = preset.off_policy_sequence_masking_delta

    if preset.enable_importance_gating and not config_dict["enable_importance_gating"]:
        config_dict["enable_importance_gating"] = True
        config_dict["importance_epsilon_low"] = preset.importance_epsilon_low
        config_dict["importance_epsilon_high"] = preset.importance_epsilon_high

    # Advantage mode
    if (
        preset.advantage_mode != "group_centered"
        and config_dict["advantage_mode"] == "group_centered"
    ):
        config_dict["advantage_mode"] = preset.advantage_mode

    # Entropy
    if (
        preset.entropy_coefficient != 0.0
        and config_dict["entropy_coefficient"] == 0.0
    ):
        config_dict["entropy_coefficient"] = preset.entropy_coefficient

    # Reconstruct config from dict
    from flashrl.framework.config import GrpoConfig

    return GrpoConfig(**config_dict)


def resolve_loss_preset(config: GrpoConfig) -> GrpoConfig:
    """Resolve loss preset to explicit configuration parameters.

    Args:
        config: GrpoConfig with loss_preset set

    Returns:
        Resolved GrpoConfig with preset parameters applied

    Raises:
        ValueError: If preset conflicts with explicit parameters
    """
    if config.loss_preset == "custom":
        return config

    # Define presets based on research papers
    presets = {
        "grpo_naive": GrpoPreset(
            clipping_mode="symmetric",
            clip_ratio=0.2,
            kl_mode="k3",
            advantage_normalization=True,
            advantage_mode="group_centered",
        ),
        "ppo_clipped": GrpoPreset(  # Alias for grpo_naive
            clipping_mode="symmetric",
            clip_ratio=0.2,
            kl_mode="k3",
            advantage_normalization=True,
            advantage_mode="group_centered",
        ),
        "deepseek_v3.2": GrpoPreset(
            clipping_mode="symmetric",
            clip_ratio=0.2,
            kl_mode="unbiased",  # Unbiased token-level KL estimator
            enable_off_policy_sequence_masking=True,
            off_policy_sequence_masking_delta=2.0,
            advantage_normalization=True,
            advantage_mode="group_centered",
            entropy_coefficient=0.01,
        ),
        "kimi_k2.5": GrpoPreset(
            clipping_mode="hard_mask",
            clip_log_ratio_alpha=-5.0,  # Typical values from paper
            clip_log_ratio_beta=5.0,
            kl_mode="none",  # No explicit reference KL
            log_ratio_penalty_coefficient=0.01,  # τ for soft penalty
            advantage_normalization=True,
            advantage_mode="group_centered",
        ),
        "glm_5": GrpoPreset(
            clipping_mode="asymmetric",
            clip_ratio_lower=0.1,
            clip_ratio_upper=0.2,
            kl_mode="none",  # No explicit KL in reasoning RL backbone
            enable_icepop_token_gate=True,
            icepop_token_gate_beta=2.0,
            advantage_normalization=True,
            advantage_mode="group_normalized",  # GLM uses group-normalized
        ),
        "mimo_v2": GrpoPreset(
            clipping_mode="none",  # Uses importance gating instead
            kl_mode="none",
            enable_importance_gating=True,
            importance_epsilon_low=1.0 - 0.2,
            importance_epsilon_high=1.0 + 0.2,
            advantage_normalization=False,  # Teacher provides advantage
        ),
    }

    if config.loss_preset not in presets:
        raise ValueError(f"Unknown preset: {config.loss_preset}")

    preset = presets[config.loss_preset]

    # Check for conflicts with explicit parameters
    conflicts = _check_preset_conflicts(config, preset)
    if conflicts:
        conflict_list = ", ".join(conflicts)
        raise ValueError(
            f"Parameters [{conflict_list}] conflict with preset '{config.loss_preset}'. "
            f"Use loss_preset='custom' to override presets."
        )

    # Merge preset with config (preset values fill in unspecified parameters)
    return _merge_preset_with_config(preset, config)


def get_current_policy_log_probs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    actor_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute current policy log probabilities log π_θ(a_t) from actor logits.

    MATH: log π_θ(a_t) = log_softmax(logits_θ)[a_t]

    During training, actor parameters θ have changed since rollout, so we MUST
    compute current policy log probabilities from the new actor logits.

    Uses F.cross_entropy which is memory-efficient - only computes log probs
    for target tokens, not full vocabulary (reduces memory from O(vocab) to O(1)).

    RL CONTEXT: In GRPO, we need TWO different log probabilities:
    - log π_old(a): Old policy (from rollout, reused directly)
    - log π_θ(a): Current policy (computed here, from updated actor)

    Args:
        input_ids: Input token IDs (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len)
        actor_logits: Actor logits from forward pass (batch, seq_len, vocab_size)

    Returns:
        log_pi_theta: Current policy log probabilities (batch, seq_len-1)
        shift_ids: Shifted token IDs (batch, seq_len-1)
        shift_mask: Shifted attention mask (batch, seq_len-1)
    """
    shift_ids = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()

    # Use cross_entropy which only computes log prob for target tokens (optimized!)
    # cross_entropy returns negative log likelihood, so we negate to get log probs
    # Reshape: (batch, seq_len-1, vocab_size) -> (batch * (seq_len-1), vocab_size)
    logits_flat = actor_logits[:, :-1, :].reshape(-1, actor_logits.shape[-1])
    ids_flat = shift_ids.reshape(-1)

    # Compute negative log probs for target tokens only
    neg_log_probs = F.cross_entropy(
        logits_flat,
        ids_flat,
        reduction='none',
    )

    # Reshape back and negate to get actual log probabilities
    log_pi_theta = -neg_log_probs.reshape(shift_ids.shape)

    return log_pi_theta, shift_ids, shift_mask


def _build_response_mask(
    shift_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Build response mask from prompt lengths (vectorized).

    Returns:
        Tuple of (response_mask, response_token_count, response_tokens_total, response_mask_bool)
    """
    # Vectorized mask construction
    prompt_starts = (prompt_lengths - 1).clamp(min=0)
    indices = torch.arange(shift_mask.shape[1], device=shift_mask.device)
    response_mask = (indices.unsqueeze(0) >= prompt_starts.unsqueeze(1)) & shift_mask.bool()
    response_mask = response_mask.float()

    response_sum = response_mask.sum()
    response_token_count = response_sum.clamp(min=1)
    response_tokens_total = int(response_sum.item())
    return response_mask, response_token_count, response_tokens_total, response_mask.bool()


def _load_rollout_log_probs(
    rollout_response_log_probs: list[list[float]],
    response_mask_bool: torch.Tensor,
    log_pi_theta: torch.Tensor,
) -> torch.Tensor:
    """Load old policy log probabilities from rollout (already computed by serving engine).

    IMPORTANT: These log probabilities were already computed during rollout by
    the serving engine (vLLM or HuggingFace). We do NOT recompute them here.

    Args:
        rollout_response_log_probs: Old policy log probs from rollout (computed once)
        response_mask_bool: Boolean mask for response tokens
        log_pi_theta: Current policy log probs (computed during training, for dtype/device)

    Returns:
        Old policy log probabilities as tensor (same shape/device as current policy)
    """
    if len(rollout_response_log_probs) != response_mask_bool.shape[0]:
        raise ValueError("rollout_response_log_probs must match the batch size.")

    # Cache dtype and device
    dtype = log_pi_theta.dtype
    device = log_pi_theta.device

    log_pi_old = torch.zeros(response_mask_bool.shape, dtype=dtype, device=device)

    for index, sample_log_probs in enumerate(rollout_response_log_probs):
        expected_tokens = response_mask_bool[index].sum()
        if len(sample_log_probs) != expected_tokens:
            raise ValueError(
                "Each rollout_response_log_probs entry must match that sample's "
                "response-token count."
            )
        if expected_tokens == 0:
            continue
        log_pi_old[index, response_mask_bool[index]] = torch.as_tensor(
            sample_log_probs, dtype=dtype, device=device
        )

    return log_pi_old


def _compute_kl_divergence(
    ref_logits: torch.Tensor | None,
    shift_ids: torch.Tensor,
    log_pi_theta: torch.Tensor,
    response_mask: torch.Tensor,
    response_token_count: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between reference and actor policies."""
    if ref_logits is None:
        return torch.tensor(0.0, device=log_pi_theta.device, dtype=log_pi_theta.dtype)

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
    response_mask_bool: torch.Tensor,
    clip_lower: torch.Tensor | float,
    clip_upper: torch.Tensor | float,
) -> dict[str, float]:
    """Compute importance sampling ratio statistics for monitoring.

    MATH: ρ_t = π_θ(a_t) / π_old(a_t) = exp(log π_θ(a_t) - log π_old(a_t))

    RL CONTEXT: The importance sampling ratio measures how much the policy has
    changed since rollout. Key statistics:
    - mean: Average ratio (should be ~1.0 for stable training)
    - clip_fraction: Percentage of tokens that were clipped (indicates policy drift)
    - min/max: Extremes that might indicate training instability

    CLIPPING REFERENCE: In PPO/GRPO, ratios are clipped to prevent large updates:
    - clip_fraction shows how often clipping occurred
    - High clip_fraction (>20%) suggests policy is changing too fast

    Args:
        ratio: Importance sampling ratios ρ = π_θ/π_old
        response_mask_bool: Boolean mask for response tokens
        clip_lower: Lower clipping bound (e.g., 0.8 for ε=0.2)
        clip_upper: Upper clipping bound (e.g., 1.2 for ε=0.2)

    Returns:
        Dictionary with mean, std, min, max, clip_fraction
    """
    ratio_values = ratio[response_mask_bool]

    if ratio_values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "clip_fraction": 0.0,
        }

    # Use torch.aminmax for more efficient min/max computation
    min_val, max_val = torch.aminmax(ratio_values)
    clip_fraction = (
        ((ratio_values < clip_lower) | (ratio_values > clip_upper)).float().mean().item()
    )

    return {
        "mean": float(ratio_values.mean().item()),
        "std": float(ratio_values.std(unbiased=False).item()),
        "min": float(min_val.item()),
        "max": float(max_val.item()),
        "clip_fraction": float(clip_fraction),
    }


def _apply_clipping(
    ratio: torch.Tensor,
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Apply clipping to importance sampling ratio based on configured mode.

    MATH:
    - Symmetric (PPO): clip(ρ, 1-ε, 1+ε)
    - Asymmetric (DeepSeek): advantage-dependent bounds
    - Hard mask (Kimi): ρ * I[α ≤ log(ρ) ≤ β]

    RL CONTEXT: Clipping prevents large policy updates that could degrade performance.
    The key insight is to create a "pessimistic bound" on the policy update.

    CLIPPING MODES:

    1. SYMMETRIC (PPO-style):
       clip(ρ, 1-ε, 1+ε)
       When A > 0: Don't increase ρ above 1+ε (prevent over-optimization)
       When A < 0: Don't decrease ρ below 1-ε (prevent over-penalization)

    2. ASYMMETRIC (DeepSeek-V3.2):
       Tighter bounds when A > 0 (conservative on good actions)
       Wider bounds when A < 0 (allow more correction on bad actions)

    3. HARD MASK (Kimi K2.5):
       Zero out tokens where log-ratio is outside [α, β]
       This creates a hard gradient mask instead of soft clipping

    Args:
        ratio: Importance sampling ratio ρ = π_θ/π_old
        log_ratio: Log ratio log(π_θ/π_old) for hard mask
        advantages: Advantage estimates A (for asymmetric clipping)
        response_mask: Response token mask
        config: GRPO configuration with clipping parameters

    Returns:
        Clipped ratio (or masked ratio for hard_mask mode)
    """
    if config.clipping_mode == "none":
        return ratio

    if config.clipping_mode == "symmetric":
        # PPO-style symmetric clipping
        clip_lower = 1.0 - config.clip_ratio
        clip_upper = 1.0 + config.clip_ratio
        return torch.clamp(ratio, clip_lower, clip_upper)

    if config.clipping_mode == "asymmetric":
        # DeepSeek-V3.2 style: advantage-dependent clipping
        expanded_advantages = advantages.unsqueeze(-1).expand_as(response_mask)
        clip_lower = torch.where(
            expanded_advantages > 0,
            1.0 - (config.clip_ratio_lower or 0.1),
            1.0 - (config.clip_ratio_upper or config.clip_ratio),
        )
        clip_upper = torch.where(
            expanded_advantages > 0,
            1.0 + (config.clip_ratio_upper or config.clip_ratio),
            1.0 + (config.clip_ratio_lower or 0.1),
        )
        return torch.clamp(ratio, clip_lower, clip_upper)

    if config.clipping_mode == "hard_mask":
        # Kimi K2.5 style: hard token-level gradient mask
        alpha = config.clip_log_ratio_alpha or -5.0
        beta = config.clip_log_ratio_beta or 5.0

        # Mask out tokens outside log-ratio band
        in_band = (log_ratio >= alpha) & (log_ratio <= beta)
        return ratio * in_band.float()

    return ratio


def _get_clip_bounds(config: GrpoConfig) -> tuple[float, float]:
    """Get clip lower/upper bounds based on clipping mode.

    Returns (clip_lower, clip_upper) tuple for statistics computation.
    For asymmetric mode, returns approximate bounds (not advantage-dependent).
    """
    if config.clipping_mode == "symmetric":
        return 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
    elif config.clipping_mode == "asymmetric":
        # Return approximate bounds for statistics (actual clipping is advantage-dependent)
        return 1.0 - (config.clip_ratio_lower or 0.1), 1.0 + (config.clip_ratio_upper or config.clip_ratio)
    elif config.clipping_mode == "hard_mask":
        # For hard_mask, bounds are log-ratio values, not ratio values
        return config.clip_log_ratio_alpha or -5.0, config.clip_log_ratio_beta or 5.0
    else:
        return 0.0, 1.0


def _compute_off_policy_sequence_masking(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask_bool: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Compute sequence-level off-policy mask (DeepSeek-V3.2 technique).

    MATH:
    For each sequence i:
    M_i = 0 if A_i < 0 AND (1/|o_i|) * sum_{t∈o_i} log(π_old/π_θ) > δ
         1 otherwise

    Then broadcast M_i to all tokens in the sequence.

    RL CONTEXT (DeepSeek-V3.2): This masks "stale negatives" - sequences where:
    1. Advantage is negative (A_i < 0): The response was bad
    2. Policy has moved away (log-ratio > δ): Actor no longer generates this

    INTUITION: If the policy has already moved away from a bad response,
    further training on it is counterproductive (gradient is "stale").

    SEQUENCE-LEVEL: Unlike token-level clipping, this masks ALL tokens in
    a response sequence if the condition is met. This is a stronger form
    of rejection that prevents training on already-corrected behaviors.

    Args:
        log_ratio: Log ratio log(π_θ/π_old) for each token
        advantages: Advantage estimates for each sequence
        response_mask_bool: Boolean mask for response tokens
        config: GRPO config with off_policy_sequence_masking_delta parameter

    Returns:
        Sequence mask (batch_size,) broadcast to all tokens
    """
    # Compute sequence-averaged old-vs-current log-ratio for each response (vectorized)
    # log_ratio is log(π_θ) - log(π_old), so we negate it
    masked_log_ratio = log_ratio * response_mask_bool.float()
    sequence_sums = masked_log_ratio.sum(dim=1)
    sequence_counts = response_mask_bool.sum(dim=1).clamp(min=1)
    sequence_log_ratio_tensor = -(sequence_sums / sequence_counts)
    sequence_log_ratio_tensor[sequence_counts == 1] = 0.0  # Handle empty masks

    # Check condition: advantage < 0 AND sequence_log_ratio > delta
    delta = config.off_policy_sequence_masking_delta
    is_stale_negative = (advantages < 0) & (sequence_log_ratio_tensor > delta)

    # Create sequence mask: 0 for stale negatives, 1 otherwise
    sequence_mask = (~is_stale_negative).float()

    return sequence_mask


def _apply_icepop_token_gate(
    ratio_train: torch.Tensor,          # r = π^train / π^train_old
    log_pi_train_old: torch.Tensor,     # π^train_old
    log_pi_infer_old: torch.Tensor,     # π^infer_old
    config: GrpoConfig,
) -> torch.Tensor:
    """Apply GLM-5 IcePop train/infer mismatch gate (per-token).

    MATH:
    ρ = π^train_old / π^infer_old  (train/infer mismatch)
    pop(ρ, 1/β, β) = 0 if ρ < 1/β or ρ > β
                  = ρ otherwise
    r_gated = r * pop(ρ, 1/β, β)

    where:
    - r = π^train / π^train_old (standard GRPO ratio)
    - ρ = π^train_old / π^infer_old (train/infer mismatch)

    RL CONTEXT (GLM-5 IcePop): During training, there can be a mismatch between
    the train-time policy (π^train) and inference-time policy (π^infer) due to:
    - Different sampling strategies (e.g., temperature, top-k)
    - Different model weights (train vs infer checkpoints)
    - Different computational graphs (e.g., dropout enabled during training)

    INTUITION: If train/infer mismatch is too large (ρ outside [1/β, β]),
    the gradient is unreliable and should be discarded. This prevents training
    on samples that don't reflect actual inference behavior.

    Args:
        ratio_train: Standard GRPO ratio r = π^train / π^train_old
        log_pi_train_old: Log probs from training backend (π^train_old)
        log_pi_infer_old: Log probs from inference engine (π^infer_old)
        config: GRPO config with icepop_token_gate_beta parameter

    Returns:
        Gated ratio r_gated = r * pop(ρ, 1/β, β)
    """
    if log_pi_train_old is None:
        # No π^train_old provided, skip gate
        return ratio_train

    beta = config.icepop_token_gate_beta

    # Compute mismatch ratio: ρ = π^train_old / π^infer_old
    # Use log-space arithmetic: log(ρ) = log(π^train_old) - log(π^infer_old)
    log_mismatch = log_pi_train_old - log_pi_infer_old
    mismatch_ratio = torch.exp(log_mismatch)  # ρ

    # Apply pop() function: pop(ρ, 1/β, β)
    lower_bound = 1.0 / beta
    upper_bound = beta
    in_range = (mismatch_ratio >= lower_bound) & (mismatch_ratio <= upper_bound)

    # Gate the training ratio: r_gated = r * I[ρ in [1/β, β]]
    gated_ratio = ratio_train * in_range.float()

    return gated_ratio


def _apply_importance_gating(
    ratio: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Apply MiMo-V2-Flash importance weight gating.

    MATH:
    w(ρ) = 0 if ρ outside [ε_low, ε_high]
         = ρ otherwise

    where ρ = π_θ/μ_θ and μ_θ is the sampling policy

    RL CONTEXT (MiMo-V2-Flash): This gates tokens based on whether the
    importance sampling ratio falls within an acceptable band. Unlike clipping
    (which bounds the ratio), this is a HARD gate that zeros out gradients.

    INTUITION: Tokens with extreme ratios (outside [ε_low, ε_high]) are
    considered "outliers" that should not contribute to the gradient. This
    is similar to robust statistics - removing outliers before computing
    the gradient update.

    COMPARISON TO CLIPPING:
    - Clipping: ρ_clipped = clamp(ρ, ε_low, ε_high) - soft constraint
    - Gating: w(ρ) = ρ * I[ε_low ≤ ρ ≤ ε_high] - hard constraint

    Gating is more aggressive than clipping - it completely removes outlier
    tokens rather than just bounding their contribution.

    Args:
        ratio: Importance sampling ratio ρ = π_θ/π_old
        config: GRPO config with importance_epsilon_low/high parameters

    Returns:
        Gated ratio (zeroed if outside acceptable band)
    """
    epsilon_low = config.importance_epsilon_low or 0.8
    epsilon_high = config.importance_epsilon_high or 1.2

    # Check if ratio is within band
    in_band = (ratio >= epsilon_low) & (ratio <= epsilon_high)

    # Apply hard gate
    return ratio * in_band.float()


def _compute_kl_divergence_enhanced(
    ref_logits: torch.Tensor | None,
    shift_ids: torch.Tensor,
    log_pi_theta: torch.Tensor,
    log_pi_old: torch.Tensor,
    response_mask: torch.Tensor,
    response_token_count: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Compute KL divergence KL(π_θ || π_ref) in specified mode.

    MATH (by mode):

    1. K1 (Simple KL - aggregate once):
       KL(π_θ || π_ref) = sum_t π_θ(t) [log π_θ(t) - log π_ref(t)]
       Applied once at the end with optional hard threshold

    2. K3 (Per-token KL - standard):
       Same as K1 but computed per-token then averaged
       This is the standard implementation in most PPO/GRPO code

    3. UNBIASED (DeepSeek-V3.2 estimator):
       KL(π_θ || π_ref) ≈ (π_θ/π_old) * [(π_ref/π_θ - log(π_ref/π_θ) - 1)]
       Uses importance sampling to reduce variance

    RL CONTEXT: KL divergence measures how much the policy has strayed from
    the reference policy. In PPO/GRPO, we use it as a regularization term to
    prevent the policy from changing too much in a single update.

    REFERENCE POLICY: π_ref is typically a frozen copy of the actor at the
    start of training (or periodically updated). Using π_ref instead of π_old
    provides a more stable baseline for KL regularization.

    Args:
        ref_logits: Reference model logits (for computing π_ref)
        shift_ids: Token IDs for indexing
        log_pi_theta: Current policy log probabilities log π_θ
        log_pi_old: Old policy log probabilities log π_old
        response_mask: Response token mask
        response_token_count: Response token count (for normalization)
        config: GRPO config with kl_mode and kl_hard_threshold

    Returns:
        KL divergence (scalar tensor)
    """
    if config.kl_mode == "none" or ref_logits is None:
        return torch.tensor(0.0, device=log_pi_theta.device, dtype=log_pi_theta.dtype)

    reference_token_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
    log_pi_ref = torch.gather(
        reference_token_log_probs,
        dim=-1,
        index=shift_ids.unsqueeze(-1),
    ).squeeze(-1)

    if config.kl_mode == "k1":
        # Simple KL: aggregate once at the end
        reference_log_gap = log_pi_ref - log_pi_theta
        kl_terms = torch.exp(reference_log_gap) - reference_log_gap - 1.0
        masked_kl = kl_terms * response_mask
        kl_divergence = masked_kl.sum() / response_token_count

        # Apply hard threshold if configured
        if config.kl_hard_threshold is not None:
            if kl_divergence.item() > config.kl_hard_threshold:
                raise ValueError(
                    f"KL divergence {kl_divergence.item():.4f} exceeds "
                    f"threshold {config.kl_hard_threshold}"
                )

        return kl_divergence

    elif config.kl_mode == "k3":
        # Per-token KL (standard implementation)
        reference_log_gap = log_pi_ref - log_pi_theta
        kl_terms = torch.exp(reference_log_gap) - reference_log_gap - 1.0
        masked_kl = kl_terms * response_mask
        return masked_kl.sum() / response_token_count

    elif config.kl_mode == "unbiased":
        # DeepSeek-V3.2 unbiased token-level KL estimator
        # KL(π_θ || π_ref) ≈ (π_θ/π_old) * [(π_ref/π_θ - log(π_ref/π_θ) - 1)]
        ratio = torch.exp(log_pi_theta - log_pi_old)
        reference_log_gap = log_pi_ref - log_pi_theta
        kl_terms = ratio * (torch.exp(reference_log_gap) - reference_log_gap - 1.0)
        masked_kl = kl_terms * response_mask
        return masked_kl.sum() / response_token_count

    raise ValueError(f"Unknown KL mode: {config.kl_mode}")


def _compute_log_ratio_penalty(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    response_token_count: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Compute soft quadratic penalty on log-ratio (Kimi K2.5 technique).

    MATH:
    L_penalty = τ * [log(π_θ/π_old)]^2

    where τ is the penalty coefficient (typically 0.01)

    RL CONTEXT (Kimi K2.5): This is a soft regularization alternative to
    hard clipping. Instead of abruptly zeroing out gradients (hard mask),
    this applies a quadratic penalty that increases smoothly with log-ratio.

    INTUITION: Large log-ratios indicate the policy has changed significantly
    from the old policy. The quadratic penalty discourages extreme changes
    while still allowing gradual policy improvement.

    COMPARISON:
    - Hard mask (Kimi hard_mask): Abrupt gradient cutoff at [α, β] boundaries
    - Soft penalty (Kimi log-ratio): Smooth quadratic penalty on log-ratio

    The soft penalty is more "gradient-friendly" - it doesn't create sharp
    discontinuities in the loss landscape.

    Args:
        log_ratio: Log ratio log(π_θ/π_old) for each token
        response_mask: Response token mask
        response_token_count: Response token count (for normalization)
        config: GRPO config with log_ratio_penalty_coefficient parameter

    Returns:
        Quadratic penalty loss (scalar tensor)
    """
    if config.log_ratio_penalty_coefficient == 0.0:
        return torch.tensor(0.0, device=log_ratio.device, dtype=log_ratio.dtype)

    # Squared log-ratio penalty
    squared_penalty = (log_ratio**2) * response_mask
    penalty_mean = squared_penalty.sum() / response_token_count

    return config.log_ratio_penalty_coefficient * penalty_mean


def _compute_surrogate_loss(
    ratio: torch.Tensor,
    clipped_ratio: torch.Tensor,
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    response_token_count: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Compute PPO/GRPO clipped surrogate loss objective.

    MATH:
    L^C(θ) = -E[min(ρ·A, clip(ρ)·A)]

    where:
    - ρ = π_θ(a) / π_old(a) is the importance sampling ratio
    - A is the advantage estimate
    - clip(ρ) depends on clipping mode (symmetric, asymmetric, etc.)

    RL CONTEXT: This is the core PPO/GRPO objective. The key insight is the
    min() operator, which creates a "pessimistic bound" on the policy update.

    WHY MIN(ρ·A, clip(ρ)·A)?

    When A > 0 (good action):
    - We want to increase ρ (increase probability of this action)
    - But clip(ρ) prevents over-optimization (stops at upper bound)
    - min() chooses the smaller value → conservative update

    When A < 0 (bad action):
    - We want to decrease ρ (decrease probability of this action)
    - But clip(ρ) prevents over-penalization (stops at lower bound)
    - min() chooses the smaller value → conservative update

    INTUITION: The clipped objective prevents the policy from changing too
    much in a single update, which stabilizes training and prevents collapse.

    SPECIAL CASES:
    - hard_mask mode: Uses clipped_ratio directly (already masked)
    - none mode: No clipping, just ratio·A (unstable, not recommended)

    Args:
        ratio: Unclipped importance sampling ratio ρ
        clipped_ratio: Clipped ratio (or masked ratio for hard_mask)
        log_ratio: Log ratio (not used directly, for API consistency)
        advantages: Advantage estimates A
        response_mask: Response token mask
        response_token_count: Response token count (for normalization)
        config: GRPO config with clipping parameters

    Returns:
        Surrogate loss (scalar tensor, negative because we maximize)
    """
    expanded_advantages = advantages.unsqueeze(-1).expand_as(response_mask)

    if config.clipping_mode == "hard_mask":
        # Kimi-style: already masked in clipped_ratio, use directly
        surrogate_objective = clipped_ratio * expanded_advantages
    elif config.clipping_mode == "none":
        surrogate_objective = ratio * expanded_advantages
    else:
        # PPO-style: min of clipped and unclipped
        surrogate_unclipped = ratio * expanded_advantages
        surrogate_clipped = clipped_ratio * expanded_advantages
        surrogate_objective = torch.minimum(surrogate_unclipped, surrogate_clipped)

    masked_surrogate = surrogate_objective * response_mask
    return -masked_surrogate.sum() / response_token_count


def _process_advantages(
    advantages: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Process advantages with normalization."""
    if not config.advantage_normalization:
        return advantages

    if config.advantage_mode == "group_centered":
        # DeepSeek-style: center by group mean
        # Advantages are already centered at group level
        return advantages

    elif config.advantage_mode == "group_normalized":
        # GLM-style: center and normalize by group std
        mean = advantages.mean()
        std = advantages.std()
        if std > 1e-8:
            return (advantages - mean) / std
        return advantages

    return advantages


def _compute_entropy_loss(
    actor_logits: torch.Tensor,
    response_mask: torch.Tensor,
    response_token_count: torch.Tensor,
    config: GrpoConfig,
) -> torch.Tensor:
    """Compute entropy regularization loss."""
    if config.entropy_coefficient == 0.0:
        return torch.tensor(0.0, device=actor_logits.device, dtype=actor_logits.dtype)

    # Compute entropy from logits
    probs = F.softmax(actor_logits[:, :-1, :], dim=-1)
    log_probs = F.log_softmax(actor_logits[:, :-1, :], dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    # Apply response mask
    masked_entropy = entropy * response_mask
    entropy_mean = masked_entropy.sum() / response_token_count

    # Negative because we want to maximize entropy
    return -config.entropy_coefficient * entropy_mean




def assemble_grpo_loss(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    actor_logits: torch.Tensor,
    ref_logits: torch.Tensor | None,
    rollout_response_log_probs: list[list[float]],  # π^infer_old
    training_response_log_probs: list[list[float]] | None = None,  # π^train_old
    advantages: torch.Tensor,
    config: GrpoConfig,
) -> LossAssemblyResult:
    """Assemble full GRPO loss with all components (surrogate + KL + entropy + penalties).

    OVERALL MATH:
    L(θ) = L^C(θ) + β_KL·KL(π_θ || π_ref) + L_entropy + L_penalty

    where:
    - L^C(θ): Clipped surrogate loss (PPO/GRPO objective)
    - KL(π_θ || π_ref): KL divergence regularization (prevents policy drift)
    - L_entropy: Entropy bonus (encourages exploration)
    - L_penalty: Additional penalties (log-ratio penalty, etc.)

    RL CONTEXT (GRPO Algorithm):
    GRPO (Group Relative Policy Optimization) extends PPO with group-based
    advantage estimation and supports multiple advanced techniques:

    1. DEEPSEEK-V3.2: Asymmetric clipping + stale negative masking + unbiased KL
    2. GLM-5: Train/infer mismatch gate + group-normalized advantages
    3. KIMI K2.5: Hard log-ratio masking + soft quadratic penalty
    4. MIMO-V2: Importance weight gating (no explicit clipping)

    DATA FLOW (CRITICAL):

    Two different log probability sources (DON'T confuse them!):

    1. log_pi_old = log π_old(a): OLD policy (from rollout)
       - Source: rollout_response_log_probs (computed by vLLM/HuggingFace)
       - Status: Already computed, reused directly
       - Purpose: Baseline for importance sampling ratio

    2. log_pi_theta = log π_θ(a): CURRENT policy (from training)
       - Source: actor_logits (computed from updated actor)
       - Status: Must compute during training (actor changed)
       - Purpose: Current policy we're optimizing

    KEY COMPUTATIONS:
    1. Importance sampling ratio: ρ = π_θ(a) / π_old(a) = exp(log π_θ - log π_old)
    2. Apply clipping/gating to ρ based on configured mode
    3. Compute surrogate loss using clipped ratio and advantages
    4. Add KL divergence, entropy, and other penalties
    5. Return total loss + statistics for monitoring

    Args:
        input_ids: Input token IDs (prompt + response)
        attention_mask: Attention mask
        prompt_lengths: Length of prompts (for masking)
        actor_logits: Current actor logits (for computing log π_θ)
        ref_logits: Reference model logits (for computing KL divergence)
        rollout_response_log_probs: Old policy log probs from inference engine (π^infer_old)
        training_response_log_probs: Old policy log probs from training backend (π^train_old)
                                     Required for GLM-5 IcePop gate, optional for other presets
        advantages: Advantage estimates A (computed from rewards)
        config: GRPO configuration (clipping mode, KL mode, penalties, etc.)

    Returns:
        LossAssemblyResult with:
        - loss: Total loss L(θ)
        - policy_loss: Surrogate + entropy + penalty (no KL)
        - kl_divergence: KL divergence (for monitoring)
        - response_tokens_total: Total response tokens
        - importance_sampling_ratio_*: Statistics (mean, std, min, max)
        - clip_fraction: Percentage of tokens clipped

    Raises:
        ValueError: If parameters conflict with preset
    """
    # Resolve presets (cached for efficiency)
    resolved_config = config.get_resolved_config()

    # Common preprocessing
    # Note: log_pi_theta (current policy) must be computed during training
    #       log_pi_old (old policy) comes from rollout and is already computed
    log_pi_theta, shift_ids, shift_mask = get_current_policy_log_probs(
        input_ids, attention_mask, actor_logits
    )
    response_mask, response_token_count, response_tokens_total, response_mask_bool = (
        _build_response_mask(shift_mask, prompt_lengths)
    )
    log_pi_old = _load_rollout_log_probs(
        rollout_response_log_probs, response_mask_bool, log_pi_theta
    )

    # Load π^train_old for IcePop gate (if provided and gate is enabled)
    # Only load when needed to avoid unnecessary tensor allocation
    log_pi_train_old: torch.Tensor | None = None
    if training_response_log_probs is not None and resolved_config.enable_icepop_token_gate:
        log_pi_train_old = _load_rollout_log_probs(
            training_response_log_probs, response_mask_bool, log_pi_theta
        )

    # Compute log ratio for various masking/penalty operations
    log_ratio = log_pi_theta - log_pi_old

    # Compute importance sampling ratio
    ratio = torch.exp(log_ratio)

    # Process advantages (normalization)
    processed_advantages = _process_advantages(advantages, resolved_config)

    # Apply sequence-level off-policy masking (DeepSeek-V3.2)
    if resolved_config.enable_off_policy_sequence_masking:
        sequence_mask = _compute_off_policy_sequence_masking(
            log_ratio, processed_advantages, response_mask_bool, resolved_config
        )
        response_mask = response_mask * sequence_mask.unsqueeze(-1)

    # Apply train/infer mismatch gate (GLM-5 IcePop-style, per-token)
    if resolved_config.enable_icepop_token_gate:
        ratio = _apply_icepop_token_gate(
            ratio_train=ratio,              # r = π^train / π^train_old
            log_pi_train_old=log_pi_train_old,  # π^train_old
            log_pi_infer_old=log_pi_old,         # π^infer_old
            config=resolved_config,
        )

    # Apply importance weight gating (MiMo-style)
    if resolved_config.enable_importance_gating:
        ratio = _apply_importance_gating(ratio, resolved_config)

    # Apply clipping based on mode
    clipped_ratio = _apply_clipping(
        ratio, log_ratio, processed_advantages, response_mask, resolved_config
    )

    # Compute surrogate loss
    policy_loss = _compute_surrogate_loss(
        ratio,
        clipped_ratio,
        log_ratio,
        processed_advantages,
        response_mask,
        response_token_count,
        resolved_config,
    )

    # Compute KL divergence
    kl_divergence = _compute_kl_divergence_enhanced(
        ref_logits,
        shift_ids,
        log_pi_theta,
        log_pi_old,
        response_mask,
        response_token_count,
        resolved_config,
    )

    # Compute log-ratio penalty (Kimi-style)
    log_ratio_penalty = _compute_log_ratio_penalty(
        log_ratio, response_mask, response_token_count, resolved_config
    )

    # Compute entropy bonus
    entropy_loss = _compute_entropy_loss(
        actor_logits, response_mask, response_token_count, resolved_config
    )

    # Assemble final loss
    total_loss = (
        policy_loss
        + entropy_loss
        + log_ratio_penalty
        + resolved_config.kl_coefficient * kl_divergence
    )

    # Compute statistics
    clip_lower, clip_upper = _get_clip_bounds(resolved_config)

    stats = _compute_ratio_statistics(
        ratio,
        response_mask_bool,
        clip_lower,
        clip_upper,
    )

    return LossAssemblyResult(
        loss=total_loss,
        policy_loss=policy_loss + entropy_loss + log_ratio_penalty,
        kl_divergence=kl_divergence,
        response_tokens_total=response_tokens_total,
        importance_sampling_ratio_mean=stats["mean"],
        importance_sampling_ratio_std=stats["std"],
        importance_sampling_ratio_min=stats["min"],
        importance_sampling_ratio_max=stats["max"],
        clip_fraction=stats["clip_fraction"],
    )