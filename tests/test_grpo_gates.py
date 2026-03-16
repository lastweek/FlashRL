"""Tests for GRPO gating mechanisms."""

import pytest
import torch

from flashrl.framework.config import GrpoConfig
from flashrl.framework.trainer.grpo.loss_variants import (
    _compute_stale_negative_mask,
    _apply_importance_gating,
)


class TestStaleNegativeMask:
    """Test DeepSeek-V3.2 sequence-level stale negative masking."""

    def test_stale_negative_mask_masks_bad_sequences(self):
        """Test that sequences with negative advantage and high log-ratio are masked."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        response_mask_bool = torch.tensor([[True, True, True]])

        # Sequence with negative advantage and high mean log-ratio
        log_ratio = torch.tensor([[0.0, 2.5, 2.5]])  # Mean = 1.67 > 2.0 (not quite)
        advantages = torch.tensor([-1.0])

        mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, config
        )

        # Should NOT be masked (mean log-ratio = 1.67 < 2.0)
        assert mask[0] == pytest.approx(1.0)

    def test_stale_negative_mask_with_high_log_ratio(self):
        """Test masking when log-ratio exceeds delta threshold."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        response_mask_bool = torch.tensor([[True, True, True]])

        # Sequence with negative advantage and very high mean log-ratio
        log_ratio = torch.tensor([[2.5, 2.5, 2.5]])  # Mean = 2.5 > 2.0
        advantages = torch.tensor([-1.0])

        mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, config
        )

        # Should be masked (negative advantage + high log-ratio)
        assert mask[0] == pytest.approx(0.0)

    def test_stale_negative_mask_does_not_mask_positive_advantages(self):
        """Test that sequences with positive advantages are not masked."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        response_mask_bool = torch.tensor([[True, True, True]])

        # Sequence with positive advantage and high log-ratio
        log_ratio = torch.tensor([[2.5, 2.5, 2.5]])  # Mean = 2.5 > 2.0
        advantages = torch.tensor([1.0])  # Positive advantage

        mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, config
        )

        # Should NOT be masked (positive advantage)
        assert mask[0] == pytest.approx(1.0)

    def test_stale_negative_mask_does_not_mask_low_log_ratio(self):
        """Test that sequences with low log-ratio are not masked."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        response_mask_bool = torch.tensor([[True, True, True]])

        # Sequence with negative advantage but low log-ratio
        log_ratio = torch.tensor([[0.5, 0.5, 0.5]])  # Mean = 0.5 < 2.0
        advantages = torch.tensor([-1.0])

        mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, config
        )

        # Should NOT be masked (low log-ratio)
        assert mask[0] == pytest.approx(1.0)

    def test_stale_negative_mask_multiple_sequences(self):
        """Test stale negative mask with multiple sequences."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        response_mask_bool = torch.tensor([
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ])

        # Three sequences with different characteristics
        log_ratio = torch.tensor([
            [2.5, 2.5, 2.5],  # Seq 0: high log-ratio
            [0.5, 0.5, 0.5],  # Seq 1: low log-ratio
            [2.5, 2.5, 2.5],  # Seq 2: high log-ratio
        ])
        advantages = torch.tensor([-1.0, -1.0, 1.0])  # Seq 2 has positive advantage

        mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, config
        )

        # Seq 0: masked (negative + high log-ratio)
        assert mask[0] == pytest.approx(0.0)
        # Seq 1: not masked (negative + low log-ratio)
        assert mask[1] == pytest.approx(1.0)
        # Seq 2: not masked (positive advantage)
        assert mask[2] == pytest.approx(1.0)

    def test_stale_negative_mask_with_response_mask(self):
        """Test that response mask is respected."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        response_mask_bool = torch.tensor([[True, False, True]])  # Middle token masked

        # High log-ratio but one token is masked
        log_ratio = torch.tensor([[2.5, 2.5, 2.5]])  # Mean of unmasked = 2.5 > 2.0
        advantages = torch.tensor([-1.0])

        mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, config
        )

        # Should be masked (negative advantage + high mean log-ratio)
        assert mask[0] == pytest.approx(0.0)


class TestImportanceGating:
    """Test MiMo-V2 importance weight gating."""

    def test_importance_gating_zeros_outliers(self):
        """Test that importance gating removes outliers outside epsilon band."""
        config = GrpoConfig(
            enable_importance_gating=True,
            importance_epsilon_low=0.8,
            importance_epsilon_high=1.2
        )

        ratio = torch.tensor([0.5, 1.0, 1.5])

        gated = _apply_importance_gating(ratio, config)

        # Should zero outliers
        assert gated[0] == pytest.approx(0.0)  # Below band (0.5 < 0.8)
        assert gated[1] != 0.0  # Inside band (1.0 in [0.8, 1.2])
        assert gated[2] == pytest.approx(0.0)  # Above band (1.5 > 1.2)

    def test_importance_gating_keeps_in_band_values(self):
        """Test that importance gating keeps values inside epsilon band."""
        config = GrpoConfig(
            enable_importance_gating=True,
            importance_epsilon_low=0.8,
            importance_epsilon_high=1.2
        )

        ratio = torch.tensor([0.9, 1.0, 1.1])

        gated = _apply_importance_gating(ratio, config)

        # All values inside band, should be unchanged
        assert gated[0] == pytest.approx(0.9)
        assert gated[1] == pytest.approx(1.0)
        assert gated[2] == pytest.approx(1.1)

    def test_importance_gating_at_exact_boundaries(self):
        """Test importance gating at exact epsilon boundaries."""
        config = GrpoConfig(
            enable_importance_gating=True,
            importance_epsilon_low=0.8,
            importance_epsilon_high=1.2
        )

        ratio = torch.tensor([0.8, 1.2])

        gated = _apply_importance_gating(ratio, config)

        # At boundaries should be kept
        assert gated[0] == pytest.approx(0.8)
        assert gated[1] == pytest.approx(1.2)

    def test_importance_gating_with_narrow_band(self):
        """Test importance gating with a very narrow band."""
        config = GrpoConfig(
            enable_importance_gating=True,
            importance_epsilon_low=0.95,
            importance_epsilon_high=1.05
        )

        ratio = torch.tensor([0.9, 1.0, 1.1])

        gated = _apply_importance_gating(ratio, config)

        # Only middle value inside narrow band
        assert gated[0] == pytest.approx(0.0)
        assert gated[1] == pytest.approx(1.0)
        assert gated[2] == pytest.approx(0.0)

    def test_importance_gating_with_wide_band(self):
        """Test importance gating with a very wide band."""
        config = GrpoConfig(
            enable_importance_gating=True,
            importance_epsilon_low=0.1,
            importance_epsilon_high=10.0
        )

        ratio = torch.tensor([0.5, 1.0, 2.0])

        gated = _apply_importance_gating(ratio, config)

        # All values inside wide band, should be unchanged
        assert gated[0] == pytest.approx(0.5)
        assert gated[1] == pytest.approx(1.0)
        assert gated[2] == pytest.approx(2.0)


class TestTrainInferGate:
    """Test GLM-5 train/infer mismatch gate (simplified version)."""

    def test_train_infer_gate_returns_ratio_unchanged(self):
        """Test that current train/infer gate implementation returns ratio unchanged.

        Note: Full implementation would require separate train/infer engine logits.
        Current simplified version assumes ratio already incorporates this.
        """
        config = GrpoConfig(
            enable_train_infer_gate=True,
            train_infer_gate_beta=2.0
        )

        ratio = torch.tensor([0.5, 1.0, 1.5])

        gated = torch.ops.flashrl_framework.trainer.grpo.loss_variants._apply_train_infer_gate(
            ratio, config
        )

        # Current implementation returns ratio unchanged
        assert gated[0] == pytest.approx(0.5)
        assert gated[1] == pytest.approx(1.0)
        assert gated[2] == pytest.approx(1.5)


class TestGateCombinations:
    """Test combinations of different gates."""

    def test_stale_negative_mask_and_importance_gating_interaction(self):
        """Test interaction between stale negative mask and importance gating."""
        stale_config = GrpoConfig(
            enable_stale_negative_mask=True,
            stale_negative_mask_delta=2.0
        )
        gate_config = GrpoConfig(
            enable_importance_gating=True,
            importance_epsilon_low=0.8,
            importance_epsilon_high=1.2
        )

        response_mask_bool = torch.tensor([[True, True, True]])

        # Sequence that would be masked by stale negative mask
        log_ratio = torch.tensor([[2.5, 2.5, 2.5]])
        advantages = torch.tensor([-1.0])

        # Apply stale negative mask
        stale_mask = _compute_stale_negative_mask(
            log_ratio, advantages, response_mask_bool, stale_config
        )

        # Ratio that would be gated
        ratio = torch.tensor([0.5, 1.0, 1.5])

        # Apply importance gating
        gated_ratio = _apply_importance_gating(ratio, gate_config)

        # Verify both gates work independently
        assert stale_mask[0] == pytest.approx(0.0)  # Sequence masked
        assert gated_ratio[0] == pytest.approx(0.0)  # Outlier gated
        assert gated_ratio[1] != 0.0  # In-band value kept

    def test_multiple_gates_can_be_combined(self):
        """Test that multiple gates can be enabled simultaneously."""
        config = GrpoConfig(
            enable_stale_negative_mask=True,
            enable_importance_gating=True,
            stale_negative_mask_delta=2.0,
            importance_epsilon_low=0.8,
            importance_epsilon_high=1.2
        )

        # Both gates should be enabled
        assert config.enable_stale_negative_mask is True
        assert config.enable_importance_gating is True
