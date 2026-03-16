"""Tests for GRPO clipping modes."""

import pytest
import torch

from flashrl.framework.config import GrpoConfig
from flashrl.framework.trainer.grpo.loss_variants import _apply_clipping


class TestSymmetricClipping:
    """Test PPO-style symmetric clipping."""

    def test_symmetric_clipping_bounds_ratio(self):
        """Test that symmetric clipping respects [1-ε, 1+ε] bounds."""
        config = GrpoConfig(clipping_mode="symmetric", clip_ratio=0.2)
        response_mask = torch.ones((1, 5))

        # Create ratio values outside bounds
        ratio = torch.tensor([[0.5, 0.8, 1.0, 1.2, 1.5]])
        log_ratio = torch.log(ratio)
        advantages = torch.zeros(1)

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Lower bound: 1.0 - 0.2 = 0.8
        assert clipped[0, 0] == pytest.approx(0.8)
        # Within bounds: unchanged
        assert clipped[0, 1] == pytest.approx(0.8)
        assert clipped[0, 2] == pytest.approx(1.0)
        assert clipped[0, 3] == pytest.approx(1.2)
        # Upper bound: 1.0 + 0.2 = 1.2
        assert clipped[0, 4] == pytest.approx(1.2)

    def test_symmetric_clipping_with_different_epsilon(self):
        """Test symmetric clipping with different epsilon values."""
        config = GrpoConfig(clipping_mode="symmetric", clip_ratio=0.1)
        response_mask = torch.ones((1, 3))

        ratio = torch.tensor([[0.5, 1.0, 1.5]])
        log_ratio = torch.log(ratio)
        advantages = torch.zeros(1)

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Bounds: [0.9, 1.1]
        assert clipped[0, 0] == pytest.approx(0.9)
        assert clipped[0, 1] == pytest.approx(1.0)
        assert clipped[0, 2] == pytest.approx(1.1)


class TestAsymmetricClipping:
    """Test DeepSeek-V3.2 style asymmetric clipping."""

    def test_asymmetric_clipping_uses_advantages(self):
        """Test that asymmetric clipping uses advantage sign for bounds."""
        config = GrpoConfig(
            clipping_mode="asymmetric",
            clip_ratio_lower=0.1,
            clip_ratio_upper=0.2
        )
        response_mask = torch.ones((2, 3))

        # Two sequences with different advantage signs
        ratio = torch.tensor([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]])
        log_ratio = torch.log(ratio)
        advantages = torch.tensor([1.0, -1.0])  # First positive, second negative

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # When A > 0: [1-0.1, 1+0.2] = [0.9, 1.2]
        assert clipped[0, 0] == pytest.approx(0.9)
        assert clipped[0, 1] == pytest.approx(1.0)
        assert clipped[0, 2] == pytest.approx(1.2)

        # When A < 0: [1-0.2, 1+0.1] = [0.8, 1.1]
        assert clipped[1, 0] == pytest.approx(0.8)
        assert clipped[1, 1] == pytest.approx(1.0)
        assert clipped[1, 2] == pytest.approx(1.1)

    def test_asymmetric_clipping_with_zero_advantage(self):
        """Test asymmetric clipping when advantage is zero."""
        config = GrpoConfig(
            clipping_mode="asymmetric",
            clip_ratio_lower=0.1,
            clip_ratio_upper=0.2
        )
        response_mask = torch.ones((1, 3))

        ratio = torch.tensor([[0.5, 1.0, 1.5]])
        log_ratio = torch.log(ratio)
        advantages = torch.tensor([0.0])  # Zero advantage

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # With A=0, should use upper bounds (default behavior)
        assert clipped[0, 0] == pytest.approx(0.9)
        assert clipped[0, 1] == pytest.approx(1.0)
        assert clipped[0, 2] == pytest.approx(1.2)


class TestHardMaskClipping:
    """Test Kimi K2.5 style hard mask clipping."""

    def test_hard_mask_zeros_out_of_band_tokens(self):
        """Test that hard mask zeros tokens outside log-ratio band."""
        config = GrpoConfig(
            clipping_mode="hard_mask",
            clip_log_ratio_alpha=-5.0,
            clip_log_ratio_beta=5.0
        )
        response_mask = torch.ones((1, 3))

        # Create log-ratio values outside band
        log_ratio = torch.tensor([[-10.0, 0.0, 10.0]])
        ratio = torch.exp(log_ratio)
        advantages = torch.zeros(1)

        masked = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Tokens outside band should be zeroed
        assert masked[0, 0] == pytest.approx(0.0)  # log-ratio = -10 < -5
        assert masked[0, 1] != 0.0  # log-ratio = 0, inside band
        assert masked[0, 2] == pytest.approx(0.0)  # log-ratio = 10 > 5

    def test_hard_mask_keeps_in_band_tokens_unchanged(self):
        """Test that hard mask keeps tokens inside log-ratio band unchanged."""
        config = GrpoConfig(
            clipping_mode="hard_mask",
            clip_log_ratio_alpha=-2.0,
            clip_log_ratio_beta=2.0
        )
        response_mask = torch.ones((1, 3))

        # All tokens inside band
        log_ratio = torch.tensor([[-1.0, 0.0, 1.0]])
        ratio = torch.tensor([[0.5, 1.0, 1.5]])
        advantages = torch.zeros(1)

        masked = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # All tokens should be unchanged
        assert masked[0, 0] == pytest.approx(0.5)
        assert masked[0, 1] == pytest.approx(1.0)
        assert masked[0, 2] == pytest.approx(1.5)

    def test_hard_mask_with_narrow_band(self):
        """Test hard mask with a very narrow band."""
        config = GrpoConfig(
            clipping_mode="hard_mask",
            clip_log_ratio_alpha=-0.5,
            clip_log_ratio_beta=0.5
        )
        response_mask = torch.ones((1, 3))

        # Only middle token inside narrow band
        log_ratio = torch.tensor([[-1.0, 0.0, 1.0]])
        ratio = torch.exp(log_ratio)
        advantages = torch.zeros(1)

        masked = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        assert masked[0, 0] == pytest.approx(0.0)
        assert masked[0, 1] != 0.0
        assert masked[0, 2] == pytest.approx(0.0)


class TestNoClipping:
    """Test no clipping mode (MiMo-V2)."""

    def test_no_clipping_returns_ratio_unchanged(self):
        """Test that no clipping mode returns ratio unchanged."""
        config = GrpoConfig(clipping_mode="none")
        response_mask = torch.ones((1, 3))

        ratio = torch.tensor([[0.1, 1.0, 10.0]])
        log_ratio = torch.log(ratio)
        advantages = torch.zeros(1)

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Should be unchanged
        assert clipped[0, 0] == pytest.approx(0.1)
        assert clipped[0, 1] == pytest.approx(1.0)
        assert clipped[0, 2] == pytest.approx(10.0)


class TestClippingEdgeCases:
    """Test edge cases for clipping modes."""

    def test_clipping_with_response_mask(self):
        """Test that clipping respects response mask."""
        config = GrpoConfig(clipping_mode="symmetric", clip_ratio=0.2)
        response_mask = torch.tensor([[1.0, 0.0, 1.0]])  # Mask out middle token

        ratio = torch.tensor([[0.5, 1.0, 1.5]])
        log_ratio = torch.log(ratio)
        advantages = torch.zeros(1)

        # Clipping function should handle mask correctly
        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Should still clip correctly
        assert clipped[0, 0] == pytest.approx(0.8)
        assert clipped[0, 2] == pytest.approx(1.2)

    def test_clipping_with_empty_response(self):
        """Test clipping when response is empty (all masked)."""
        config = GrpoConfig(clipping_mode="symmetric", clip_ratio=0.2)
        response_mask = torch.zeros((1, 3))  # All masked out

        ratio = torch.tensor([[0.5, 1.0, 1.5]])
        log_ratio = torch.log(ratio)
        advantages = torch.zeros(1)

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Should not crash, return valid tensor
        assert clipped.shape == (1, 3)

    def test_clipping_with_exactly_at_bounds(self):
        """Test clipping when ratio is exactly at bounds."""
        config = GrpoConfig(clipping_mode="symmetric", clip_ratio=0.2)
        response_mask = torch.ones((1, 3))

        # Ratio exactly at bounds
        ratio = torch.tensor([[0.8, 1.0, 1.2]])
        log_ratio = torch.log(ratio)
        advantages = torch.zeros(1)

        clipped = _apply_clipping(ratio, log_ratio, advantages, response_mask, config)

        # Should remain unchanged (at bounds)
        assert clipped[0, 0] == pytest.approx(0.8)
        assert clipped[0, 1] == pytest.approx(1.0)
        assert clipped[0, 2] == pytest.approx(1.2)
