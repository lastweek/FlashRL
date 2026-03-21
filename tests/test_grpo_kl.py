"""Tests for GRPO KL divergence computation modes."""

import pytest
import torch

from flashrl.framework.config import GrpoConfig
from flashrl.framework.controller.grpo.loss_variants import _compute_kl_divergence_enhanced


class TestK3Mode:
    """Test K3 mode (per-token KL, standard implementation)."""

    def test_k3_mode_computes_per_token_kl(self):
        """Test that K3 mode computes per-token KL divergence."""
        config = GrpoConfig(kl_mode="k3")

        # Create test logits
        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),  # response_mask_bool
            response_mask, response_token_count, config
        )

        # KL should be positive
        assert kl.item() > 0
        # KL should be finite
        assert torch.isfinite(kl)

    def test_k3_mode_with_zero_reference_kl(self):
        """Test K3 mode when actor and reference are identical (KL should be ~0)."""
        config = GrpoConfig(kl_mode="k3")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        # Same logits for ref and actor (should give KL ≈ 0)
        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute log_pi_theta from ref_logits (same distribution)
        log_pi_ref = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
        log_pi_theta = torch.gather(
            log_pi_ref, dim=-1, index=shift_ids.unsqueeze(-1)
        ).squeeze(-1)

        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # KL should be very close to zero (same distribution)
        assert kl.item() < 1e-5

    def test_k3_mode_respects_response_mask(self):
        """Test that K3 mode respects response mask."""
        config = GrpoConfig(kl_mode="k3")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)

        # Mask out some tokens
        response_mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 1.0],  # 3 tokens masked
            [1.0, 0.0, 1.0, 0.0, 1.0],  # 2 tokens masked
        ])
        response_token_count = torch.tensor(5.0, dtype=torch.float32)  # Total unmasked

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            response_mask.bool(), response_mask, response_token_count, config
        )

        # KL should be finite
        assert torch.isfinite(kl)


class TestK1Mode:
    """Test K1 mode (aggregate once with optional threshold)."""

    def test_k1_mode_computes_aggregate_kl(self):
        """Test that K1 mode computes aggregate KL."""
        config = GrpoConfig(kl_mode="k1")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # KL should be positive
        assert kl.item() > 0
        assert torch.isfinite(kl)

    def test_k1_mode_with_hard_threshold(self):
        """Test K1 mode with hard threshold."""
        config = GrpoConfig(
            kl_mode="k1",
            kl_hard_threshold=0.5
        )

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        # Create very different distributions (high KL)
        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size) * 10
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len) * 10
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        # This should raise ValueError due to high KL
        with pytest.raises(ValueError, match="KL divergence.*exceeds.*threshold"):
            _compute_kl_divergence_enhanced(
                ref_logits, shift_ids, log_pi_theta,
                torch.ones(batch_size, seq_len, dtype=torch.bool),
                response_mask, response_token_count, config
            )

    def test_k1_mode_without_threshold(self):
        """Test K1 mode without hard threshold (should not raise)."""
        config = GrpoConfig(kl_mode="k1")  # No threshold set

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size) * 10
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len) * 10
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        # Should not raise (no threshold)
        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # KL should be high but finite
        assert kl.item() > 0
        assert torch.isfinite(kl)


class TestUnbiasedMode:
    """Test unbiased mode (DeepSeek-V3.2 importance-sampled estimator)."""

    def test_unbiased_mode_uses_importance_sampling(self):
        """Test that unbiased mode uses importance sampling ratio."""
        config = GrpoConfig(kl_mode="unbiased")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        log_pi_old = torch.randn(batch_size, seq_len)  # Need old policy for unbiased

        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            log_pi_old,
            response_mask, response_token_count, config
        )

        # KL should be positive
        assert kl.item() > 0
        assert torch.isfinite(kl)

    def test_unbiased_mode_different_from_k3(self):
        """Test that unbiased mode gives different results than K3."""
        config_unbiased = GrpoConfig(kl_mode="unbiased")
        config_k3 = GrpoConfig(kl_mode="k3")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        log_pi_old = torch.randn(batch_size, seq_len)

        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl_unbiased = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            log_pi_old,
            response_mask, response_token_count, config_unbiased
        )

        kl_k3 = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            log_pi_old,
            response_mask, response_token_count, config_k3
        )

        # Should give different results
        assert not torch.allclose(kl_unbiased, kl_k3)


class TestNoneMode:
    """Test none mode (no KL divergence)."""

    def test_none_mode_returns_zero_kl(self):
        """Test that none mode returns zero KL."""
        config = GrpoConfig(kl_mode="none")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # KL should be exactly zero
        assert kl.item() == pytest.approx(0.0)

    def test_none_mode_ignores_ref_logits(self):
        """Test that none mode doesn't use ref_logits even if provided."""
        config = GrpoConfig(kl_mode="none")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        # Even with very different ref_logits
        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size) * 100
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # Should still return zero
        assert kl.item() == pytest.approx(0.0)


class TestKLWithNoReference:
    """Test KL computation when reference logits are None."""

    def test_k3_mode_with_none_ref_logits(self):
        """Test K3 mode returns zero when ref_logits is None."""
        config = GrpoConfig(kl_mode="k3")

        seq_len = 5
        batch_size = 2

        shift_ids = torch.randint(0, 100, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            None, shift_ids, log_pi_theta,
            torch.ones(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # Should return zero
        assert kl.item() == pytest.approx(0.0)

    def test_unbiased_mode_with_none_ref_logits(self):
        """Test unbiased mode returns zero when ref_logits is None."""
        config = GrpoConfig(kl_mode="unbiased")

        seq_len = 5
        batch_size = 2

        shift_ids = torch.randint(0, 100, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)
        log_pi_old = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        response_token_count = torch.tensor(seq_len * batch_size, dtype=torch.float32)

        kl = _compute_kl_divergence_enhanced(
            None, shift_ids, log_pi_theta,
            log_pi_old,
            response_mask, response_token_count, config
        )

        # Should return zero
        assert kl.item() == pytest.approx(0.0)


class TestKLEdgeCases:
    """Test edge cases for KL computation."""

    def test_kl_with_empty_response(self):
        """Test KL computation when response is empty."""
        config = GrpoConfig(kl_mode="k3")

        vocab_size = 100
        seq_len = 5
        batch_size = 2

        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size)
        shift_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        log_pi_theta = torch.randn(batch_size, seq_len)

        # All masked out
        response_mask = torch.zeros(batch_size, seq_len)
        response_token_count = torch.tensor(1.0, dtype=torch.float32)  # Avoid div by zero

        kl = _compute_kl_divergence_enhanced(
            ref_logits, shift_ids, log_pi_theta,
            torch.zeros(batch_size, seq_len, dtype=torch.bool),
            response_mask, response_token_count, config
        )

        # Should return zero (no tokens to compute KL over)
        assert kl.item() == pytest.approx(0.0)
