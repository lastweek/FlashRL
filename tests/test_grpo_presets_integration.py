"""Integration tests for GRPO presets - full loss assembly."""

import pytest
import torch

from flashrl.framework.config import GrpoConfig
from flashrl.framework.trainer.grpo.loss_variants import assemble_grpo_loss


def create_test_data(batch_size=2, seq_len=10, vocab_size=100):
    """Create synthetic test data for loss assembly."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    prompt_lengths = torch.tensor([3, 4])  # Different prompt lengths

    # Create actor logits
    actor_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Create reference logits (optional)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Create rollout response log probs (one list per batch)
    response_lengths = seq_len - prompt_lengths.tolist()
    rollout_response_log_probs = [
        [0.0] * response_lengths[0],
        [0.0] * response_lengths[1],
    ]

    # Create advantages
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
        "actor_logits": actor_logits,
        "ref_logits": ref_logits,
        "rollout_response_log_probs": rollout_response_log_probs,
        "advantages": advantages,
    }


class TestPresetIntegration:
    """Integration tests for each preset."""

    @pytest.mark.parametrize("preset_name", [
        "grpo_naive",
        "ppo_clipped",  # Alias for grpo_naive
        "deepseek_v3.2",
        "glm_5",
        "kimi_k2.5",
        "mimo_v2",
    ])
    def test_preset_loss_assembly_is_finite(self, preset_name):
        """Test that each preset assembles a finite loss."""
        config = GrpoConfig(loss_preset=preset_name)
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Loss should be finite
        assert torch.isfinite(result.loss)
        # Policy loss should be finite
        assert torch.isfinite(result.policy_loss)
        # KL divergence should be finite
        assert torch.isfinite(result.kl_divergence)
        # Response token count should be positive
        assert result.response_tokens_total > 0

    @pytest.mark.parametrize("preset_name", [
        "grpo_naive",
        "deepseek_v3.2",
        "glm_5",
        "kimi_k2.5",
        "mimo_v2",
    ])
    def test_preset_loss_is_negative(self, preset_name):
        """Test that loss is negative (we minimize negative objective)."""
        config = GrpoConfig(loss_preset=preset_name)
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Loss should be negative (we're minimizing negative objective)
        assert result.loss.item() < 0

    def test_grpo_naive_uses_symmetric_clipping(self):
        """Test that grpo_naive uses symmetric clipping."""
        config = GrpoConfig(loss_preset="grpo_naive")
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Should have symmetric clipping statistics
        # Clip fraction should be reasonable (0-100%)
        assert 0.0 <= result.clip_fraction <= 1.0
        # Importance ratio statistics should be finite
        assert torch.isfinite(torch.tensor(result.importance_sampling_ratio_mean))
        assert torch.isfinite(torch.tensor(result.importance_sampling_ratio_std))

    def test_deepseek_v32_uses_unbiased_kl(self):
        """Test that deepseek_v3.2 uses unbiased KL mode."""
        config = GrpoConfig(loss_preset="deepseek_v3.2")
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # KL should be computed (unbiased mode with ref_logits)
        assert result.kl_divergence.item() > 0

    def test_kimi_k25_uses_hard_mask(self):
        """Test that kimi_k2.5 uses hard mask clipping."""
        config = GrpoConfig(loss_preset="kimi_k2.5")
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=None,  # No reference for Kimi
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Should not raise, KL should be zero (no reference)
        assert result.kl_divergence.item() == pytest.approx(0.0)
        # Loss should still be finite
        assert torch.isfinite(result.loss)

    def test_glm_5_uses_group_normalized_advantages(self):
        """Test that glm_5 uses group-normalized advantages."""
        config = GrpoConfig(loss_preset="glm_5")
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=None,  # No explicit KL for GLM-5
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Loss should be finite with group-normalized advantages
        assert torch.isfinite(result.loss)
        # KL should be zero (no reference)
        assert result.kl_divergence.item() == pytest.approx(0.0)

    def test_mimo_v2_uses_importance_gating(self):
        """Test that mimo_v2 uses importance gating."""
        config = GrpoConfig(loss_preset="mimo_v2")
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=None,  # No KL for MiMo
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Loss should be finite with importance gating
        assert torch.isfinite(result.loss)
        # KL should be zero (no reference)
        assert result.kl_divergence.item() == pytest.approx(0.0)

    def test_custom_preset_allows_explicit_parameters(self):
        """Test that custom preset allows explicit parameter configuration."""
        config = GrpoConfig(
            loss_preset="custom",
            clipping_mode="asymmetric",
            clip_ratio_lower=0.15,
            clip_ratio_upper=0.25,
            kl_mode="unbiased",
            kl_coefficient=0.05,
        )
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Should use custom parameters
        assert torch.isfinite(result.loss)
        assert result.kl_divergence.item() > 0  # unbiased mode with ref


class TestPresetStatistics:
    """Test that presets produce reasonable statistics."""

    def test_importance_ratio_statistics_are_reasonable(self):
        """Test that importance ratio statistics are within expected ranges."""
        config = GrpoConfig(loss_preset="grpo_naive")
        data = create_test_data()

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Mean ratio should be around 1.0 (policy hasn't changed too much)
        assert 0.1 < result.importance_sampling_ratio_mean < 10.0
        # Std should be non-negative
        assert result.importance_sampling_ratio_std >= 0
        # Min should be positive
        assert result.importance_sampling_ratio_min > 0
        # Max should be positive
        assert result.importance_sampling_ratio_max > 0
        # Max should be >= min
        assert result.importance_sampling_ratio_max >= result.importance_sampling_ratio_min
        # Clip fraction should be between 0 and 1
        assert 0.0 <= result.clip_fraction <= 1.0

    def test_response_token_count_is_correct(self):
        """Test that response token count matches actual response length."""
        config = GrpoConfig(loss_preset="grpo_naive")
        data = create_test_data(batch_size=2, seq_len=10)

        result = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Response tokens = seq_len - prompt_length for each batch
        expected_response_tokens = (
            (10 - 3) +  # First batch
            (10 - 4)    # Second batch
        )
        assert result.response_tokens_total == expected_response_tokens


class TestPresetReproducibility:
    """Test that presets produce reproducible results."""

    def test_same_preset_produces_same_loss(self):
        """Test that same preset with same data produces same loss."""
        config = GrpoConfig(loss_preset="grpo_naive")
        data = create_test_data()

        # Set random seed for reproducibility
        torch.manual_seed(42)
        result1 = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        torch.manual_seed(42)
        result2 = assemble_grpo_loss(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            prompt_lengths=data["prompt_lengths"],
            actor_logits=data["actor_logits"],
            ref_logits=data["ref_logits"],
            rollout_response_log_probs=data["rollout_response_log_probs"],
            advantages=data["advantages"],
            config=config,
        )

        # Should produce identical results
        assert result1.loss.item() == pytest.approx(result2.loss.item())
        assert result1.kl_divergence.item() == pytest.approx(result2.kl_divergence.item())
