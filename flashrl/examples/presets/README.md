# GRPO Loss Preset Examples

This directory contains example configurations demonstrating the new GRPO loss preset system. Each preset implements techniques from different research papers.

## Available Presets

### `grpo_naive` (default)
Classical PPO-style symmetric clipping. Good baseline for most RLHF tasks.

### `deepseek_v3.2`
DeepSeek-V3.2 implementation with:
- PPO-style symmetric clipping (standard [1-ε, 1+ε] bounds with ε=0.2)
- Unbiased token-level KL estimator
- Sequence-level stale-negative masking
- Entropy regularization

### `kimi_k2.5`
Kimi K2.5 implementation with:
- Asymmetric clipping (same as GLM-5)
- Soft quadratic log-ratio penalty
- No explicit reference KL

### `glm_5`
GLM-5 implementation with:
- Train/infer mismatch gate (IcePop-style)
- Group-normalized advantages
- No explicit KL in reasoning RL backbone

### `custom`
Use explicit configuration parameters for full control.

## Configuration Examples

See individual YAML files for complete examples of each preset.

These local preset YAMLs now pin their Hugging Face actor/reference/serving
sections to `cpu` for reliable local runs. Explicit `device: mps` remains an
advanced opt-in rather than the recommended default on macOS.

## Research Paper Reference

These implementations are based on techniques from:
"DeepSeek-V3.2, GLM-5, and Kimi K2.5: A Comparative Analysis of GRPO Variants"
