# GRPO Loss Presets

This page documents the GRPO loss building blocks that are currently implemented
in FlashRL and how the shipped example presets use them.

The source of truth here is code, not paper claims or YAML prose:

- built-in named presets come from
  [`flashrl/framework/controller/grpo/loss_variants.py`](../flashrl/framework/controller/grpo/loss_variants.py)
- the `custom` example comes from
  [`flashrl/examples/presets/custom.yaml`](../flashrl/examples/presets/custom.yaml)

## Supported Building Blocks

The current GRPO loss implementation supports these high-level building blocks:

- Clipping modes:
  `symmetric`, `asymmetric`, `hard_mask`, and `none`
- KL modes:
  `none`, `k1`, `k3`, and `unbiased`
- Sequence-level off-policy masking:
  masks entire stale-negative responses after comparing average sequence drift
  against `off_policy_sequence_masking_delta`
- IcePop train/infer mismatch gate:
  gates token updates when the train-old and infer-old policies diverge too far
- MiMo-style importance gating:
  hard-gates tokens outside an importance-ratio band
- Soft quadratic log-ratio penalty:
  adds a Kimi-style penalty on `log(pi_theta / pi_old)^2`
- Advantage processing:
  `group_centered` and `group_normalized`
- Entropy regularization:
  optional response-token entropy bonus

## Preset Matrix

The table below uses resolved code behavior for the four named presets and the
explicit `grpo:` block for `custom`.

| Preset | Clipping | KL | Off-Policy Seq Mask | IcePop Gate | Importance Gate | Log-Ratio Penalty | Advantage Processing | Entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `grpo_naive` | `symmetric (epsilon=0.2)` | `k3` | `off` | `off` | `off` | `off` | `group_centered` | `off` |
| `deepseek_v3.2` | `symmetric (epsilon=0.2)` | `unbiased` | `on (delta=2.0)` | `off` | `off` | `off` | `group_centered` | `0.01` |
| `glm_5` | `asymmetric (lower=0.1, upper=0.2)` | `none` | `off` | `on (beta=2.0)` | `off` | `off` | `group_normalized` | `off` |
| `kimi_k2.5` | `asymmetric (lower=0.1, upper=0.2)` | `none` | `off` | `off` | `off` | `0.01` | `group_centered` | `off` |
| `custom` | `asymmetric (lower=0.05, upper=0.15)` | `unbiased` | `on (delta=2.5)` | `off` | `off` | `0.005` | `group_centered` | `0.005` |

## Implemented But Unused By Shipped Presets

These branches exist in the GRPO loss code but are not selected by the five
example presets under `flashrl/examples/presets/`:

- `hard_mask` clipping
- MiMo-style importance gating
- `k1` KL mode

## Current Mismatches To Be Aware Of

Some comments and preset descriptions currently drift from the code:

- `deepseek_v3.2.yaml` prose mentions asymmetric clipping, but
  `resolve_loss_preset()` resolves `deepseek_v3.2` to `symmetric` clipping
- `kimi_k2.5.yaml` prose mentions hard token masking, but the current preset
  uses `asymmetric` clipping plus a soft log-ratio penalty
- `glm_5` comments and naming imply broader paper behavior, but the preset
  logic in `loss_variants.py` implements IcePop-style gating plus
  `group_normalized` advantages, not teacher-distillation logic

## Source Of Truth

Primary references for this page:

- preset resolution:
  [`flashrl/framework/controller/grpo/loss_variants.py`](../flashrl/framework/controller/grpo/loss_variants.py)
- GRPO config surface:
  [`flashrl/framework/config.py`](../flashrl/framework/config.py)
- custom example:
  [`flashrl/examples/presets/custom.yaml`](../flashrl/examples/presets/custom.yaml)
