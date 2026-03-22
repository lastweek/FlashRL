# FlashRL Math Example

This example keeps the strict math output contract and supports both:

- blackbox rollout construction in user code
- whitebox rollout construction from `flashrl.framework.agent`

## Files

- `train.py`
- `eval.py`
- `config.yaml`
- `config-vllm.yaml`

`config.yaml` is the normal default config. It contains:

- `framework:` for local FlashRL runtime and training settings
- `platform:` for Kubernetes images and workload policy

`config-vllm.yaml` is the managed local vLLM variant for the same example.
The local minikube smoke config lives separately at
`flashrl/platform/dev/math-minikube.yaml`.

## Local Runs

Default local Hugging Face path:

```bash
python3 -m flashrl.examples.math.train
```

The default local config is CPU-first for reliability on Apple Silicon and
other low-memory local setups. Explicit `device: mps` remains an advanced
opt-in, not the recommended local path. During rollout, the example also
prints one concise `math rollout ...` progress line per prompt rollout.

Managed local vLLM:

```bash
python3 -m flashrl.examples.math.train --config flashrl/examples/math/config-vllm.yaml
python3 -m flashrl.examples.math.eval --config flashrl/examples/math/config-vllm.yaml
```

`config-vllm.yaml` remains the faster managed-local path when you want serving
performance over the CPU-first local default.

Whitebox rollout:

```bash
python3 -m flashrl.examples.math.train \
  --config flashrl/examples/math/config-vllm.yaml \
  --training-mode reasoning \
  --rollout-mode whitebox \
  --dataset gsm8k \
  --train-limit 64
```

## Kubernetes Run

Platform runs are config-driven. The math example knobs that `train.py` accepts
locally, such as dataset, limit, training mode, and rollout mode, come from the
YAML hook kwargs under `framework.hooks`, not from extra shell flags.

The public math config now defaults to the agentic whitebox rollout:

```yaml
framework:
  hooks:
    dataset_fn:
      kwargs:
        dataset: gsm8k
        limit: 8
        training_mode: math
    rollout_fn:
      kwargs:
        rollout_mode: whitebox
    reward_fn:
      kwargs: {}
```

Switch back to the plain example rollout by editing
`hooks.rollout_fn.kwargs.rollout_mode` to `blackbox`. Switch the whole run to
reasoning mode by changing `dataset_fn.kwargs.training_mode` from `math` to
`reasoning`; rollout and reward inherit it from prompt metadata by default.

Render one `FlashRLJob`:

```bash
python3 -m flashrl platform render \
  --config flashrl/examples/math/config.yaml \
  --output flashrl-job.yaml
```

Apply it:

```bash
kubectl apply -f flashrl-job.yaml
kubectl get flashrljobs
kubectl logs -n default -l flashrl.dev/job=flashrl-math-demo
```

For the local minikube smoke path, use the dev-only config:

```bash
python3 scripts/run_minikube_math_e2e.py --config flashrl/platform/dev/math-minikube.yaml
```

## Notes

- `FLASHRL_VLLM_PYTHON` is auto-filled by the example entrypoint when the selected config uses `serving.backend: vllm` and a prepared local runtime is found.
- TensorBoard logs are written under `logs/`.
- Managed checkpoints still use the YAML `checkpointing:` section.
