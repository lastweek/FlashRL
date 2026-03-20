# FlashRL Math Example

This example keeps the strict math output contract and supports both:

- blackbox rollout construction in user code
- whitebox rollout construction from `flashrl.framework.agent`

## Files

- `train.py`
- `eval.py`
- `config.yaml`

`config.yaml` is the only config file. It contains:

- `framework:` for local FlashRL runtime and training settings
- `platform:` for Kubernetes images and workload policy
- `profiles.vllm` for managed local vLLM
- `profiles.minikube` for the local cluster smoke profile

## Local Runs

Default local Hugging Face path:

```bash
python3 -m flashrl.examples.math.train
```

Managed local vLLM:

```bash
python3 -m flashrl.examples.math.train --profile vllm
python3 -m flashrl.examples.math.eval --profile vllm
```

Whitebox rollout:

```bash
python3 -m flashrl.examples.math.train \
  --profile vllm \
  --training-mode reasoning \
  --rollout-mode whitebox \
  --dataset gsm8k \
  --train-limit 64
```

## Kubernetes Run

Render one `FlashRLJob`:

```bash
python3 -m flashrl platform render \
  --config flashrl/examples/math/config.yaml \
  --profile minikube \
  --output flashrl-job.yaml
```

Apply it:

```bash
kubectl apply -f flashrl-job.yaml
kubectl get flashrljobs
kubectl logs -l flashrl.dev/job=flashrl-math-minikube
```

## Notes

- `FLASHRL_VLLM_PYTHON` is auto-filled by the example entrypoint when `--profile vllm` is selected and a prepared local runtime is found.
- TensorBoard logs are written under `logs/`.
- Managed checkpoints still use the YAML `checkpointing:` section.
