# FlashRL Examples

Examples live in their own folders and can include a training entrypoint,
evaluation helpers, one or more YAML configs, and an example-specific README.

## Reasoning Example

The reasoning example is a strict R1-Zero-style math prototype over GSM8K. It
trains a base Qwen model with rule-based rewards, no system prompt, and a strict
`<think>...</think><answer>...</answer>` output contract.

See [examples/reasoning/README.md](reasoning/README.md) for:

- supported run modes
- the split between `config*.yaml` and `math.yaml`
- config differences
- evaluation commands
- environment variables
- expected outputs and logs
- troubleshooting

## YAML Hook Format

FlashRL YAML configs reference Python code with `module:attribute` strings:

```yaml
hooks:
  rollout_fn: examples.reasoning.train:reasoning_rollout_fn
  reward_fn: examples.reasoning.train:math_reward_fn
  dataset_fn: examples.reasoning.train:build_math_train_dataset
```

## Managed vLLM Backend

Use `serving.backend: vllm` when you want FlashRL to launch managed local
`vllm serve` processes.

```yaml
serving:
  backend: vllm
  runtime_python: ${FLASHRL_VLLM_PYTHON}
  num_replicas: 1
  debug_live_rollout: false
```

Notes:
- FlashRL uses `/v1/completions` against the managed local vLLM servers
- the backend keeps restart-based weight sync after optimizer steps
- same-environment setup is supported with the optional `vllm` extra: `pip install -e '.[vllm]'`
- dedicated runtimes are also supported by setting `FLASHRL_VLLM_PYTHON` and using `serving.runtime_python`
- on macOS Apple Silicon, prepare a `vllm-metal` runtime and point `FLASHRL_VLLM_PYTHON` at it
- `serving.debug_live_rollout: true` is not supported with `vllm`

The repo helper can prepare a default dedicated runtime:

```bash
./dev.sh vllm setup
source ./dev.sh
```

## Local Observability

TensorBoard is the default local metrics path. Each FlashRL run writes
TensorBoard event files directly into its run directory under `logs/`.

**Open TensorBoard:**
```bash
tensorboard --logdir logs
```

FlashRL also ships with a small optional `Grafana + Prometheus + Pushgateway`
stack. Enable `metrics.pushgateway.enabled: true` in your run config when you
want to publish the same training metrics into the local dashboard stack.

**Start the stack:**
```bash
./dev.sh metrics up
```

`./dev.sh metrics up` waits until Grafana, Prometheus, and Pushgateway are endpoint-ready before reporting success.

**Open the dashboard:**
- Grafana: `http://localhost:3000` (`admin` / `admin`)

**Optional troubleshooting URLs:**
- Prometheus: `http://localhost:9090`
- Pushgateway: `http://localhost:9091`

**Inspect FlashRL runs locally:**
- open `docs/viewer.html` in Chrome or Edge
- use the `Run History` workspace to inspect the default `logs/` directory
- use the `Live Runtime` workspace to connect to a running admin endpoint

**Stop or clear the stack:**
```bash
./dev.sh metrics down
./dev.sh metrics reset
```
