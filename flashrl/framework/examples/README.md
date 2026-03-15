# FlashRL Examples

Examples live in their own folders and can include a training entrypoint,
evaluation helpers, one or more YAML configs, and an example-specific README.

## Reasoning-Math Example

The reasoning-math example is a strict R1-Zero-style math prototype with explicit
dataset selection. It trains a base Qwen model with rule-based rewards, no
system prompt, and a strict `<think>...</think><answer>...</answer>` output
contract.

See [flashrl/framework/examples/reasoning-math/README.md](reasoning-math/README.md) for:

- supported run modes
- the CLI-first example workflow
- config differences
- evaluation commands
- environment variables
- expected outputs and logs
- troubleshooting

## Reasoning-Code Example

The reasoning-code example is a strict R1-style Codeforces prototype with local
execution reward. It keeps the same visible `<think>...</think><answer>...</answer>`
contract as the math example, but the final answer is a fenced Python code
block that is executed against official tests.

See [flashrl/framework/examples/reasoning-code/README.md](reasoning-code/README.md) for:

- the script-based train and eval commands
- dataset filtering and rating defaults
- execution limits and sandbox notes
- config differences
- reward behavior

Both example folders are intentionally hyphenated, so they are run as scripts.
Their `train.py` / `eval.py` entrypoints load the YAML profiles directly and
construct `FlashRL(...)` in code.

For production-style training, prefer config-driven checkpointing in
`RunConfig` / YAML. The explicit `save_checkpoint(...)` / `load_checkpoint(...)`
calls in the task examples are still supported, but they are the manual escape
hatch rather than the recommended training workflow.

## Mock Checkpointing Example

The mock-checkpointing example is fully offline. It uses fake local backends to
exercise the real managed checkpoint subsystem, including interval saves,
`latest.json`, and append-resume into the same run directory.

See [flashrl/framework/examples/mock-checkpointing/README.md](mock-checkpointing/README.md) for:

- initial training with managed interval checkpoints
- resume from `latest`
- optional final checkpoint behavior

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
