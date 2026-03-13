# FlashRL Examples

Each example now lives in its own folder with:
- `train.py` for the Python hooks and example entrypoint
- `config.yaml` for the run configuration

## Reasoning Example

The reasoning example teaches a model to emit `<reason>` tags for step-by-step
math answers.

**Folder:**
```text
examples/reasoning/
├── train.py
└── config.yaml
```

**Run the example script:**
```bash
python3 -m examples.reasoning.train
```

**Run directly from YAML:**
```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config.yaml
```

**Run with the managed `vllm` serving backend:**
```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config_vllm.yaml
```

**Inspect runs after training:**
```bash
open docs/run_viewer.html
```

In Chrome or Edge, click `Open run folder` and choose the default `logs/`
folder to inspect run summaries, events, console logs, and grouped GRPO rollouts.

Older `.flashrl-runs/` directories can still be opened manually if you already
have them.

**What the YAML does:**
- chooses shared defaults plus separate training and serving settings
- configures grouped GRPO rollout and optimization
- keeps logging in compact console mode
- keeps metrics enabled by default
- wires the rollout, reward, and dataset hooks through Python import strings
- keeps the public rollout hook sample-oriented: one rollout per input prompt
- treats `training.batch_size` as total sampled completions per optimizer step, so prompts per step are `batch_size / grpo.group_size`
- supports `serving.debug_live_rollout: true` for slower token-level live serving debug with TTFT/TPOT capture
- supports `serving.backend: vllm` for managed local `vllm serve` replicas in a separate Python runtime

**Shared defaults with training/serving overrides:**
```yaml
common:
  model_name: Qwen/Qwen2.5-0.5B-Instruct

training:
  num_threads: 1
  batch_size: 4
  max_epochs: 3

serving:
  num_threads: 1
  debug_live_rollout: false

grpo:
  group_size: 2
  clip_ratio: 0.2
```

**Example prompt:**
```text
Please solve this step by step. Use <reason> tags to show your reasoning.

Question: What is 15 + 27?
```

## YAML Hook Format

FlashRL YAML configs reference Python code with `module:attribute` strings:

```yaml
hooks:
  rollout_fn: examples.reasoning.train:reasoning_rollout_fn
  reward_fn: examples.reasoning.train:reasoning_reward_fn
  dataset_fn: examples.reasoning.train:build_dataset
```

## Managed vLLM Backend

Use `serving.backend: vllm` when you want FlashRL to launch managed local
`vllm serve` processes in a separate Python runtime.

```yaml
serving:
  backend: vllm
  runtime_python: ~/.venv-vllm/bin/python
  num_replicas: 2
  debug_live_rollout: false
  vllm_args:
    - --dtype=float16
    - --max-model-len=4096
    - --gpu-memory-utilization=0.85
    - --max-num-seqs=32
    - --enable-prefix-caching
```

Notes:
- FlashRL uses `/v1/completions` against the managed local vLLM servers
- the backend keeps restart-based weight sync after optimizer steps
- `serving.debug_live_rollout: true` is not supported with `vllm`

## Local Observability

FlashRL ships with a small local `Grafana + Prometheus + Pushgateway` stack.
Metrics are enabled by default and use best-effort Pushgateway pushes, so
training still runs if the metrics stack is not available.

**Start the stack:**
```bash
./dev.sh metrics up
```

`./dev.sh metrics up` waits until Grafana, Prometheus, and Pushgateway are endpoint-ready before reporting success.

**Run the reasoning example:**
```bash
python3 -m examples.reasoning.train
```

**Open the dashboard:**
- Grafana: `http://localhost:3000` (`admin` / `admin`)

**Optional troubleshooting URLs:**
- Prometheus: `http://localhost:9090`
- Pushgateway: `http://localhost:9091`

**Stop or clear the stack:**
```bash
./dev.sh metrics down
./dev.sh metrics reset
```
