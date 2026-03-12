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

**What the YAML does:**
- chooses shared defaults plus separate training and serving settings
- configures grouped GRPO rollout and optimization
- keeps logging in compact console mode
- keeps metrics enabled by default
- wires the rollout, reward, and dataset hooks through Python import strings
- keeps the public rollout hook sample-oriented: one rollout per input prompt
- treats `training.batch_size` as total sampled completions per optimizer step, so prompts per step are `batch_size / grpo.group_size`

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
