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
- chooses the training model, serving model, and trainer settings
- keeps logging in compact console mode
- keeps metrics enabled by default
- wires the rollout, reward, and dataset hooks through Python import strings

**Serving is configured separately:**
```yaml
model:
  model_name: Qwen/Qwen2.5-0.5B-Instruct

serving:
  model_name: Qwen/Qwen2.5-0.5B-Instruct
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
