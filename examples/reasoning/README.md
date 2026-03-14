# FlashRL Reasoning Example

This example is a strict R1-Zero-style math prototype over GSM8K. It trains a
base Qwen model with rule-based rewards only, uses no system prompt, and
requires exactly one `<think>...</think>` block followed by one
`<answer>...</answer>` block.

The example stays concrete on purpose:

- `config.yaml` and `config_vllm.yaml` control FlashRL runtime and training
  behavior.
- `math.yaml` controls math-example specifics such as dataset source, split
  names, default limits, checkpoint path, and eval batch size.
- `train.py` and `eval.py` are the primary entrypoints.

It is a math-only prototype for local experimentation. It is not a full
DeepSeek-R1 reproduction, does not include cold-start SFT, and does not include
code-task verification yet.

Run the commands below from the repository root.

## Files In This Folder

- `train.py`: main math training entrypoint and the example hook functions.
- `eval.py`: held-out evaluation entrypoint for GSM8K test.
- `config.yaml`: cheap local Hugging Face profile.
- `config_vllm.yaml`: canonical managed local vLLM profile.
- `math.yaml`: math-example sidecar config.

## Config Responsibility

Use the files for different jobs:

- `config.yaml` / `config_vllm.yaml`
  These are FlashRL profiles. They own things like:
  `common.model_name`, `training.batch_size`, `training.max_epochs`,
  `serving.backend`, `runtime.reference_enabled`, and `grpo.group_size`.

- `math.yaml`
  This is the example-side config. It owns things like:
  dataset candidates, `train_split`, `eval_split`, default dataset limits,
  default checkpoint path, and default eval batch size.

That separation keeps the example-specific knobs out of the main FlashRL run
schema without inventing a new top-level YAML section.

## Prerequisites

Install the project dependencies first:

```bash
pip install -e .
```

That includes `datasets`, which this example uses to load GSM8K.

If you want the managed vLLM path, use one of these setups:

```bash
pip install -e '.[vllm]'
```

or prepare a dedicated runtime:

```bash
./dev.sh vllm setup
```

Notes:

- Training and evaluation need access to the configured model weights and the
  GSM8K dataset from Hugging Face unless those assets are already cached
  locally.
- TensorBoard metrics are on by default:

```bash
tensorboard --logdir logs
```

- Pushgateway metrics are optional. To bring up the local stack:

```bash
./dev.sh metrics up
```

Training still runs if the optional Pushgateway stack is unavailable.

## Primary Commands

### Canonical managed vLLM training

```bash
python3 -m examples.reasoning.train
```

Equivalent explicit form:

```bash
python3 -m examples.reasoning.train \
  --config examples/reasoning/config_vllm.yaml \
  --task-config examples/reasoning/math.yaml
```

### Local Hugging Face debug training

```bash
python3 -m examples.reasoning.train --config examples/reasoning/config.yaml
```

Use this when you want a smaller local run and `serving.debug_live_rollout`
support for rollout debugging.

### Held-out evaluation

```bash
python3 -m examples.reasoning.eval
```

By default, `eval.py` will try to load the checkpoint path from `math.yaml` if
that file exists locally. If no checkpoint is found there, it evaluates the
base model from the selected FlashRL profile.

Explicit checkpoint form:

```bash
python3 -m examples.reasoning.eval \
  --config examples/reasoning/config_vllm.yaml \
  --task-config examples/reasoning/math.yaml \
  --checkpoint /tmp/flashrl_reasoning_checkpoint.pt
```

### Advanced direct YAML entrypoint

The framework entrypoint still works, but it uses the default `math.yaml`
because only the FlashRL profile is passed directly:

```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config.yaml
```

```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config_vllm.yaml
```

## Run-Time Knobs

### Training

Useful flags:

```bash
python3 -m examples.reasoning.train \
  --config examples/reasoning/config.yaml \
  --task-config examples/reasoning/math.yaml \
  --train-limit 64 \
  --checkpoint-out /tmp/flashrl_reasoning_checkpoint.pt
```

Available flags:

- `--config`: FlashRL runtime/training profile.
- `--task-config`: math-example sidecar config.
- `--train-limit`: cap the number of GSM8K training questions.
- `--checkpoint`: load a checkpoint before training.
- `--checkpoint-out`: save the final checkpoint to a specific path.

### Evaluation

Useful flags:

```bash
python3 -m examples.reasoning.eval \
  --config examples/reasoning/config.yaml \
  --task-config examples/reasoning/math.yaml \
  --eval-limit 50 \
  --batch-size 4
```

Available flags:

- `--config`: FlashRL runtime/training profile.
- `--task-config`: math-example sidecar config.
- `--checkpoint`: load a checkpoint before evaluation.
- `--eval-limit`: cap the number of held-out GSM8K questions.
- `--batch-size`: override the eval batch size from `math.yaml`.

## Controlling How Many Questions Are Used

Three knobs matter, and they do different things:

- Dataset size: how many distinct GSM8K questions are loaded.
- Batch shape: how many prompts/completions are sampled per optimizer step.
- Epoch count: how many passes training makes over the loaded questions.

Use them in this order:

1. CLI flags
   `--train-limit` and `--eval-limit` are the clearest way to cap dataset size.
2. `math.yaml`
   `default_train_limit` and `default_eval_limit` become the example defaults.
3. Environment fallback
   `FLASHRL_REASONING_TRAIN_LIMIT`, `FLASHRL_REASONING_TEST_LIMIT`, and
   `FLASHRL_REASONING_DATASET_LIMIT` still work if no CLI flag or `math.yaml`
   default is set.

Examples:

```bash
python3 -m examples.reasoning.train --train-limit 100
```

```bash
python3 -m examples.reasoning.eval --eval-limit 50
```

Environment fallback example:

```bash
FLASHRL_REASONING_DATASET_LIMIT=32 python3 -m examples.reasoning.train --config examples/reasoning/config.yaml
```

`FLASHRL_REASONING_DATASET_LIMIT` means "use at most this many GSM8K questions
when no split-specific override is set." `FLASHRL_REASONING_TRAIN_LIMIT` is the
train override, and `FLASHRL_REASONING_TEST_LIMIT` is the held-out eval
override.

Related FlashRL profile knobs:

- `training.batch_size` changes total sampled completions per optimizer step.
- `grpo.group_size` changes how many completions are sampled per prompt.
- `training.max_epochs` changes how many times the loaded training questions are
  reused.

So the number of distinct questions comes from the limit settings above, while
total training exposure depends on both that limit and `training.max_epochs`.

## Config Guide

| Setting | `config.yaml` | `config_vllm.yaml` |
| --- | --- | --- |
| Model | `Qwen/Qwen2.5-0.5B` | `Qwen/Qwen2.5-1.5B` |
| Serving backend | Hugging Face | `vllm` |
| `debug_live_rollout` | `true` | `false` |
| `grpo.max_new_tokens` | `256` | `768` |
| Intended use | Cheap local debug and plumbing checks | Serious local prototype training |

Important batch semantics:

- `training.batch_size` is the total number of sampled completions per optimizer
  step.
- Prompts per step are `training.batch_size / grpo.group_size`.
- Example: `batch_size: 8` with `group_size: 4` means 2 prompts per optimizer
  step, each with 4 sampled completions.

## Reward And Output Contract

- `accuracy_reward = 1.0` only when the parsed `<answer>` exactly matches the
  normalized GSM8K final answer.
- `format_reward = 0.1` only when the response is exactly one non-empty
  `<think>...</think>` block followed by exactly one non-empty
  `<answer>...</answer>` block, with no extra text after `</answer>`.
- Truncation, missing closing tags, duplicate blocks, or malformed structure
  lose the format reward.
- Evaluation reports `exact_match`, `format_pass_rate`, and `truncation_rate`.

The example does not fall back to parsing answers from free-form text.

## What To Edit First

- `common.model_name`: switch to a different local or remote base model.
- `training.batch_size`: raise or lower total sampled completions per step.
- `training.max_epochs`: control how long the run lasts.
- `grpo.group_size`: change how many completions are sampled per prompt.
- `grpo.max_new_tokens`: increase or cap reasoning length.
- `serving.backend`: choose between local Hugging Face and managed vLLM.
- `serving.runtime_python`: point vLLM mode at a prepared Python runtime.
- `runtime.reference_enabled`: turn on the frozen reference model.
- `grpo.kl_coefficient`: add KL regularization.
- `math.yaml`: change dataset source, split names, default limits, checkpoint
  path, or eval batch size.

## What To Expect

- Base models often start with many zero-reward samples.
- The local Hugging Face path is slower but better for live rollout debugging.
- The vLLM path is the serious local training path and does not support
  `debug_live_rollout`.
- Default configs are prototype-scale and meant for local experimentation, not
  benchmark reproduction.
- Training saves a checkpoint to the configured `checkpoint_path` from
  `math.yaml` unless `--checkpoint-out` overrides it.
- Run artifacts are written under `logs/` as `console.log`, `events.jsonl`, and
  `rollouts.jsonl`.

## Troubleshooting

- Missing `datasets`: install project dependencies with `pip install -e .`.
- Model or dataset download failures: make sure the configured model and GSM8K
  dataset are accessible from the current machine or already cached.
- Missing `FLASHRL_VLLM_PYTHON`: run `./dev.sh vllm setup`, set the env var
  directly, or install the optional `vllm` extra in the current environment.
- vLLM with `debug_live_rollout: true`: this is not supported; use
  `config.yaml` for the local Hugging Face path when you want live rollout
  debugging.
