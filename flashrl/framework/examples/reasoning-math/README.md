# FlashRL Reasoning-Math Example

This example is a strict R1-Zero-style math prototype with explicit dataset
selection. It trains a base Qwen model with rule-based rewards only, uses no
system prompt, and requires exactly one `<think>...</think>` block followed by
one `<answer>...</answer>` block.

The example keeps one simple split of responsibility:

- `config.yaml` and `config_vllm.yaml` control FlashRL runtime and training
  behavior.
- `train.py` and `eval.py` control dataset choice and the other math-example
  details through explicit CLI flags.

It is a math-only prototype for local experimentation. It is not a full
DeepSeek-R1 reproduction, does not include cold-start SFT, and does not include
code-task verification yet.

Run the commands below from the repository root.

## Files In This Folder

- `train.py`: main math training entrypoint and the example helper functions.
- `eval.py`: held-out evaluation entrypoint for the selected math dataset.
- `config.yaml`: cheap local Hugging Face profile.
- `config_vllm.yaml`: canonical managed local vLLM profile.

## Config Responsibility

Use the files for different jobs:

- `config.yaml` / `config_vllm.yaml`
  These are FlashRL profiles. They own things like:
  `common.model_name`, `training.batch_size`, `training.max_epochs`,
  `serving.backend`, `runtime.reference_enabled`, and `grpo.group_size`.

- `train.py` / `eval.py` CLI flags
  These own math-example details such as dataset limits, checkpoint paths, and
  eval batch size.

That keeps the example-specific knobs visible in `--help` instead of hidden in a
second YAML file.

## Prerequisites

Install the project dependencies first:

```bash
pip install -e .
```

That includes `datasets`, which this example uses to load GSM8K and AIME25.

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
  selected math dataset from Hugging Face unless those assets are already
  cached locally.
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
python3 flashrl/framework/examples/reasoning-math/train.py
```

Equivalent explicit form:

```bash
python3 flashrl/framework/examples/reasoning-math/train.py --config flashrl/framework/examples/reasoning-math/config_vllm.yaml
```

### Local Hugging Face debug training

```bash
python3 flashrl/framework/examples/reasoning-math/train.py --config flashrl/framework/examples/reasoning-math/config.yaml
```

Use this when you want a smaller local run and `serving.debug_live_rollout`
support for rollout debugging.

### Dataset selection

Built-in datasets:

- `gsm8k`: default dataset for both train and eval.
- `aime25`: `math-ai/aime25`, supported for both train and eval on its `test`
  split.

Examples:

```bash
python3 flashrl/framework/examples/reasoning-math/train.py --dataset aime25
```

```bash
python3 flashrl/framework/examples/reasoning-math/eval.py --dataset aime25
```

Before training or evaluation starts, the selected dataset is printed to the
terminal in a compact summary such as:

```text
dataset  name=gsm8k  source=openai/gsm8k  split=train  available=7473  selected=256
format   problem_field=question  answer_field=answer  target=parse #### + numeric normalize
```

For `aime25`, the format line becomes:

```text
format   problem_field=problem  answer_field=answer  target=direct numeric normalize
```

This summary is terminal-only. It is not copied into `console.log`.

### Held-out evaluation

```bash
python3 flashrl/framework/examples/reasoning-math/eval.py
```

By default, `eval.py` will try to load
`/tmp/flashrl_reasoning_checkpoint.pt` if that file exists. If not, it evaluates
the base model from the selected FlashRL profile.

Explicit checkpoint form:

```bash
python3 flashrl/framework/examples/reasoning-math/eval.py \
  --config flashrl/framework/examples/reasoning-math/config_vllm.yaml \
  --checkpoint /tmp/flashrl_reasoning_checkpoint.pt
```

This example is intentionally script-run. Like `reasoning-code`, the folder is
hyphenated, so `train.py` and `eval.py` load the YAML profiles directly and
construct `FlashRL(...)` in code instead of using `FlashRL.from_yaml(...)`.

## Run-Time Knobs

### Training

Useful flags:

```bash
python3 flashrl/framework/examples/reasoning-math/train.py \
  --config flashrl/framework/examples/reasoning-math/config.yaml \
  --dataset aime25 \
  --train-limit 64 \
  --checkpoint-out /tmp/flashrl_reasoning_checkpoint.pt
```

Available flags:

- `--config`: FlashRL runtime/training profile.
- `--dataset`: choose `gsm8k` or `aime25`.
- `--train-limit`: cap the number of training questions from the selected dataset.
- `--checkpoint`: load a checkpoint before training.
- `--checkpoint-out`: save the final checkpoint to a specific path.

### Evaluation

Useful flags:

```bash
python3 flashrl/framework/examples/reasoning-math/eval.py \
  --config flashrl/framework/examples/reasoning-math/config.yaml \
  --dataset aime25 \
  --eval-limit 50 \
  --batch-size 4
```

Available flags:

- `--config`: FlashRL runtime/training profile.
- `--dataset`: choose `gsm8k` or `aime25`.
- `--checkpoint`: load a checkpoint before evaluation.
- `--eval-limit`: cap the number of held-out questions from the selected dataset.
- `--batch-size`: override the default eval batch size.

## Controlling How Many Questions Are Used

Three knobs matter, and they do different things:

- Dataset size: how many distinct math questions are loaded.
- Batch shape: how many prompts/completions are sampled per optimizer step.
- Epoch count: how many passes training makes over the loaded questions.

Use these flags to control dataset size directly:

```bash
python3 flashrl/framework/examples/reasoning-math/train.py --train-limit 100
```

```bash
python3 flashrl/framework/examples/reasoning-math/eval.py --eval-limit 50
```

Related FlashRL profile knobs:

- `training.batch_size` changes total sampled completions per optimizer step.
- `grpo.group_size` changes how many completions are sampled per prompt.
- `training.max_epochs` changes how many times the loaded training questions are
  reused.

So the number of distinct questions comes from `--train-limit` or
`--eval-limit`, while total training exposure depends on both that limit and
`training.max_epochs`.

In the printed dataset summary:

- `available` is the full size of the selected source split before any limit is applied.
- `selected` is the number of rows actually used after `--train-limit` or `--eval-limit`.

Dataset-specific defaults:

- `gsm8k`: train on `train`, evaluate on `test`.
- `aime25`: both `train.py` and `eval.py` use the `test` split of
  `math-ai/aime25`.

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
  normalized dataset answer stored in prompt metadata.
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

## What To Expect

- Base models often start with many zero-reward samples.
- The local Hugging Face path is slower but better for live rollout debugging.
- The vLLM path is the serious local training path and does not support
  `debug_live_rollout`.
- Default configs are prototype-scale and meant for local experimentation, not
  benchmark reproduction.
- Training saves a checkpoint to `/tmp/flashrl_reasoning_checkpoint.pt` unless
  `--checkpoint-out` overrides it.
- Run artifacts are written under `logs/` as `console.log`, `events.jsonl`, and
  `rollouts.jsonl`.

## Troubleshooting

- Missing `datasets`: install project dependencies with `pip install -e .`.
- Model or dataset download failures: make sure the configured model and the
  selected math dataset are accessible from the current machine or already
  cached.
- Missing `FLASHRL_VLLM_PYTHON`: run `./dev.sh vllm setup`, set the env var
  directly, or install the optional `vllm` extra in the current environment.
- vLLM with `debug_live_rollout: true`: this is not supported; use
  `config.yaml` for the local Hugging Face path when you want live rollout
  debugging.
