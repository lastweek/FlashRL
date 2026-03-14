# FlashRL Reasoning Example

This example is a strict R1-Zero-style math prototype over GSM8K. It trains a
base Qwen model with rule-based rewards only, uses no system prompt, and
requires the model to emit exactly one `<think>...</think>` block followed by
exactly one `<answer>...</answer>` block.

It is a math-only prototype for local experimentation. It is not a full
DeepSeek-R1 reproduction, does not include cold-start SFT, and does not include
code-task verification.

Run the commands below from the repository root.

## What It Supports

| Capability | What it does today |
| --- | --- |
| Local Hugging Face training | In-process training with `config.yaml`; slower, but supports `debug_live_rollout`. |
| Managed local vLLM training | Canonical serious local path with `config_vllm.yaml`; launches managed local `vllm serve` replicas. |
| Held-out evaluation | `eval.py` runs exact-match and format metrics on GSM8K test. |
| Run inspection | Training writes `console.log`, `events.jsonl`, and `rollouts.jsonl` under `logs/`. |
| Viewer support | Open `docs/viewer.html` to inspect run history and grouped rollouts. |
| Optional metrics | Works with the local Grafana/Prometheus/Pushgateway stack from `./dev.sh metrics up`. |
| Smoke runs | Limit train/test dataset size with environment variables. |

## Files In This Folder

- `train.py`: training entrypoint plus dataset, rollout, and reward hooks.
- `eval.py`: held-out evaluation entrypoint for GSM8K test.
- `config.yaml`: cheap local Hugging Face debug profile.
- `config_vllm.yaml`: canonical managed vLLM training profile.

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

- Both training and evaluation need access to the configured model weights and
  the GSM8K dataset from Hugging Face unless those assets are already cached
  locally.
- Metrics are optional. To bring up the local stack:

```bash
./dev.sh metrics up
```

Training still runs if the metrics stack is unavailable.

## Run Modes

### 1. Canonical managed vLLM training

This is the default behavior of `python3 -m examples.reasoning.train`.

```bash
python3 -m examples.reasoning.train
```

Equivalent explicit form:

```bash
python3 -m examples.reasoning.train --config examples/reasoning/config_vllm.yaml
```

Use this when you want the intended local training path with managed vLLM
serving and longer generations.

### 2. Local Hugging Face debug training

```bash
python3 -m examples.reasoning.train --config examples/reasoning/config.yaml
```

Use this when you want a smaller local run and `serving.debug_live_rollout`
support for token-level rollout debugging.

### 3. YAML entrypoint alternatives

The same example can be run directly from the framework YAML entrypoint:

```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config.yaml
```

```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config_vllm.yaml
```

### 4. Held-out evaluation

Default vLLM-profile evaluation:

```bash
python3 -m examples.reasoning.eval --checkpoint /tmp/flashrl_reasoning_checkpoint.pt
```

Evaluate a checkpoint on GSM8K test:

```bash
python3 -m examples.reasoning.eval --config examples/reasoning/config_vllm.yaml --checkpoint /tmp/flashrl_reasoning_checkpoint.pt
```

For the local Hugging Face path:

```bash
python3 -m examples.reasoning.eval --config examples/reasoning/config.yaml --checkpoint /tmp/flashrl_reasoning_checkpoint.pt
```

If `--checkpoint` is omitted, evaluation runs against the base model from the
selected config.

### 5. Smoke runs with small datasets

Limit both train and test splits:

```bash
FLASHRL_REASONING_DATASET_LIMIT=32 python3 -m examples.reasoning.train --config examples/reasoning/config.yaml
```

Or control splits independently:

```bash
FLASHRL_REASONING_TRAIN_LIMIT=64 FLASHRL_REASONING_TEST_LIMIT=128 python3 -m examples.reasoning.eval --config examples/reasoning/config.yaml
```

`FLASHRL_REASONING_DATASET_LIMIT` means "use at most this many GSM8K questions
from each split when no split-specific override is set." For example,
`FLASHRL_REASONING_DATASET_LIMIT=32` makes training read at most 32 questions
from GSM8K train, and makes evaluation read at most 32 questions from GSM8K test
unless you override one split with `FLASHRL_REASONING_TRAIN_LIMIT`,
`FLASHRL_REASONING_TEST_LIMIT`, or `--limit`.

## Controlling How Many Questions Are Used

There are three separate levers here:

- Dataset size: how many distinct GSM8K questions the run loads.
- Batch shape: how many prompts/completions are sampled per optimizer step.
- Epoch count: how many passes training makes over the loaded train questions.

To control the number of training questions:

- Set `FLASHRL_REASONING_TRAIN_LIMIT=N` to train on the first `N` GSM8K train
  questions.
- Or set `FLASHRL_REASONING_DATASET_LIMIT=N` if you want the same cap to apply
  to both training and evaluation by default.
- If neither is set, training uses the full GSM8K train split.

Examples:

```bash
FLASHRL_REASONING_TRAIN_LIMIT=100 python3 -m examples.reasoning.train --config examples/reasoning/config.yaml
```

```bash
FLASHRL_REASONING_DATASET_LIMIT=500 python3 -m examples.reasoning.train
```

Related knobs that do something different:

- `training.batch_size` does not change how many questions are in the dataset.
  It changes how many sampled completions are produced per optimizer step.
- `grpo.group_size` controls how many completions are sampled for each prompt.
- `training.max_epochs` controls how many times the loaded training questions are
  reused.

So the number of distinct training questions comes from the dataset limit
environment variables, while total training exposure depends on both that limit
and `training.max_epochs`.

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

## What To Edit First

- `common.model_name`: switch to a different local or remote base model.
- `training.batch_size`: raise or lower total sampled completions per step.
- `training.max_epochs`: control how long the run lasts.
- `grpo.group_size`: change how many completions are sampled per prompt.
- `grpo.max_new_tokens`: increase or cap reasoning length.
- `serving.backend`: choose between local Hugging Face and managed vLLM.
- `serving.runtime_python`: point vLLM mode at a prepared Python runtime.
- `runtime.reference_enabled`: turn on the frozen reference model.
- `grpo.kl_coefficient`: add KL regularization; pair this with
  `runtime.reference_enabled: true` if you want the reference path to be active.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `FLASHRL_VLLM_PYTHON` | Python executable for managed vLLM mode. |
| `FLASHRL_REASONING_DATASET_LIMIT` | Shared default cap on the number of GSM8K questions loaded from each split. |
| `FLASHRL_REASONING_TRAIN_LIMIT` | Train split override for the number of GSM8K training questions. |
| `FLASHRL_REASONING_TEST_LIMIT` | Test split override for the number of GSM8K evaluation questions. |

`train.py` auto-populates `FLASHRL_VLLM_PYTHON` for `config_vllm.yaml` when it
can discover a prepared runtime, including the default `~/.venv-vllm` and
`~/.venv-vllm-metal` locations.

Limit precedence:

- Training uses `FLASHRL_REASONING_TRAIN_LIMIT` first, then
  `FLASHRL_REASONING_DATASET_LIMIT`, then the full GSM8K train split.
- Evaluation uses `--limit` first, then `FLASHRL_REASONING_TEST_LIMIT`, then
  `FLASHRL_REASONING_DATASET_LIMIT`, then the full GSM8K test split.

## Reward And Output Contract

- `accuracy_reward = 1.0` only when the parsed `<answer>` exactly matches the
  normalized GSM8K final answer.
- `format_reward = 0.1` only when the response is exactly one non-empty
  `<think>...</think>` block followed by exactly one non-empty
  `<answer>...</answer>` block, with no extra text after `</answer>`.
- Truncation, missing closing tags, duplicate blocks, or malformed structure
  lose the format reward.
- Evaluation reports `exact_match`, `format_pass_rate`, and
  `truncation_rate`.

The prompt contract is strict by design. The example does not reward nice
formatting outside that contract, and it does not fall back to parsing answers
from free-form text.

## How Evaluation Works

`python3 -m examples.reasoning.eval` runs held-out evaluation on the GSM8K test
split, not the training split.

What it does:

- loads the test questions with the same prompt template used for training
- optionally loads a checkpoint with `--checkpoint`; if omitted, it evaluates the
  base model from the selected config
- switches generation to deterministic settings for evaluation:
  `temperature=0.0`, `do_sample=false`, `top_p=1.0`, `top_k=0`
- keeps the configured `max_new_tokens`
- scores each completion with the same strict reward parser used during training

Evaluation does not decoder-stop on `</answer>`. The prompt and reward contract
still expect nothing after `</answer>`, so trailing text can make a sample lose
the format reward.

What the reported metrics mean:

- `sample_count`: number of test questions evaluated
- `reward_mean`: average total reward across evaluated questions
- `exact_match`: fraction of questions whose `<answer>` exactly matches the
  normalized GSM8K final answer
- `format_pass_rate`: fraction of questions whose response matches the strict
  `<think>...</think><answer>...</answer>` contract
- `truncation_rate`: fraction of questions that ended with `finish_reason ==
  "length"`

Use `--limit` when you want a smaller evaluation run regardless of the
environment variables:

```bash
python3 -m examples.reasoning.eval --config examples/reasoning/config.yaml --checkpoint /tmp/flashrl_reasoning_checkpoint.pt --limit 50
```

## Outputs And Inspection

After training, FlashRL writes a new run directory under `logs/` with files such
as:

- `console.log`: compact console transcript for the run.
- `events.jsonl`: structured runtime and training events.
- `rollouts.jsonl`: grouped prompt/completion/reward records.

The training entrypoint also saves a checkpoint to:

```text
/tmp/flashrl_reasoning_checkpoint.pt
```

To inspect runs visually, open `docs/viewer.html` in Chrome or Edge and point
the `Run History` workspace at the `logs/` directory.

`eval.py` prints compact JSON metrics to stdout.

## What To Expect

- Base models usually start with many zero-reward samples.
- The local Hugging Face path is slower, but it is the easier path for live
  rollout debugging.
- The vLLM path is the serious local training path and does not support
  `debug_live_rollout`.
- The shipped configs are prototype-scale. They are suitable for local
  experimentation, not benchmark reproduction.
- If you run `python3 -m examples.reasoning.train` with no flags, it defaults to
  `config_vllm.yaml`.

## Troubleshooting

- `examples.reasoning requires the datasets package`: install project
  dependencies with `pip install -e .`.
- Model or dataset download failures: make sure the configured model and GSM8K
  are reachable from Hugging Face, or use a locally cached/local model path.
- vLLM runtime errors: set `FLASHRL_VLLM_PYTHON` to a prepared runtime, install
  the optional `vllm` extra in the current environment, or run
  `./dev.sh vllm setup`.
- `debug_live_rollout` errors with vLLM: this is expected in v1; use
  `config.yaml` if you need live rollout debugging.
