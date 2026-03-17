# FlashRL Reasoning-Code Example

This example is a strict R1-style Codeforces prototype with local execution
reward. It trains a base Qwen coder model on `open-r1/codeforces`
`verifiable-prompts`, uses no system prompt, and requires exactly one
`<think>...</think>` block followed by one `<answer>...</answer>` block that
contains exactly one fenced Python code block.

It is a simple, local, single-turn prototype:

- Python only
- official tests first
- default `rating <= 1600`
- local subprocess execution with best-effort limits

It is not a hardened hostile-code sandbox, and it does not support direct
`FlashRL.from_yaml(...)` runs.

Run the commands below from the repository root.

## Files In This Folder

- `train.py`: main Codeforces training entrypoint and example logic.
- `eval.py`: held-out evaluation entrypoint.
- `executor.py`: local Python execution helper for official tests and checkers.
- `config.yaml`: cheap local Hugging Face profile.
- `config_vllm.yaml`: canonical managed local vLLM profile.

## Primary Commands

### Canonical managed vLLM training

```bash
python3 flashrl/framework/examples/reasoning-code/train.py
```

Equivalent explicit form:

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml
```

### Local Hugging Face debug training

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config.yaml
```

## Common Run Scripts

### Quick Debug Runs (64 Samples)

**Codeforces, Easy Problems (rating <= 1400):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400
```

**Codeforces, Medium Problems (rating <= 1600, default):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1600
```

### Medium Training Runs (256 Samples)

**Codeforces, Easy Problems (rating <= 1400):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 256 \
  --rating-max 1400
```

**Codeforces, Medium Problems (rating <= 1600):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 256 \
  --rating-max 1600
```

### Local Debug Runs (Smaller Model)

**Quick debug with 0.5B Coder model:**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config.yaml \
  --train-limit 64 \
  --rating-max 1400
```

### Quick Evaluation Runs

**Evaluate trained checkpoint (32 samples, rating <= 1400):**

```bash
python3 flashrl/framework/examples/reasoning-code/eval.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --checkpoint /tmp/flashrl_reasoning_code_checkpoint.pt \
  --eval-limit 32 \
  --rating-max 1400
```

**Evaluate trained checkpoint (50 samples, rating <= 1600):**

```bash
python3 flashrl/framework/examples/reasoning-code/eval.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --checkpoint /tmp/flashrl_reasoning_code_checkpoint.pt \
  --eval-limit 50 \
  --rating-max 1600
```

**Evaluate base model without checkpoint:**

```bash
python3 flashrl/framework/examples/reasoning-code/eval.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --eval-limit 32 \
  --rating-max 1400
```

### Batch Size Variations

**Larger batch for faster training (batch_size=8, 4 prompts per step):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 256 \
  --rating-max 1400 \
  --batch-size 8 \
  --group-size 4
```

**Smaller batch for debugging (batch_size=4, 2 prompts per step):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --batch-size 4 \
  --group-size 2
```

### Execution Time Limits

**Fast execution (3 second timeout):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --run-timeout-seconds 3 \
  --memory-limit-mb 256
```

**Standard execution (10 second timeout):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --run-timeout-seconds 10 \
  --memory-limit-mb 512
```

### Problem Difficulty Ranges

**Very Easy Problems (rating <= 1200):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 128 \
  --rating-max 1200
```

**Easy Problems (rating 800-1400):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 128 \
  --rating-min 800 \
  --rating-max 1400
```

**Medium Problems (rating 1200-1800):**

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --train-limit 128 \
  --rating-min 1200 \
  --rating-max 1800
```

### Held-out evaluation

```bash
python3 flashrl/framework/examples/reasoning-code/eval.py
```

Explicit checkpoint form:

```bash
python3 flashrl/framework/examples/reasoning-code/eval.py \
  --config flashrl/framework/examples/reasoning-code/config_vllm.yaml \
  --checkpoint /tmp/flashrl_reasoning_code_checkpoint.pt
```

## Config Responsibility

- `config.yaml` / `config_vllm.yaml`
  These are FlashRL runtime/training profiles. They own things like
  `actor.model_name`, `trainer.batch_size`, `trainer.max_epochs`,
  `serving.backend`, `reference`, `grpo.group_size`, and
  training checkpoint policy under `checkpointing`.

- `train.py` / `eval.py` CLI flags
  These own Codeforces-specific knobs such as limits, rating filters, execution
  limits, and evaluation checkpoint paths.

The scripts load these profiles directly and construct `FlashRL(...)` in code.
Because this folder is intentionally named `reasoning-code`, it does not use
import-string hooks and does not support direct
`python3 -m flashrl.framework.flashrl --config ...`.

## Dataset Policy

The built-in dataset is:

- source: `open-r1/codeforces`
- config: `verifiable-prompts`
- train split: `train`
- eval split: `test`
- language: `python`
- default difficulty slice: `rating <= 1600`
- test scope: official tests only in v1

Before training or evaluation starts, the selected data is printed to the
terminal in a compact summary such as:

```text
dataset  name=codeforces  source=open-r1/codeforces  config=verifiable-prompts  split=train  available=12000  selected=256
filters  language=python  rating=<= 1600  input_mode=stdio  tests=official  checker=when-provided  max_tests_per_problem=all
```

`available` is the full split size before filtering or limits.
`selected` is the number of problems actually used after filtering and any
`--train-limit` or `--eval-limit`.

This summary is terminal-only. It is not copied into `console.log`.

## Run-Time Knobs

### Training

```bash
python3 flashrl/framework/examples/reasoning-code/train.py \
  --config flashrl/framework/examples/reasoning-code/config.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --max-tests-per-problem 4 \
  --run-timeout-seconds 3 \
  --memory-limit-mb 256
```

Available flags:

- `--config`
- `--train-limit`
- `--rating-min`
- `--rating-max`
- `--run-timeout-seconds`
- `--memory-limit-mb`
- `--max-tests-per-problem`

Training checkpointing is configured in YAML. The shipped example configs already
write a final checkpoint to `/tmp/flashrl_reasoning_code_checkpoint.pt`.

Example:

```yaml
checkpointing:
  save_on_run_end: true
  final_path: /tmp/flashrl_reasoning_code_checkpoint.pt
  # Optional explicit resume:
  # resume_from: /tmp/flashrl_reasoning_code_checkpoint.pt
  # Optional managed latest resume:
  # directory: logs/reasoning-code-checkpoints
  # resume_from: latest
```

### Evaluation

```bash
python3 flashrl/framework/examples/reasoning-code/eval.py \
  --config flashrl/framework/examples/reasoning-code/config.yaml \
  --eval-limit 32 \
  --batch-size 4 \
  --rating-max 1400
```

Available flags:

- `--config`
- `--checkpoint`
- `--eval-limit`
- `--batch-size`
- `--rating-min`
- `--rating-max`
- `--run-timeout-seconds`
- `--memory-limit-mb`
- `--max-tests-per-problem`

## Reward And Output Contract

The prompt requires:

- exactly one `<think>...</think>`
- exactly one `<answer>...</answer>`
- exactly one fenced Python code block inside `<answer>`
- no text after `</answer>`

Reward:

- `correctness_reward = passed_tests / total_tests`
- `format_bonus = 0.1` only when the strict outer format and fenced Python block both pass
- total reward = `correctness_reward + format_bonus`

So:

- fully correct, strict response -> `1.1`
- fully correct, trailing junk -> `1.0`
- wrong code, strict response -> `0.1`
- malformed response or unrunnable code -> usually `0.0`

Evaluation reports:

- `reward_mean`
- `pass_rate_mean`
- `solve_rate`
- `format_pass_rate`
- `truncation_rate`

During training and evaluation, each scored rollout also prints a compact local
execution summary such as:

```text
code     task=codeforces-train-1000A  tests=1/1  pass_rate=1.00  format=ok  reward=1.10  exec=0.182s  status=passed
preview  import sys | print(sum(map(int, sys.stdin.read().split())))
```

This example also promotes `execution_status`, `code_preview`, `pass_rate`,
`failure_reason`, and related checker fields into the candidate reward block in
`rollouts.jsonl`, with uncommon leftovers kept under `reward.metadata`.

## Sandbox Notes

The executor writes one temporary Python file per rollout and runs it with
isolated Python flags such as `-I -B -s`.

Best-effort local limits:

- wall-clock timeout
- temp working directory
- stripped environment
- POSIX resource limits where available

This is a local research/example runner. It is not a hardened hostile-code
sandbox. Only run it in an environment where local code execution is acceptable.

## vLLM Runtime

Like the reasoning math example, this script will auto-populate
`FLASHRL_VLLM_PYTHON` when `config_vllm.yaml` is selected and a prepared local
vLLM runtime is discoverable.

If you want the managed vLLM path, either install:

```bash
pip install -e '.[vllm]'
```

or prepare a dedicated runtime:

```bash
./dev.sh vllm setup
```
