# FlashRL Code-Single-Turn Example

This example is a strict R1-style Codeforces prototype with local execution
reward. It trains a base Qwen coder model on `open-r1/codeforces`
`verifiable-prompts`, uses FlashRL's built-in `Agent` for a single-turn rollout,
uses no system prompt, and requires exactly one
`<think>...</think>` block followed by one `<answer>...</answer>` block that
contains exactly one fenced Python code block.

It is a simple, local, single-turn prototype:

- built-in `Agent`, but no multi-step agent loop
- Python only
- official tests first
- default `rating <= 1600`
- local subprocess execution with best-effort limits

It is not a hardened hostile-code sandbox, and it does not support direct
`FlashRL.from_yaml(...)` runs.

Run the commands below from the repository root.

## Files In This Folder

- `train.py`: main Codeforces training entrypoint and the shared single-turn `Agent` builder.
- `eval.py`: held-out evaluation entrypoint using the same `Agent`.
- `executor.py`: local Python execution helper for official tests and checkers.
- `config.yaml`: cheap local Hugging Face profile.
- `config_vllm.yaml`: canonical managed local vLLM profile.

## Primary Commands

### Canonical managed vLLM training

```bash
python3 flashrl/framework/examples/code-single-turn/train.py
```

Equivalent explicit form:

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml
```

### Local Hugging Face debug training

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config.yaml
```

## Common Run Scripts

### Quick Debug Runs (64 Samples)

**Codeforces, Code Mode (no thinking tags), Easy Problems (rating <= 1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode code \
  --train-limit 64 \
  --rating-max 1400
```

**Codeforces, Reasoning-Code Mode (requires thinking tags), Easy Problems (rating <= 1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode reasoning-code \
  --train-limit 64 \
  --rating-max 1400
```

**Codeforces, Medium Problems (rating <= 1600, default):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode code \
  --train-limit 64 \
  --rating-max 1600
```

### Medium Training Runs (256 Samples)

**Codeforces, Code Mode, Easy Problems (rating <= 1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode code \
  --train-limit 256 \
  --rating-max 1400
```

**Codeforces, Reasoning-Code Mode, Easy Problems (rating <= 1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode reasoning-code \
  --train-limit 256 \
  --rating-max 1400
```

**Codeforces, Medium Problems (rating <= 1600):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode code \
  --train-limit 256 \
  --rating-max 1600
```

### Local Debug Runs (Smaller Model)

**Quick debug with 0.5B Coder model (code mode):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config.yaml \
  --training-mode code \
  --train-limit 64 \
  --rating-max 1400
```

### Quick Evaluation Runs

**Evaluate trained checkpoint (32 samples, code mode, rating <= 1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode code \
  --checkpoint /tmp/flashrl_code_single_turn_checkpoint.pt \
  --eval-limit 32 \
  --rating-max 1400
```

**Evaluate trained checkpoint (32 samples, reasoning-code mode, rating <= 1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --training-mode reasoning-code \
  --checkpoint /tmp/flashrl_code_single_turn_checkpoint.pt \
  --eval-limit 32 \
  --rating-max 1400
```

**Evaluate trained checkpoint (50 samples, rating <= 1600):**

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --checkpoint /tmp/flashrl_code_single_turn_checkpoint.pt \
  --eval-limit 50 \
  --rating-max 1600
```

**Evaluate base model without checkpoint:**

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --eval-limit 32 \
  --rating-max 1400
```

### Batch Size Variations

**Larger batch for faster training (batch_size=8, 4 prompts per step):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 256 \
  --rating-max 1400 \
  --batch-size 8 \
  --group-size 4
```

**Smaller batch for debugging (batch_size=4, 2 prompts per step):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --batch-size 4 \
  --group-size 2
```

### Execution Time Limits

**Fast execution (3 second timeout):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --run-timeout-seconds 3 \
  --memory-limit-mb 256
```

**Standard execution (10 second timeout):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --run-timeout-seconds 10 \
  --memory-limit-mb 512
```

### Problem Difficulty Ranges

**Very Easy Problems (rating <= 1200):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 128 \
  --rating-max 1200
```

**Easy Problems (rating 800-1400):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 128 \
  --rating-min 800 \
  --rating-max 1400
```

**Medium Problems (rating 1200-1800):**

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --train-limit 128 \
  --rating-min 1200 \
  --rating-max 1800
```

### Held-out evaluation

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py
```

Explicit checkpoint form:

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --checkpoint /tmp/flashrl_code_single_turn_checkpoint.pt
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
Because this folder is intentionally named `code-single-turn`, it does not use
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

## Training Modes

This example supports two training modes:

### Code Mode (Default)

```bash
python3 flashrl/framework/examples/code-single-turn/train.py --training-mode code
```

- Evaluates only code correctness
- No `<thinking>` tags required
- Uses simple prompt asking for Python code
- Reward: 0.0 to 1.0 (pass rate only)
- Best for: Pure code capability training
- Extracts code from anywhere in response

### Reasoning-Code Mode

```bash
python3 flashrl/framework/examples/code-single-turn/train.py --training-mode reasoning-code
```

- Evaluates both format compliance and code correctness
- Requires ````thinking``` tags and `<answer>` blocks
- Uses strict format prompt
- Reward: 0.0 to 1.1 (pass rate + format bonus)
- Format bonus: `+0.1` only when strict format passes
- Best for: Training models to show reasoning process before code
- Enforces structured output format

**When to use code mode (default):**

- When you only care about code correctness
- When you want faster training without format evaluation
- When you're focusing on pure coding capability
- When format restrictions get in the way of learning

**When to use reasoning-code mode:**

- When you want model to learn structured reasoning with code
- When you care about reasoning process, not just final code
- When you're training models to explain their approach before coding

## Run-Time Knobs

### Training

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config.yaml \
  --train-limit 64 \
  --rating-max 1400 \
  --max-tests-per-problem 4 \
  --run-timeout-seconds 3 \
  --memory-limit-mb 256
```

Available flags:

- `--config`
- `--train-limit`
- `--training-mode`: choose `code` (default) or `reasoning-code`.
- `--rating-min`
- `--rating-max`
- `--run-timeout-seconds`
- `--memory-limit-mb`
- `--max-tests-per-problem`
- `--log-dir`: directory to save generated code and rewards (default: `generated_code/`)

Training checkpointing is configured in YAML. The shipped example configs already
save a final checkpoint under `run_dir/checkpoints/final.pt`.

Example:

```yaml
checkpointing:
  save_on_run_end: true
  # Default final checkpoint: run_dir/checkpoints/final.pt
  # Optional explicit override:
  # final_path: /path/to/final.pt
  # Optional explicit resume:
  # resume_from: /path/to/final.pt
  # Optional managed latest resume:
  # directory: logs/code-single-turn-checkpoints
  # resume_from: latest
```

When `config_vllm.yaml` is used, synced serving weights for the run are kept
under `run_dir/vllm/weights`.

## Generated Code Logging

This example automatically logs generated Python code and reward information to help with debugging and training analysis. Logging is enabled by default and saves to a `generated_code/` directory in the current working directory.

### Log Files

Each scored rollout creates a file named `{task_id}_{timestamp}.txt` containing:

- Task ID
- Timestamp
- Reward value
- Pass rate and test results (passed/total)
- Format pass status
- Execution status
- Execution time
- Generated code preview

### Example Log File

```text
Task ID: codeforces-train-863/A
Timestamp: 20260317_213202_243931
Log File: codeforces-train-863_A_20260317_213202_243931.txt
Reward: 1.1000
Pass Rate: 1.0000
Passed/Total Tests: 5/5
Format Pass: True
Execution Status: passed
Execution Time: 0.182s

================================================================================
GENERATED CODE:
================================================================================
import sys
print(sum(map(int, sys.stdin.read().split())))
```

**Note**: Special characters in task IDs (like `/` and `\`) are replaced with `_` in the log filename to avoid filesystem issues. The original task ID is preserved in the file content for reference.

### Custom Log Directory

To use a custom log directory:

```bash
python3 flashrl/framework/examples/code-single-turn/train.py \
  --config flashrl/framework/examples/code-single-turn/config_vllm.yaml \
  --log-dir /path/to/custom/logs
```

### Evaluation

Available flags:

```bash
python3 flashrl/framework/examples/code-single-turn/eval.py \
  --config flashrl/framework/examples/code-single-turn/config.yaml \
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

### Code Mode

The prompt requires:

- Python code solution (no format restrictions)
- Code can appear anywhere in response

Reward:

- `reward = passed_tests / total_tests` (0.0 to 1.0)

So:

- fully correct -> `1.0`
- partially correct -> `0.5`
- completely wrong -> `0.0`

### Reasoning-Code Mode

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

### Common Metrics

- `reward_mean`
- `pass_rate_mean`
- `solve_rate`
- `format_pass_rate`
- `truncation_rate`

During training and evaluation, each scored rollout also prints a compact local
execution summary such as:

```text
code     task=codeforces-train-1000A  mode=reasoning-code  tests=1/1  pass_rate=1.00  format=ok  reward=1.10  exec=0.182s  status=passed
preview  import sys | print(sum(map(int, sys.stdin.read().split())))
```

For code mode, the output will look like:

```text
code     task=codeforces-train-1000A  mode=code  tests=1/1  pass_rate=1.00  reward=1.00  exec=0.182s  status=passed
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
