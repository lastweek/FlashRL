# FlashRL Code-Single-Turn Example

## What This Example Is

This example is a single-turn Codeforces baseline built on FlashRL's built-in
`Agent`.

It supports two prompt/response modes:

- `code`: plain code generation
- `reasoning-code`: strict `<think>...</think><answer>...</answer>` output with
  one fenced Python code block inside `<answer>`

The default CLI mode is `code`. `reasoning-code` is optional and must be
selected explicitly with `--training-mode reasoning-code`.

## Files

- `train.py`
- `eval.py`
- `executor.py`
- `config.yaml`
- `config-vllm.yaml`

`config.yaml` is the default local Hugging Face config.

`config-vllm.yaml` is the managed local vLLM variant for the same example.

## Quick Start

Default local Hugging Face path:

```bash
python3 -m flashrl.examples.code_single_turn.train
python3 -m flashrl.examples.code_single_turn.eval
```

The default local config is CPU-first for reliability on Apple Silicon and
other low-memory local setups. Explicit `device: mps` is now an opt-in path.

## Recommended First Smoke Run

Use a small filtered run first so dataset loading, rollout logs, and local code
execution stay readable:

```bash
python3 -m flashrl.examples.code_single_turn.train \
  --train-limit 16 \
  --rating-max 1400 \
  --max-tests-per-problem 2

python3 -m flashrl.examples.code_single_turn.eval \
  --eval-limit 8 \
  --rating-max 1400 \
  --max-tests-per-problem 2
```

## Reasoning-Code Mode

Use this when you want the strict R1-style output contract:

```bash
python3 -m flashrl.examples.code_single_turn.train \
  --training-mode reasoning-code \
  --train-limit 16

python3 -m flashrl.examples.code_single_turn.eval \
  --training-mode reasoning-code \
  --eval-limit 8
```

## Managed Local vLLM

The same example can run against the managed local vLLM config:

```bash
python3 -m flashrl.examples.code_single_turn.train --config flashrl/examples/code_single_turn/config-vllm.yaml
python3 -m flashrl.examples.code_single_turn.eval --config flashrl/examples/code_single_turn/config-vllm.yaml
```

You can combine that with a smaller smoke configuration:

```bash
python3 -m flashrl.examples.code_single_turn.train \
  --config flashrl/examples/code_single_turn/config-vllm.yaml \
  --run-timeout-seconds 3 \
  --train-limit 16 \
  --rating-max 1400 \
  --max-tests-per-problem 2
```

## Checkpoints and Outputs

To evaluate a specific checkpoint explicitly:

```bash
python3 -m flashrl.examples.code_single_turn.eval \
  --checkpoint /path/to/checkpoint.pt \
  --eval-limit 8
```

If `--checkpoint` is omitted, evaluation will automatically load
`/tmp/flashrl_reasoning_code_checkpoint.pt` when that file exists.

Outputs:

- training logs are written under `logs/`
- generated code and reward artifacts go under
  `logs/<run-id>/generated_code/` by default
- `--log-dir` overrides only the generated-code artifact location
- evaluation prints compact JSON metrics such as solve rate, pass rate, and
  truncation rate

## Troubleshooting

- Apple Silicon MPS runs can OOM when both actor and serving are placed on
  `mps`, especially with grouped GRPO and long sampled responses.
- The default `config.yaml` pins both actor and serving to `cpu` to make the
  local baseline more reliable.
- If you want to try `mps` anyway, treat it as an explicit opt-in:
  set `device: mps` in the config, lower `grpo.max_new_tokens`, and expect
  higher instability than the CPU default.

## Caveats

- This example requires the `datasets` package.
- It downloads data from `open-r1/codeforces`.
- It runs generated Python code locally against official tests.
- If dataset access or local code execution is unavailable, the example will
  fail early.
- `FLASHRL_VLLM_PYTHON` is auto-filled by the example entrypoint when the
  selected config uses `serving.backend: vllm` and a prepared local runtime is
  found.
- This example is local-first. It does not ship a documented Kubernetes config
  yet.
- It still does not support direct raw `FlashRL.from_yaml(...)` usage because the rollout and reward are constructed explicitly in code.
