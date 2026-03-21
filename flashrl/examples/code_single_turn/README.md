# FlashRL Code-Single-Turn Example

This example is a strict R1-style Codeforces prototype with local execution
reward. It uses a single-turn `Agent`, requires exactly one
`<think>...</think>` block followed by one `<answer>...</answer>` block, and
expects the answer block to contain one fenced Python code block.

## Files

- `train.py`
- `eval.py`
- `executor.py`
- `config.yaml`
- `config-vllm.yaml`

`config.yaml` is the default local Hugging Face config.

`config-vllm.yaml` is the managed local vLLM variant for the same example.

## Local Runs

Default local Hugging Face path:

```bash
python3 -m flashrl.examples.code_single_turn.train
```

Managed local vLLM:

```bash
python3 -m flashrl.examples.code_single_turn.train --config flashrl/examples/code_single_turn/config-vllm.yaml
python3 -m flashrl.examples.code_single_turn.eval --config flashrl/examples/code_single_turn/config-vllm.yaml
```

Custom execution limits:

```bash
python3 -m flashrl.examples.code_single_turn.train \
  --config flashrl/examples/code_single_turn/config-vllm.yaml \
  --run-timeout-seconds 3 \
  --train-limit 64
```

Smaller local run:

```bash
python3 -m flashrl.examples.code_single_turn.train \
  --train-limit 64 \
  --rating-max 1400 \
  --training-mode reasoning-code
```

## Notes

- This example is local-first. It does not ship a documented Kubernetes config yet.
- It still does not support direct raw `FlashRL.from_yaml(...)` usage because the rollout and reward are constructed explicitly in code.
- `FLASHRL_VLLM_PYTHON` is auto-filled by the example entrypoint when the selected config uses `serving.backend: vllm` and a prepared local runtime is found.
- TensorBoard logs are written under `logs/`.
- Training writes generated code and reward artifacts under `logs/<run-id>/generated_code/` by default. Use `--log-dir /absolute/path` to write them elsewhere.
