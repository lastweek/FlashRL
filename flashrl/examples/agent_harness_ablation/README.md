# FlashRL Agent Harness Ablation Study

This example compares named variants of the reference agent harness under one
controlled training and evaluation workflow.

It varies only harness config and seed. Model, trainer, reward function, and
dataset split stay fixed.

Role in the ladder:

- read this after `agent_harness`
- use it when you want to compare harness variants, not build a new harness
- it is a study runner, not another assembled agent system

## Files

- `train.py`
- `eval.py`
- `study.py`
- `config.yaml`
- `config-vllm.yaml`

## Matrix Shape

The study config uses a named row-wise matrix:

- `base_harness`
- `matrix`, a list of named variant rows
- `seeds`

The default variants are:

- `tools_only`
- `tools_skills`
- `tools_compaction`
- `tools_subagents`
- `full_harness`

The study imports the reference harness from `flashrl.examples.agent_harness`
and varies only `AgentHarnessConfig` values.

## Local Runs

Default local path:

```bash
python3 -m flashrl.examples.agent_harness_ablation.train
python3 -m flashrl.examples.agent_harness_ablation.eval
```

The default local config is CPU-first for reliability on Apple Silicon and
other low-memory local setups. Explicit `device: mps` remains an advanced
opt-in, not the recommended local path.

Managed local vLLM:

```bash
python3 -m flashrl.examples.agent_harness_ablation.train --config flashrl/examples/agent_harness_ablation/config-vllm.yaml
python3 -m flashrl.examples.agent_harness_ablation.eval --manifest logs/studies/<study>/manifest.json
```

`config-vllm.yaml` is the performance-oriented local serving variant. It still
keeps the local actor on `cpu` by default.

## Outputs

`train.py` writes a study manifest under `logs/studies/<name>-<timestamp>/manifest.json`.

`eval.py` reads that manifest and writes:

- `summary.json`
- `leaderboard.md`

The report is Pareto-style: accuracy is compared against token cost and
rollout time instead of folding efficiency into the training reward.
