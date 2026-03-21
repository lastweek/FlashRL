# FlashRL Reference Agent Harness

This example assembles a usable coding-oriented agent harness from the generic
`flashrl.framework.agent` building blocks.

It is intentionally the reference example system, not a framework API. The
companion ablation example imports this harness and varies only its config.

Role in the ladder:

- read this after `agent_dynamic_tools`
- use it when you want a serious assembled system built from the generic framework primitives
- it is the reference harness, not the harness-comparison workflow

## What It Uses

- `Agent` as the rollout object
- `ToolRegistry` and `ToolProfile`
- `AgentToolExecutor`
- `SkillManager`
- `CompactionManager`
- `SubagentManager`

## Files

- `harness.py`
- `config.py`
- `tool_helpers.py`
- `train.py`
- `eval.py`
- `config.yaml`
- `config-vllm.yaml`
- `fixtures/`
- `skills/`

## Local Runs

Default local path:

```bash
python3 -m flashrl.examples.agent_harness.train
python3 -m flashrl.examples.agent_harness.eval
```

Managed local vLLM:

```bash
python3 -m flashrl.examples.agent_harness.train --config flashrl/examples/agent_harness/config-vllm.yaml
python3 -m flashrl.examples.agent_harness.eval --config flashrl/examples/agent_harness/config-vllm.yaml
```

## Notes

- Tasks are deterministic local repo-inspection fixtures, not an external dataset.
- Reward is task correctness only.
- The harness records tool, skill, compaction, subagent, and scheduler activity in `agent_trace`.
- The public framework still remains generic; this package is the coding-oriented reference assembly.
