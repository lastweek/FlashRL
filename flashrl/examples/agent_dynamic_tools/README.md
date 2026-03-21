# Agent Dynamic Tools Example

This example shows a general-purpose agent loop with two important extensions:

Role in the ladder:

- read this after `agent_tools`
- use it when you need step-local tool visibility and history control
- it is intentionally not the assembled reference harness

- dynamic `tools(state)` gating
- `WindowedContextManager` for history control

It stays offline and deterministic, but the runtime shape is the same one used
for larger whitebox agents. It does not cover skills, compaction, subagents, or
study orchestration.

Run it from the repository root:

```bash
python3 -m flashrl.examples.agent_dynamic_tools.run
```
