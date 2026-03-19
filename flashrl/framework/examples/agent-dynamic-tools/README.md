# Agent Dynamic Tools Example

This example shows a general-purpose agent loop with two important extensions:

- dynamic `tools(state)` gating
- `WindowedContextManager` for history control

It stays offline and deterministic, but the runtime shape is the same one used
for larger whitebox agents.

Run it from the repository root:

```bash
python3 flashrl/framework/examples/agent-dynamic-tools/run.py
```
