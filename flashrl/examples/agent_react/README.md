# Agent ReAct Example

This example shows how to build a reusable ReAct recipe directly in example
code without relying on any public preset API.

Positioning:

- this is an optional recipe/reference example
- it is not part of the primary agent learning ladder
- use it when you specifically want a ReAct parsing pattern in plain Python

It demonstrates:

- `Agent(run_fn=...)`
- a system prompt that carries the full ReAct contract
- `Agent.build_prompt(...)` as the completion-ready prompt helper
- explicit ReAct parsing in normal Python code
- reusable local helper functions that can be copied into user code

Run it from the repository root:

```bash
python3 -m flashrl.examples.agent_react.run
```
