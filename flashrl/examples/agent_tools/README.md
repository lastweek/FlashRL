# Agent Tools Demo

This is a small whitebox example built directly from the core `Agent`
building blocks.

It demonstrates:

- an explicit custom `run_fn(agent)`
- a system message that carries the tool-use contract
- `Agent.build_prompt(...)` as the completion-ready prompt helper
- traced assistant and tool messages
- subprocess-backed tool execution
- parallel tool calls in one assistant step
- transcript output with `assistant` and `tool` messages

Run it from the repository root:

```bash
python3 -m flashrl.examples.agent_tools.run
```

The script uses a small scripted serving backend so the example stays offline
and focused on the agent/tool API rather than model quality.
