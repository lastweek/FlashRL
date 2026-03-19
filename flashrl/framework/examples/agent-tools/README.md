# Agent Tools Demo

This is a small whitebox example for the built-in `ReActRollout`.

It demonstrates:

- explicit `system_prompt` in user code
- subprocess-backed tool execution
- parallel tool calls in one assistant step
- transcript output with `assistant` and `tool` messages

Run it from the repository root:

```bash
python3 flashrl/framework/examples/agent-tools/run.py
```

The script uses a small scripted serving backend so the example stays offline
and focused on the agent/tool API rather than model quality.
