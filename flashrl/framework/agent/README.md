# FlashRL Agent Building Blocks

`flashrl.framework.agent` is the public toolbox for building whitebox agents.

It provides generic building blocks. The assembled reference system lives in
`flashrl.examples.agent_harness`, and the controlled comparison workflow lives
in `flashrl.examples.agent_harness_ablation`.

The mental model is:

- `Agent` is both the rollout you pass to `FlashRL(..., rollout_fn=...)`
- and the live object you use inside `run_fn(agent)`
- FlashRL training routes `Agent` directly, while `agent([...], backend)` remains a manual convenience
- system messages carry the agent policy and response contract
- `Tool` and `SubprocessToolRuntime` are optional tool-execution building blocks
- `ToolRegistry`, `ToolProfile`, and `AgentToolExecutor` support serious tool surfaces
- `SkillManager`, `CompactionManager`, and `SubagentManager` support richer harnesses
- `WindowedContextManager` remains the simplest context helper

Typical usage:

1. Create an `Agent(run_fn=..., tools=..., max_steps=...)`
2. Inside `run_fn(agent)`, call:
   - `agent.add_message("system", "...")` for policy / contract text
   - `agent.available_tools()`
   - `agent.build_prompt(...)`
   - `agent.generate(prompt)`
   - `agent.record_generation(sample, ...)`
   - `agent.run_tools(...)` or `agent.finish(...)`
3. Pass the `Agent` to `FlashRL(..., rollout_fn=...)`

`Agent.build_prompt(...)` returns a completion-ready prompt. It renders the
current visible tools, the current transcript, any optional footer, and then a
final `Assistant:` cue so completion-style serving backends continue at the
assistant turn.

The examples under `flashrl.examples` show this progression:

1. `flashrl.examples.agent_tools`: learn the smallest custom `Agent` loop with fixed tools
2. `flashrl.examples.agent_dynamic_tools`: add dynamic tool gating and context management
3. `flashrl.examples.agent_harness`: inspect the assembled reference agent harness
4. `flashrl.examples.agent_harness_ablation`: compare harness variants systematically

Additional examples:

- `flashrl.examples.math`: training-integrated whitebox rollout example
