# FlashRL Examples

The canonical example surface is `flashrl.examples`.

## Primary Agent Ladder

Start with these four examples in order:

| Example | Teaches | Use it when | Intentionally not covered |
| --- | --- | --- | --- |
| `agent_tools` | Basic `Agent` loop, fixed tools, traced assistant/tool messages | You want the smallest end-to-end whitebox agent example | Dynamic tool gating, advanced managers, assembled harness behavior |
| `agent_dynamic_tools` | Dynamic `tools(state)` gating and context control | You understand the basic loop and need step-local tool visibility | Skills, compaction, subagents, study orchestration |
| `agent_harness` | A serious coding-oriented reference harness assembled from framework primitives | You want a usable reference system rather than another toy loop | Harness comparison workflow |
| `agent_harness_ablation` | Controlled variant comparison for the reference harness | You want to compare harness configurations under one training/eval workflow | New agent-building primitives or another assembled harness |

Other examples:

- `math`: training-integrated whitebox rollout example
- `code_single_turn`: single-turn code reasoning baseline

## Module-First Commands

Primary ladder:

```bash
python3 -m flashrl.examples.agent_tools.run
python3 -m flashrl.examples.agent_dynamic_tools.run
python3 -m flashrl.examples.agent_harness.train
python3 -m flashrl.examples.agent_harness_ablation.train
```

Other examples:

```bash
python3 -m flashrl.examples.math.train
python3 -m flashrl.examples.math.train --config flashrl/examples/math/config-vllm.yaml
python3 -m flashrl.examples.code_single_turn.train
```

Example config layout is now explicit:

- `config.yaml` is the normal default config for that example
- `config-vllm.yaml` is the managed local vLLM variant when the example ships one
- cluster smoke configs live under `flashrl/platform/dev/`, not under public examples

When a selected example config uses `framework.serving.backend: vllm`, the
entrypoint will try to auto-fill `FLASHRL_VLLM_PYTHON` when a prepared runtime
is available.

Example docs:

- [agent_tools/README.md](agent_tools/README.md)
- [agent_dynamic_tools/README.md](agent_dynamic_tools/README.md)
- [agent_harness/README.md](agent_harness/README.md)
- [agent_harness_ablation/README.md](agent_harness_ablation/README.md)
- [math/README.md](math/README.md)
- [code_single_turn/README.md](code_single_turn/README.md)

Local observability:

- TensorBoard is the default path: `tensorboard --logdir logs`
- optional Pushgateway metrics use `metrics.pushgateway.enabled: true`
- local stack helpers remain `./dev.sh metrics up`, `./dev.sh metrics down`, and `./dev.sh metrics reset`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Pushgateway: `http://localhost:9091`
