# FlashRL Examples

The canonical example surface is `flashrl.examples`.

Module-first commands:

```bash
python3 -m flashrl.examples.math.train
python3 -m flashrl.examples.math.train --config flashrl/examples/math/config-vllm.yaml
python3 -m flashrl.examples.code_single_turn.train
python3 -m flashrl.examples.agent_tools.run
python3 -m flashrl.examples.agent_react.run
python3 -m flashrl.examples.agent_dynamic_tools.run
```

Example config layout is now explicit:

- `config.yaml` is the normal default config for that example
- `config-vllm.yaml` is the managed local vLLM variant when the example ships one
- cluster smoke configs live under `flashrl/platform/dev/`, not under public examples

When a selected example config uses `framework.serving.backend: vllm`, the
entrypoint will try to auto-fill `FLASHRL_VLLM_PYTHON` when a prepared runtime
is available.

Example docs:

- [math/README.md](math/README.md)
- [code_single_turn/README.md](code_single_turn/README.md)
- [mock_checkpointing/README.md](mock_checkpointing/README.md)

Local observability:

- TensorBoard is the default path: `tensorboard --logdir logs`
- optional Pushgateway metrics use `metrics.pushgateway.enabled: true`
- local stack helpers remain `./dev.sh metrics up`, `./dev.sh metrics down`, and `./dev.sh metrics reset`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Pushgateway: `http://localhost:9091`
