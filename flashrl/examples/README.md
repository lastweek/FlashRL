# FlashRL Examples

The canonical example surface is `flashrl.examples`.

Module-first commands:

```bash
python3 -m flashrl.examples.math.train
python3 -m flashrl.examples.math.train --profile vllm
python3 -m flashrl.examples.code_single_turn.train
python3 -m flashrl.examples.agent_tools.run
python3 -m flashrl.examples.agent_react.run
python3 -m flashrl.examples.agent_dynamic_tools.run
```

Cluster-capable training examples use one `config.yaml` with:

- `framework:` for FlashRL run semantics
- `platform:` for Kubernetes policy and image refs
- `profiles:` for environment-specific overrides such as `vllm` and `minikube`

When a local example uses the managed vLLM profile, the entrypoint will try to
auto-fill `FLASHRL_VLLM_PYTHON` when a prepared runtime is available.

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
