# FlashRL

FlashRL is a learning-first RL project for LLM post-training. The repo now has
two primary workflows:

- local runs from one example `config.yaml`
- Kubernetes runs by rendering one `FlashRLJob` and applying it with `kubectl`

## Install

```bash
pip install -e '.[dev,platform]'
```

For managed local vLLM serving, also install:

```bash
pip install -e '.[vllm]'
```

## Local Examples

Math example:

```bash
python3 -m flashrl.examples.math.train
python3 -m flashrl.examples.math.train --profile vllm
python3 -m flashrl.examples.math.eval --profile vllm
```

Code example:

```bash
python3 -m flashrl.examples.code_single_turn.train
python3 -m flashrl.examples.code_single_turn.train --profile vllm
python3 -m flashrl.examples.code_single_turn.eval --profile vllm
```

Example docs:

- [flashrl/examples/README.md](flashrl/examples/README.md)
- [flashrl/examples/math/README.md](flashrl/examples/math/README.md)
- [flashrl/examples/code_single_turn/README.md](flashrl/examples/code_single_turn/README.md)

## Kubernetes Workflow

The platform path is raw-Kubernetes first:

1. build images locally
2. install the operator from committed YAML
3. render one `FlashRLJob`
4. apply it with `kubectl`

Build images:

```bash
docker build -f docker/operator.Dockerfile -t flashrl-operator:dev .
docker build -f docker/runtime.Dockerfile -t flashrl-runtime:dev .
docker build -f docker/serving-vllm.Dockerfile -t flashrl-serving-vllm:dev .
docker build -f docker/training-fsdp.Dockerfile -t flashrl-training-fsdp:dev .
```

Install the operator:

```bash
kubectl apply -f flashrl/platform/k8s/namespace.yaml
kubectl apply -f flashrl/platform/k8s/job-crd.yaml
kubectl apply -f flashrl/platform/k8s/operator-rbac.yaml
kubectl apply -f flashrl/platform/k8s/operator.yaml
```

The committed `operator.yaml` points at `flashrl-operator:dev`, so the image
you built in step 1 is the image Kubernetes starts for the operator pod.

Render one job:

```bash
python3 -m flashrl platform render \
  --config flashrl/examples/math/config.yaml \
  --profile minikube \
  --output flashrl-job.yaml
```

Apply and inspect it:

```bash
kubectl apply -f flashrl-job.yaml
kubectl get flashrljobs
kubectl describe flashrljob flashrl-math-minikube
kubectl logs -l flashrl.dev/job=flashrl-math-minikube
```

Operator lifecycle:

- `kubectl apply -f flashrl/platform/k8s/operator.yaml` creates a Kubernetes `Deployment`
- Kubernetes starts the operator pod from `flashrl-operator:dev`
- the runtime images are not started until you apply a `FlashRLJob`

See [flashrl/platform/README.md](flashrl/platform/README.md) for the full
platform install and run flow.

## Observability

TensorBoard is the default local metrics path:

```bash
tensorboard --logdir logs
```

The optional local metrics stack is still available:

```bash
./dev.sh metrics up
./dev.sh metrics down
./dev.sh metrics reset
```

Useful local URLs:

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Pushgateway: `http://localhost:9091`
