# FlashRL Platform

FlashRL platform mode is raw-Kubernetes first.

The primary workflow is:

1. build images locally
2. install the operator from committed YAML
3. render one `FlashRLJob`
4. apply it with `kubectl`

## Code Map

- `job.py`
  The `FlashRLJob` API, schema, status model, and CRD manifest generator.
- `config.py`
  The config compiler from `config.yaml` into a validated `FlashRLJob`.
- `k8s/renderer.py`
  Pure rendering of child Kubernetes resources for one job.
- `k8s/operator.py`
  The reconcile loop, autoscaling, recovery, and status aggregation.
- `runtime/common.py`
  Shared pod-runtime helpers such as job loading, service URL resolution, and shared paths.
- `runtime/components.py`
  FastAPI app factories for rollout, reward, learner, and serving pods.
- `runtime/controller.py`
  The controller runtime that drives GRPO over HTTP clients.
- `runtime/cli.py`
  The `flashrl component run ...` entrypoint used by pod commands.
- `dev/minikube.py`
  The local minikube E2E helper used by the opt-in smoke path.

## Build Images

```bash
docker build -f docker/operator.Dockerfile -t flashrl-operator:dev .
docker build -f docker/runtime.Dockerfile -t flashrl-runtime:dev .
docker build -f docker/serving-vllm.Dockerfile -t flashrl-serving-vllm:dev .
docker build -f docker/training-fsdp.Dockerfile -t flashrl-training-fsdp:dev .
```

## Install The Operator

```bash
kubectl apply -f flashrl/platform/k8s/namespace.yaml
kubectl apply -f flashrl/platform/k8s/job-crd.yaml
kubectl apply -f flashrl/platform/k8s/operator-rbac.yaml
kubectl apply -f flashrl/platform/k8s/operator.yaml
```

The committed `operator.yaml` references `flashrl-operator:dev`, matching the
local build command above.

What this does:

- `job-crd.yaml` installs the `FlashRLJob` custom resource
- `operator-rbac.yaml` installs the service account and RBAC the operator needs
- `operator.yaml` creates the operator `Deployment`
- Kubernetes then starts the operator pod from `flashrl-operator:dev`

The runtime images are not started yet. They are only created after you apply a
`FlashRLJob`. The rendered job YAML carries the runtime image refs from the
selected config profile under `platform.images`.

## Render A Job

The repo examples now use one `config.yaml` with:

- `framework:` for FlashRL semantics
- `platform:` for Kubernetes policy
- `profiles:` for overrides such as `vllm` and `minikube`

Render one job:

```bash
python3 -m flashrl platform render \
  --config flashrl/examples/math/config.yaml \
  --profile minikube \
  --output flashrl-job.yaml
```

## Apply And Inspect

```bash
kubectl apply -f flashrl-job.yaml
kubectl get flashrljobs
kubectl describe flashrljob flashrl-math-minikube
kubectl logs -l flashrl.dev/job=flashrl-math-minikube
```

Convenience helpers still exist:

```bash
python3 -m flashrl platform status flashrl-math-minikube --namespace flashrl-e2e
python3 -m flashrl platform describe flashrl-math-minikube --namespace flashrl-e2e
python3 -m flashrl platform cancel flashrl-math-minikube --namespace flashrl-e2e
```

## Minikube Smoke Run

```bash
python3 scripts/run_minikube_math_e2e.py
```

That helper:

- builds the local images into minikube
- installs the CRD and operator from `flashrl/platform/k8s/`
- renders the math example with `--profile minikube`
- submits the job
- waits for at least one completed training step
