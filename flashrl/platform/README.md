# FlashRL Platform

FlashRL platform mode is raw-Kubernetes first.

The primary workflow is:

1. build images locally
2. install the operator from committed YAML
3. render one `FlashRLJob`
4. apply it with `kubectl`

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
kubectl apply -f flashrl/platform/k8s/crd.yaml
kubectl apply -f flashrl/platform/k8s/operator-rbac.yaml
kubectl apply -f flashrl/platform/k8s/operator.yaml
```

The committed `operator.yaml` references `flashrl-operator:dev`, matching the
local build command above.

What this does:

- `crd.yaml` installs the `FlashRLJob` custom resource
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
