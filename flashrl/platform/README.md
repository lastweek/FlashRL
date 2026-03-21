# FlashRL Platform

FlashRL platform mode is raw-Kubernetes first.

The primary workflow is:

1. build images locally
2. install the operator from committed YAML
3. render one `FlashRLJob`
4. apply it with `kubectl`

If you want the system picture before the code map, start here:

- [docs/platform-architecture.md](../../docs/platform-architecture.md)

## Code Map

- `k8s/job.py`
  The `FlashRLJob` API, schema, status model, and CRD manifest generator.
- `config.py`
  The config compiler from `config.yaml` into a validated `FlashRLJob`.
- `k8s/job_resources.py`
  Explicit rendering of the config, controller, learner, serving, rollout, and reward resources for one job.
- `k8s/operator/__init__.py`
  The small public operator facade with `ensure_crd`, `apply_job`, `get_job`, `delete_job`, `reconcile`, and `watch`.
- `k8s/operator/kube.py`
  Tiny helper for lazy Kubernetes client loading, watch loading, and normalized error/object handling.
- `k8s/operator/status.py`
  Component observation collection plus phase, condition, and CRD status summary logic, using explicit Kubernetes API reads.
- `k8s/operator/scaling.py`
  Autoscaling logic for serving, rollout, and reward pools, using explicit AppsV1 replica patches.
- `k8s/operator/recovery.py`
  Recovery logic for learner and elastic pools, using explicit StatefulSet and Deployment operations.
- `k8s/operator/reconcile.py`
  The explicit reconcile sequence: apply children, observe, scale, recover, summarize, and patch status through raw Kubernetes APIs.
- `runtime/platform_shim_controller.py`
  `PlatformShimController`, the controller-side platform shim that loads the mounted job, builds remote clients, and starts GRPO training.
- `runtime/platform_shim_common.py`
  The literal shared substrate for every `PlatformShim`: load the mounted job, resolve sibling service URLs, and turn storage URIs into container paths.
- `runtime/platform_shim_base.py`
  The tiny `PlatformShim` base class that owns only `create_app()` and `run()`.
- `runtime/cli.py`
  The thin pod-command dispatcher used by `flashrl controller|rollout|reward|learner|serving`; it instantiates the matching `PlatformShim*`.
- `runtime/platform_shim_rollout.py`
  `PlatformShimRollout`: rollout hook plus remote serving client plus rollout HTTP service.
- `runtime/platform_shim_reward.py`
  `PlatformShimReward`: reward hook plus reward HTTP service.
- `runtime/platform_shim_learner.py`
  `PlatformShimLearner`: actor/reference backends plus learner HTTP service.
- `runtime/platform_shim_serving.py`
  `PlatformShimServing`: serving backend plus serving HTTP service.
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
