# FlashRL Platform Architecture

This page is the quickest way to build a mental model of FlashRL platform mode.
It focuses on three questions:

1. how many images exist and what each image is for
2. what logical layers live inside each image
3. how a `config.yaml` turns into running training on Kubernetes

## Mental Model

There are four separate concerns in platform mode:

- local CLI and config compilation
  This is where `flashrl platform render` or `flashrl platform submit` runs.
- operator control plane
  This is the long-lived Kubernetes controller that watches `FlashRLJob`.
- job workloads
  These are the controller, learner, serving, rollout, and reward pods created for one job.
- framework service layer
  These are the remote clients, in-process services, and shared HTTP route helpers inside the pods that actually move rollout, reward, optimize, and serve requests.
- observability
  These are the shared controller run artifacts, per-pod platform log files, and bounded per-job operator events that make one Kubernetes run debuggable.

The important split is:

- the operator manages Kubernetes objects
- the controller pod runs training
- the other job pods expose services the controller talks to
- controller training artifacts follow the same framework run lifecycle in local and platform modes

Inside `flashrl.framework.distributed`, the current code shape is:

- `*_client.py`
  Remote callers used by the controller or by one pod calling another pod.
- `*_service.py`
  Domain-owned in-process service logic plus the `create_*_service_app(...)` FastAPI app builders.
- `http_common.py`
  Shared HTTP lifecycle, readiness, drain, status, and admin route wiring.

One more important split now exists around observability:

- `flashrl.framework.train_runtime`
  Shared run-scoped controller lifecycle for `RunLogger`, metrics sinks, checkpoint load/save, and run finalization.
- `flashrl.platform.runtime.platform_pod_logging`
  Platform-owned pod logging for split service pods and shim startup/runtime events.
- `FlashRLJob.status`
  Per-job `logRoot`, active controller run metadata, and recent operator/controller events.

## What Runs Where

Workload to image mapping:

- `flashrl-operator`
  Runs the Kubernetes operator deployment only.
- `flashrl-runtime`
  Runs the job controller, rollout, and reward pods.
- `flashrl-serving-vllm`
  Runs the serving pod.
- `flashrl-training-fsdp`
  Runs the learner pod.

That means there is no separate controller image. Controller, rollout, and reward share the runtime image.

## What Platform Adds Per Pod

| Pod | Platform software in the pod | What platform adds | Framework software used | User hook or backend |
| --- | --- | --- | --- | --- |
| controller | `PlatformShimController` in `flashrl.platform.runtime.platform_shim_controller` | Load mounted `FlashRLJob`, resolve sibling service URLs, merge live CRD status, patch controller-owned status, resolve checkpoints, start background training loop, and open the shared framework run logger/metrics lifecycle under a job-scoped log root | `flashrl.framework.train_runtime`, `RunLogger`, metrics sinks, `RolloutClient`, `RewardClient`, `LearnerClient`, `ServingClient`, `GRPOController`, `install_common_routes` from `http_common.py`, controller status routes | dataset hook or dataset URI |
| rollout | `PlatformShimRollout` in `flashrl.platform.runtime.platform_shim_rollout` | Load mounted job, instantiate rollout hook, create remote serving client, wire rollout generator into the rollout service | `ServingClient` from `flashrl.framework.distributed`, `RemoteServingBackend` from `flashrl.framework.serving`, `build_rollout_generator`, `RolloutService`, `create_rollout_service_app` from `flashrl.framework.rollout` | `userCode.rollout` |
| reward | `PlatformShimReward` in `flashrl.platform.runtime.platform_shim_reward` | Load mounted job and instantiate the reward hook, then wire it into the reward service | `UserDefinedReward`, `RewardService`, `create_reward_service_app` from `flashrl.framework.reward` | `userCode.reward` |
| learner | `PlatformShimLearner` in `flashrl.platform.runtime.platform_shim_learner` | Load mounted job, create actor/reference backends, resolve shared storage paths, publish learner artifacts | `create_training_backend`, `LearnerService`, `create_learner_service_app` from `flashrl.framework.training` | training backends from framework config |
| serving | `PlatformShimServing` in `flashrl.platform.runtime.platform_shim_serving` | Load mounted job, resolve shared artifact paths, create the configured serving backend, expose serving RPCs | `create_serving_backend`, `ServingService`, `create_serving_service_app`, `RemoteServingBackend` from `flashrl.framework.serving` | serving backend from framework config |

## Image Inventory

```mermaid
flowchart LR
    subgraph Images
        OP[flashrl-operator]
        RT[flashrl-runtime]
        SV[flashrl-serving-vllm]
        TR[flashrl-training-fsdp]
    end

    subgraph Workloads
        OPD[operator Deployment]
        CTRL[controller Deployment]
        ROLL[rollout Deployment]
        REW[reward Deployment]
        SERV[serving Deployment]
        LEARN[learner StatefulSet]
    end

    OP --> OPD
    RT --> CTRL
    RT --> ROLL
    RT --> REW
    SV --> SERV
    TR --> LEARN
```

Interpretation: Kubernetes starts the operator from `flashrl-operator`. When the operator reconciles a `FlashRLJob`, it creates the controller, rollout, reward, learner, and serving workloads, each of which pulls one of the remaining three images.

## Layers Inside an Image

```mermaid
flowchart TB
    subgraph RuntimeImage[flashrl-runtime]
        R1[container ENTRYPOINT: flashrl]
        R2[pod command: controller or rollout or reward]
        R3[platform runtime PlatformShim]
        R4[distributed clients plus domain service app builders]
        R5[user hooks and framework logic]
        R1 --> R2 --> R3 --> R4 --> R5
    end

    subgraph ServingImage[flashrl-serving-vllm]
        S1[container ENTRYPOINT: flashrl]
        S2[pod command: serving]
        S3[platform runtime PlatformShim]
        S4[serving domain service app builder]
        S5[serving backend: huggingface or vllm]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    subgraph TrainingImage[flashrl-training-fsdp]
        T1[container ENTRYPOINT: flashrl]
        T2[pod command: learner]
        T3[platform runtime PlatformShim]
        T4[learner domain service app builder]
        T5[training backend: huggingface or fsdp2]
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph OperatorImage[flashrl-operator]
        O1[container ENTRYPOINT: flashrl]
        O2[command: platform operator]
        O3[operator facade and reconcile loop]
        O4[Kubernetes API]
        O1 --> O2 --> O3 --> O4
    end
```

Interpretation: each image has a thin CLI/runtime entry layer on top of the real service or operator logic. The runtime images mostly bootstrap a concrete implementation and then hand off to distributed transport clients, domain service app builders, and shared HTTP helpers.

## Execution Workflow

```mermaid
sequenceDiagram
    participant User
    participant CLI as flashrl platform CLI
    participant API as Kubernetes API
    participant Operator as flashrl-operator
    participant Ctrl as controller pod
    participant Roll as rollout pod
    participant Rew as reward pod
    participant Learn as learner pod
    participant Serv as serving pod
    participant PVC as shared storage

    User->>CLI: flashrl platform render --config config.yaml
    CLI->>CLI: load config.yaml
    CLI->>CLI: build FlashRLJob

    User->>API: kubectl apply operator manifests
    API->>Operator: start operator Deployment
    Operator->>API: ensure FlashRLJob CRD exists

    User->>API: kubectl apply flashrl-job.yaml
    API->>Operator: watch FlashRLJob event
    Operator->>Operator: reconcile job
    Operator->>API: create ConfigMap, Services, PVC, controller, learner, serving, rollout, reward

    API->>Ctrl: start pod with `flashrl controller`
    API->>Roll: start pod with `flashrl rollout`
    API->>Rew: start pod with `flashrl reward`
    API->>Learn: start pod with `flashrl learner`
    API->>Serv: start pod with `flashrl serving`

    Ctrl->>Ctrl: load mounted FlashRLJob
    Roll->>Roll: load mounted FlashRLJob
    Rew->>Rew: load mounted FlashRLJob
    Learn->>Learn: load mounted FlashRLJob
    Serv->>Serv: load mounted FlashRLJob

    Ctrl->>Roll: rollout requests
    Roll->>Serv: generation requests
    Ctrl->>Rew: reward requests
    Ctrl->>Learn: optimize requests
    Learn->>PVC: publish weights/checkpoints
    Ctrl->>Serv: activate weight version
    Ctrl->>API: patch job status/progress
```

Interpretation: submitting a `FlashRLJob` does not itself run training. The operator reacts to the job, creates the workloads, and Kubernetes starts the pods. Training begins only after the controller pod comes up and its startup path launches the GRPO loop.

## Inside Each Pod

Every workload pod follows the same broad pattern:

1. Kubernetes starts the container with a `flashrl ...` command
2. a `PlatformShim*` class in `flashrl.platform.runtime` bootstraps the pod-specific implementation
3. the domain service modules expose the HTTP service surface through `create_*_service_app(...)`
4. the hook or backend implementation does the real work

In the diagrams below:

- `platform shim common` means `flashrl.platform.runtime.platform_shim_common`
- `platform shim` means one of `flashrl.platform.runtime.platform_shim_controller|platform_shim_rollout|platform_shim_reward|platform_shim_learner|platform_shim_serving`
- `framework transport` means `flashrl.framework.distributed`
- `framework service` means the domain-owned `flashrl.framework.rollout|reward|training|serving` service modules

That gives each workload the same three-layer picture: `PlatformShim*` for pod bootstrap, domain service plus distributed transport code for the RPC boundary, and hook/backend code for the actual work.

### Controller Pod

The controller pod owns orchestration and training, so its runtime has both an init path and a long-running execution path.
Kubernetes naming: `container=controller`, workload/pod prefix=`<job>-controller`.
Platform modules: `PlatformShimController` in `flashrl.platform.runtime.platform_shim_controller` plus shared `flashrl.platform.runtime.platform_shim_common`.

#### Init Workflow

```mermaid
sequenceDiagram
    participant Container as Controller Container
    participant PodContract as flashrl.platform.runtime.platform_shim_common
    participant Platform as PlatformShimController
    participant Framework as flashrl.framework.distributed.http_common plus GRPOController
    participant API as Kubernetes API

    Container->>Platform: flashrl controller
    Platform->>Platform: PlatformShimController.create_app
    Platform->>Framework: install_common_routes
    Platform->>Platform: FastAPI startup hook fires
    Platform->>Platform: background thread starts
    Platform->>PodContract: load_mounted_job
    Platform->>API: fetch live FlashRLJob status
    Platform->>Platform: merge mounted spec with live status
    Platform->>PodContract: service_url_for(rollout/reward/learner/serving)
    Platform->>Framework: build RolloutClient / RewardClient / LearnerClient / ServingClient
    Platform->>Framework: construct GRPOController
```

#### Execution Workflow

```mermaid
sequenceDiagram
    participant PodContract as flashrl.platform.runtime.platform_shim_common
    participant Platform as PlatformShimController
    participant Framework as flashrl.framework.distributed plus GRPOController
    participant Rollout as rollout service
    participant Reward as reward service
    participant Learner as learner service
    participant Serving as serving service
    participant Storage as shared storage
    participant API as Kubernetes API

    Platform->>Platform: load dataset
    Platform->>Framework: GRPOController.train(...)
    Framework->>Rollout: rollout requests
    Rollout-->>Framework: rollout batches
    Framework->>Reward: reward requests
    Reward-->>Framework: scored rewards
    Framework->>Learner: /v1/optimize-steps
    Learner-->>Framework: optimize result
    Framework->>Serving: status / activate weight version
    Platform->>PodContract: storage_path_from_uri(checkpoints)
    Framework->>Storage: save checkpoints
    Platform->>API: patch progress, checkpoint, weightVersion
```

Interpretation: the controller pod is different from every other pod because `PlatformShimController` does not just expose one adapter. It installs common HTTP routes through `http_common.py`, then launches a background training loop that coordinates the other services and patches job status back into Kubernetes.

## Observability And Logs

Platform mode now uses the same controller run lifecycle as local framework mode.

- `framework.logging.log_dir` is the canonical logging knob in both modes
- local mode writes directly under that configured log root
- platform mode resolves it into one job-scoped root:
  - relative `log_dir`: `<shared-mount>/<log_dir>/<job-name>/<job-uid>/`
  - absolute `log_dir`: `<absolute-log-dir>/<job-name>/<job-uid>/`

Within that job root, the controller uses the normal framework run artifact layout:

- `<job-root>/<run-id>/console.log`
- `<job-root>/<run-id>/events.jsonl`
- `<job-root>/<run-id>/rollouts.jsonl`

Platform-only pod logs live alongside those controller run artifacts:

- `<job-root>/_pods/controller/<pod-name>/console.log`
- `<job-root>/_pods/controller/<pod-name>/events.jsonl`
- `<job-root>/_pods/rollout/<pod-name>/...`
- `<job-root>/_pods/reward/<pod-name>/...`
- `<job-root>/_pods/learner/<pod-name>/...`
- `<job-root>/_pods/serving/<pod-name>/...`

The operator stays cluster-scoped, but it still contributes job-scoped visibility:

- bounded per-job events are appended into `FlashRLJob.status.events`
- `FlashRLJob.status.logRoot` points at the canonical job log root
- `FlashRLJob.status.activeControllerRunDir` and `activeControllerRunId` point at the active controller run
- the controller also writes JSONL status snapshots under `<job-root>/_status/status-snapshots.jsonl`

That gives one supported debugging flow:

1. inspect job status and log root with `flashrl platform logs <job>`
2. follow live CRD job events and one component stream with `flashrl platform logs <job> --follow --component controller`
3. inspect durable files under the shared job root for postmortem analysis

### Rollout Pod

The rollout pod is an adapter-plus-hook service: platform wires the hook and remote serving client together, while the framework app builder inside `rollout_service.py` owns the HTTP surface.
Kubernetes naming: `container=rollout`, workload/pod prefix=`<job>-rollout`.
Platform modules: `PlatformShimRollout` in `flashrl.platform.runtime.platform_shim_rollout` plus shared `flashrl.platform.runtime.platform_shim_common`.

#### Init Workflow

```mermaid
sequenceDiagram
    participant Container as Rollout Container
    participant PodContract as flashrl.platform.runtime.platform_shim_common
    participant Platform as PlatformShimRollout
    participant Framework as flashrl.framework.distributed
    participant Hook as rollout hook

    Container->>Platform: flashrl rollout
    Platform->>PodContract: load_mounted_job
    Platform->>Hook: instantiate userCode.rollout
    Platform->>PodContract: service_url_for(serving)
    Platform->>Framework: create ServingClient
    Platform->>Framework: wrap remote serving backend
    Platform->>Framework: build rollout generator
    Platform->>Framework: create_rollout_service_app
```

#### Execution Workflow

```mermaid
sequenceDiagram
    participant Controller
    participant Framework as framework distributed
    participant Hook as rollout hook and generator
    participant Serving as serving service

    Controller->>Framework: /v1/rollout-batches
    Framework->>Hook: rollout_batch
    Hook->>Serving: grouped generation request
    Serving-->>Hook: generated outputs
    Hook-->>Framework: rollout batch
    Framework-->>Controller: batched rollouts
```

Interpretation: steady-state rollout execution is request-driven. The controller calls the rollout service, `PlatformShimRollout` has already wired the serving client and hook together, and the framework app hands the request into `RolloutService`.

### Reward Pod

The reward pod is the simplest workflow: it boots one reward implementation and then scores batches on demand.
Kubernetes naming: `container=reward`, workload/pod prefix=`<job>-reward`.
Platform modules: `PlatformShimReward` in `flashrl.platform.runtime.platform_shim_reward` plus shared `flashrl.platform.runtime.platform_shim_common`.

#### Init Workflow

```mermaid
sequenceDiagram
    participant Container as Reward Container
    participant PodContract as flashrl.platform.runtime.platform_shim_common
    participant Platform as PlatformShimReward
    participant Framework as flashrl.framework.distributed
    participant Hook as reward hook

    Container->>Platform: flashrl reward
    Platform->>PodContract: load_mounted_job
    Platform->>Hook: instantiate userCode.reward
    Platform->>Platform: build UserDefinedReward
    Platform->>Framework: create_reward_service_app
```

#### Execution Workflow

```mermaid
sequenceDiagram
    participant Controller
    participant Framework as framework distributed
    participant Hook as reward implementation

    Controller->>Framework: /v1/reward-batches
    Framework->>Hook: reward_batch
    Hook-->>Framework: scored rewards
    Framework-->>Controller: reward response
```

Interpretation: the reward pod is the cleanest boundary split. `PlatformShimReward` bootstraps the reward object once, and every request after that stays on the `RewardService` plus reward implementation path.

### Learner Pod

The learner pod is backend-driven rather than hook-driven. It boots training backends and then handles optimize and checkpoint RPCs.
Kubernetes naming: `container=learner`, workload/pod prefix=`<job>-learner`.
Platform modules: `PlatformShimLearner` in `flashrl.platform.runtime.platform_shim_learner` plus shared `flashrl.platform.runtime.platform_shim_common`.

#### Init Workflow

```mermaid
sequenceDiagram
    participant Container as Learner Container
    participant PodContract as flashrl.platform.runtime.platform_shim_common
    participant Platform as PlatformShimLearner
    participant Framework as flashrl.framework.distributed
    participant Backend as training backends
    participant Storage as shared storage

    Container->>Platform: flashrl learner
    Platform->>PodContract: load_mounted_job
    Platform->>Backend: create actor backend
    Platform->>Backend: create optional reference backend
    Platform->>PodContract: storage_path_from_uri(weights)
    PodContract->>Storage: resolve weights/checkpoint paths
    Platform->>Framework: construct LearnerService
    Platform->>Framework: create_learner_service_app
```

#### Execution Workflow

```mermaid
sequenceDiagram
    participant Controller
    participant Framework as framework distributed
    participant Backend as training backend
    participant Storage as shared storage

    Controller->>Framework: /v1/optimize-steps
    Framework->>Backend: optimize_step
    Backend->>Storage: publish weight artifact
    Backend-->>Framework: optimize result
    Framework-->>Controller: optimize response
    Controller->>Framework: /v1/checkpoints/save or load
    Framework->>Backend: save/load checkpoint
    Framework-->>Controller: checkpoint response
```

Interpretation: after startup, the learner pod is a pure RPC service around training backends. The `LearnerService` module owns both the in-process behavior and the HTTP app builder, while the backend does optimization and writes artifacts to shared storage.

### Serving Pod

The serving pod is also backend-driven. It boots one serving backend, then handles generation and weight-activation requests.
Kubernetes naming: `container=serving`, workload/pod prefix=`<job>-serving`.
Platform modules: `PlatformShimServing` in `flashrl.platform.runtime.platform_shim_serving` plus shared `flashrl.platform.runtime.platform_shim_common`.

#### Init Workflow

```mermaid
sequenceDiagram
    participant Container as Serving Container
    participant PodContract as flashrl.platform.runtime.platform_shim_common
    participant Platform as PlatformShimServing
    participant Framework as flashrl.framework.distributed
    participant Backend as serving backend
    participant Storage as shared storage

    Container->>Platform: flashrl serving
    Platform->>PodContract: load_mounted_job
    Platform->>Backend: create serving backend from framework config
    Platform->>PodContract: storage_path_from_uri(weights)
    PodContract->>Storage: resolve shared artifact path
    Platform->>Framework: construct ServingService
    Platform->>Framework: create_serving_service_app
```

#### Execution Workflow

```mermaid
sequenceDiagram
    participant Rollout
    participant Controller
    participant Framework as framework distributed
    participant Backend as serving backend

    Rollout->>Framework: /v1/generate-grouped
    Framework->>Backend: generate_grouped
    Backend-->>Framework: grouped generations
    Framework-->>Rollout: generation response
    Controller->>Framework: /v1/activate-weight-version
    Framework->>Backend: load and switch active weights
    Backend-->>Framework: activation result
    Framework-->>Controller: activation response
```

Interpretation: the serving pod spends most of its life handling two RPCs: grouped generation for rollout and active-weight switching for the controller. `PlatformShimServing` matters during initialization; the steady-state path is `ServingService` plus serving backend.

Across all five pods, the recurring pattern is stable:

- controller is orchestration-heavy
- rollout and reward are hook-heavy
- learner and serving are backend-heavy

## How Training Actually Starts

The key startup sequence is:

1. the operator is already running independently as a Kubernetes deployment
2. a `FlashRLJob` is created
3. the operator renders and applies the child workloads
4. Kubernetes starts the controller pod
5. the controller pod runs `flashrl controller`
6. the controller runtime loads the mounted `FlashRLJob`, builds HTTP clients to rollout, reward, learner, and serving, then calls into the GRPO controller

So:

- `flashrl platform submit` or `kubectl apply -f flashrl-job.yaml` creates the job resource
- the operator deployment reacts to that resource
- the controller pod is where training actually begins

## Where To Read Next

- [flashrl/platform/README.md](../flashrl/platform/README.md) for install and run flow
- [README.md](../README.md) for the top-level Kubernetes workflow
