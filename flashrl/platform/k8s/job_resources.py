"""Explicit Kubernetes resource rendering for one FlashRLJob."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from flashrl.platform.k8s.job import GROUP, PLURAL, FlashRLJob, WorkloadSpec


HTTP_PORT = 8000
JOB_WORKLOADS = ("controller", "learner", "serving", "rollout", "reward")
ELASTIC_WORKLOADS = ("serving", "rollout", "reward")


def job_namespace(job: FlashRLJob, fallback: str = "default") -> str:
    """Return the namespace where one job should render its child resources."""
    return str(job.metadata.get("namespace") or fallback)


def job_workload_name(job: FlashRLJob, workload: str) -> str:
    """Return the workload resource name for one job role."""
    return f"{job.name}-{workload}"


def job_workload_selector_labels(job: FlashRLJob, workload: str) -> dict[str, str]:
    """Return the selector labels shared by one workload and its pods."""
    return {
        "app.kubernetes.io/name": "flashrl",
        "app.kubernetes.io/component": workload,
        "flashrl.dev/job": job.name,
    }


def job_workload_spec(job: FlashRLJob, workload: str) -> WorkloadSpec:
    """Return the workload policy block for one job role."""
    if workload == "controller":
        return job.spec.controller
    if workload == "learner":
        return job.spec.learner
    if workload == "serving":
        return job.spec.serving
    if workload == "rollout":
        return job.spec.rollout
    if workload == "reward":
        return job.spec.reward
    raise KeyError(workload)


def job_workload_kind(workload: str) -> str:
    """Return the Kubernetes workload kind for one job role."""
    if workload == "learner":
        return "StatefulSet"
    if workload in {"controller", "serving", "rollout", "reward"}:
        return "Deployment"
    raise KeyError(workload)


def desired_job_workload_replicas(job: FlashRLJob, workload: str) -> int:
    """Return the initial desired replica count for one workload role."""
    if workload == "controller":
        return 0 if job.spec.suspend else 1
    if workload == "learner":
        return int(job.spec.framework.actor.dp_size)
    workload_spec = job_workload_spec(job, workload)
    if workload_spec.replicas is None:
        return 1
    return int(workload_spec.replicas.min)


def render_job_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render the full Kubernetes resource set for one FlashRLJob."""
    resources: list[dict[str, Any]] = []
    resources.extend(render_runtime_config_resources(job))
    resources.extend(render_controller_resources(job))
    resources.extend(render_learner_resources(job))
    resources.extend(render_serving_resources(job))
    resources.extend(render_rollout_resources(job))
    resources.extend(render_reward_resources(job))
    return resources


def render_runtime_config_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render shared runtime configuration resources for one job."""
    resources = [_job_config_map_resource(job)]
    shared_storage_claim = _shared_storage_claim_resource(job)
    if shared_storage_claim is not None:
        resources.append(shared_storage_claim)
    return resources


def render_controller_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render controller-only RBAC, Deployment, and Service resources."""
    return [
        _controller_service_account_resource(job),
        _controller_status_writer_role_resource(job),
        _controller_status_writer_binding_resource(job),
        _controller_deployment_resource(job),
        _service_resource(job, "controller"),
    ]


def render_learner_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render learner StatefulSet, Service, and disruption policy resources."""
    return [
        _learner_stateful_set_resource(job),
        _service_resource(job, "learner", headless=True),
        _pod_disruption_budget_resource(job, "learner", max_unavailable=0),
    ]


def render_serving_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render serving Deployment, Service, and optional disruption policy resources."""
    resources = [
        _serving_deployment_resource(job),
        _service_resource(job, "serving"),
    ]
    serving_spec = job.spec.serving
    if serving_spec.replicas is not None and serving_spec.replicas.min > 1:
        resources.append(_pod_disruption_budget_resource(job, "serving", max_unavailable=1))
    return resources


def render_rollout_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render rollout Deployment, Service, and optional disruption policy resources."""
    resources = [
        _rollout_deployment_resource(job),
        _service_resource(job, "rollout"),
    ]
    rollout_spec = job.spec.rollout
    if rollout_spec.replicas is not None and rollout_spec.replicas.min > 1:
        resources.append(_pod_disruption_budget_resource(job, "rollout", max_unavailable=1))
    return resources


def render_reward_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render reward Deployment, Service, and optional disruption policy resources."""
    resources = [
        _reward_deployment_resource(job),
        _service_resource(job, "reward"),
    ]
    reward_spec = job.spec.reward
    if reward_spec.replicas is not None and reward_spec.replicas.min > 1:
        resources.append(_pod_disruption_budget_resource(job, "reward", max_unavailable=1))
    return resources


def _job_owner_references(job: FlashRLJob) -> list[dict[str, Any]]:
    uid = job.metadata.get("uid")
    if uid is None:
        return []
    return [
        {
            "apiVersion": job.apiVersion,
            "kind": job.kind,
            "name": job.name,
            "uid": str(uid),
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]


def _resource_metadata(
    job: FlashRLJob,
    workload: str,
    *,
    name: str,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": name,
        "labels": {**job_workload_selector_labels(job, workload), **(labels or {})},
    }
    if annotations:
        metadata["annotations"] = dict(annotations)
    owner_references = _job_owner_references(job)
    if owner_references:
        metadata["ownerReferences"] = owner_references
    return metadata


def _job_with_mounted_storage_paths(job: FlashRLJob) -> FlashRLJob:
    if not job.spec.sharedStorage.enabled:
        return job
    runtime_job = job.model_copy(deep=True)
    mount_path = PurePosixPath(runtime_job.spec.sharedStorage.mountPath)
    runtime_job.spec.storage.checkpoints.uriPrefix = str(
        mount_path / runtime_job.spec.sharedStorage.checkpointsSubPath
    )
    runtime_job.spec.storage.weights.uriPrefix = str(
        mount_path / runtime_job.spec.sharedStorage.weightsSubPath
    )
    return runtime_job


def _job_uid(job: FlashRLJob) -> str:
    value = job.metadata.get("uid")
    if value is None:
        return f"{job.name}-pending"
    return str(value)


def _job_log_root(job: FlashRLJob) -> str:
    log_dir = PurePosixPath(str(job.spec.framework.logging.log_dir))
    job_suffix = PurePosixPath(job.name) / _job_uid(job)
    if log_dir.is_absolute():
        return str(log_dir / job_suffix)
    mount_root = PurePosixPath(job.spec.sharedStorage.mountPath if job.spec.sharedStorage.enabled else "/tmp/flashrl-platform-logs")
    return str(mount_root / log_dir / job_suffix)


def _component_log_root(job: FlashRLJob, workload: str) -> str:
    return str(PurePosixPath(_job_log_root(job)) / "_pods" / workload)


def _job_config_map_resource(job: FlashRLJob) -> dict[str, Any]:
    runtime_job = _job_with_mounted_storage_paths(job)
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": _resource_metadata(job, "config", name=job_workload_name(job, "config")),
        "data": {"job.json": runtime_job.model_dump_json(indent=2, by_alias=True)},
    }


def _controller_service_account_name(job: FlashRLJob) -> str:
    return job_workload_name(job, "controller")


def _controller_service_account_resource(job: FlashRLJob) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": _resource_metadata(
            job,
            "controller",
            name=_controller_service_account_name(job),
        ),
    }


def _controller_status_writer_role_resource(job: FlashRLJob) -> dict[str, Any]:
    role_name = job_workload_name(job, "controller-status-writer")
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": _resource_metadata(job, "controller", name=role_name),
        "rules": [
            {
                "apiGroups": [GROUP],
                "resources": [PLURAL],
                "resourceNames": [job.name],
                "verbs": ["get", "patch", "update"],
            },
            {
                "apiGroups": [GROUP],
                "resources": [f"{PLURAL}/status"],
                "resourceNames": [job.name],
                "verbs": ["get", "patch", "update"],
            },
        ],
    }


def _controller_status_writer_binding_resource(job: FlashRLJob) -> dict[str, Any]:
    role_name = job_workload_name(job, "controller-status-writer")
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": _resource_metadata(
            job,
            "controller",
            name=job_workload_name(job, "controller-binding"),
        ),
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": _controller_service_account_name(job),
                "namespace": job_namespace(job),
            }
        ],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": role_name,
        },
    }


def _pod_environment(job: FlashRLJob, workload: str, workload_spec: WorkloadSpec) -> list[dict[str, Any]]:
    namespace = job_namespace(job)
    learner_host = (
        f"{job_workload_name(job, 'learner')}-0."
        f"{job_workload_name(job, 'learner')}.{namespace}.svc.cluster.local"
    )
    env = [
        {"name": "FLASHRL_JOB_NAME", "value": job.name},
        {"name": "FLASHRL_JOB_CONFIG_PATH", "value": "/etc/flashrl/job/job.json"},
        {"name": "FLASHRL_CONTROLLER_URL", "value": f"http://{job_workload_name(job, 'controller')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_ROLLOUT_URL", "value": f"http://{job_workload_name(job, 'rollout')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_REWARD_URL", "value": f"http://{job_workload_name(job, 'reward')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_LEARNER_URL", "value": f"http://{learner_host}"},
        {"name": "FLASHRL_SERVING_URL", "value": f"http://{job_workload_name(job, 'serving')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_JOB_UID", "value": _job_uid(job)},
        {"name": "FLASHRL_COMPONENT_NAME", "value": workload},
        {"name": "FLASHRL_JOB_LOG_ROOT", "value": _job_log_root(job)},
        {"name": "FLASHRL_COMPONENT_LOG_DIR", "value": _component_log_root(job, workload)},
        {
            "name": "FLASHRL_NAMESPACE",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
        },
        {
            "name": "FLASHRL_POD_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
    ]
    env.extend({"name": key, "value": value} for key, value in sorted(workload_spec.env.items()))
    return env


def _health_probe(path: str) -> dict[str, Any]:
    return {
        "httpGet": {"path": path, "port": "http"},
        "periodSeconds": 10,
        "timeoutSeconds": 3,
        "failureThreshold": 3,
    }


def _drain_lifecycle_hook(wait_seconds: int) -> dict[str, Any]:
    script = (
        "import urllib.request\n"
        f"url='http://127.0.0.1:{HTTP_PORT}/v1/lifecycle/drain?wait_seconds={int(wait_seconds)}'\n"
        "try:\n"
        "    urllib.request.urlopen(urllib.request.Request(url, data=b'', method='POST'), timeout=30).read()\n"
        "except Exception:\n"
        "    pass\n"
    )
    return {
        "preStop": {
            "exec": {
                "command": ["python", "-c", script],
            }
        }
    }


def _shared_storage_claim_name(job: FlashRLJob) -> str:
    claim_name = job.spec.sharedStorage.claim.claimName
    if claim_name:
        return str(claim_name)
    return job_workload_name(job, "shared")


def _shared_storage_mount_path(job: FlashRLJob) -> dict[str, Any] | None:
    if not job.spec.sharedStorage.enabled:
        return None
    return {
        "name": "shared-storage",
        "mountPath": job.spec.sharedStorage.mountPath,
    }


def _shared_storage_volume(job: FlashRLJob) -> dict[str, Any] | None:
    if not job.spec.sharedStorage.enabled:
        return None
    return {
        "name": "shared-storage",
        "persistentVolumeClaim": {
            "claimName": _shared_storage_claim_name(job),
        },
    }
def _persistent_volume_claim_resource(job: FlashRLJob) -> dict[str, Any] | None:
    if not job.spec.sharedStorage.enabled or not job.spec.sharedStorage.claim.create:
        return None
    claim = job.spec.sharedStorage.claim
    spec: dict[str, Any] = {
        "accessModes": list(claim.accessModes),
        "resources": {"requests": {"storage": claim.size}},
    }
    if claim.storageClassName is not None:
        spec["storageClassName"] = claim.storageClassName
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": _resource_metadata(job, "storage", name=_shared_storage_claim_name(job)),
        "spec": spec,
    }


def _controller_deployment_resource(job: FlashRLJob) -> dict[str, Any]:
    controller_spec = job.spec.controller
    selector_labels = job_workload_selector_labels(job, "controller")
    pod_labels = {**selector_labels, **controller_spec.labels}
    container: dict[str, Any] = {
        "name": "controller",
        "image": str(job.spec.images.runtime),
        "imagePullPolicy": str(job.spec.images.pullPolicy),
        "command": ["flashrl", "controller"],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}],
        "env": _pod_environment(job, "controller", controller_spec),
        "resources": controller_spec.resources.model_dump(mode="json"),
        "volumeMounts": [{"name": "job-config", "mountPath": "/etc/flashrl/job", "readOnly": True}],
        "readinessProbe": _health_probe("/readyz"),
        "livenessProbe": _health_probe("/healthz"),
    }
    shared_storage_mount = _shared_storage_mount_path(job)
    if shared_storage_mount is not None:
        container["volumeMounts"].append(shared_storage_mount)
    volumes = [
        {
            "name": "job-config",
            "configMap": {"name": job_workload_name(job, "config")},
        }
    ]
    shared_storage_volume = _shared_storage_volume(job)
    if shared_storage_volume is not None:
        volumes.append(shared_storage_volume)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _resource_metadata(
            job,
            "controller",
            name=job_workload_name(job, "controller"),
            labels=controller_spec.labels,
        ),
        "spec": {
            "replicas": desired_job_workload_replicas(job, "controller"),
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": pod_labels, "annotations": dict(controller_spec.annotations)},
                "spec": {
                    "serviceAccountName": _controller_service_account_name(job),
                    "containers": [container],
                    "volumes": volumes,
                    "terminationGracePeriodSeconds": 45,
                },
            },
        },
    }


def _learner_stateful_set_resource(job: FlashRLJob) -> dict[str, Any]:
    learner_spec = job.spec.learner
    selector_labels = job_workload_selector_labels(job, "learner")
    pod_labels = {**selector_labels, **learner_spec.labels}
    container: dict[str, Any] = {
        "name": "learner",
        "image": str(job.spec.images.training),
        "imagePullPolicy": str(job.spec.images.pullPolicy),
        "command": ["flashrl", "learner"],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}],
        "env": _pod_environment(job, "learner", learner_spec),
        "resources": learner_spec.resources.model_dump(mode="json"),
        "volumeMounts": [{"name": "job-config", "mountPath": "/etc/flashrl/job", "readOnly": True}],
        "readinessProbe": _health_probe("/readyz"),
        "livenessProbe": _health_probe("/healthz"),
    }
    shared_storage_mount = _shared_storage_mount_path(job)
    if shared_storage_mount is not None:
        container["volumeMounts"].append(shared_storage_mount)
    volumes = [
        {
            "name": "job-config",
            "configMap": {"name": job_workload_name(job, "config")},
        }
    ]
    shared_storage_volume = _shared_storage_volume(job)
    if shared_storage_volume is not None:
        volumes.append(shared_storage_volume)
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": _resource_metadata(
            job,
            "learner",
            name=job_workload_name(job, "learner"),
            labels=learner_spec.labels,
        ),
        "spec": {
            "serviceName": job_workload_name(job, "learner"),
            "replicas": desired_job_workload_replicas(job, "learner"),
            "selector": {"matchLabels": selector_labels},
            "podManagementPolicy": "Parallel",
            "template": {
                "metadata": {"labels": pod_labels, "annotations": dict(learner_spec.annotations)},
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                    "terminationGracePeriodSeconds": 120,
                },
            },
        },
    }


def _serving_deployment_resource(job: FlashRLJob) -> dict[str, Any]:
    serving_spec = job.spec.serving
    selector_labels = job_workload_selector_labels(job, "serving")
    pod_labels = {**selector_labels, **serving_spec.labels}
    container: dict[str, Any] = {
        "name": "serving",
        "image": str(job.spec.images.serving),
        "imagePullPolicy": str(job.spec.images.pullPolicy),
        "command": ["flashrl", "serving"],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}],
        "env": _pod_environment(job, "serving", serving_spec),
        "resources": serving_spec.resources.model_dump(mode="json"),
        "volumeMounts": [{"name": "job-config", "mountPath": "/etc/flashrl/job", "readOnly": True}],
        "readinessProbe": _health_probe("/readyz"),
        "livenessProbe": _health_probe("/healthz"),
        "lifecycle": _drain_lifecycle_hook(25),
    }
    shared_storage_mount = _shared_storage_mount_path(job)
    if shared_storage_mount is not None:
        container["volumeMounts"].append(shared_storage_mount)
    volumes = [
        {
            "name": "job-config",
            "configMap": {"name": job_workload_name(job, "config")},
        }
    ]
    shared_storage_volume = _shared_storage_volume(job)
    if shared_storage_volume is not None:
        volumes.append(shared_storage_volume)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _resource_metadata(
            job,
            "serving",
            name=job_workload_name(job, "serving"),
            labels=serving_spec.labels,
        ),
        "spec": {
            "replicas": desired_job_workload_replicas(job, "serving"),
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": pod_labels, "annotations": dict(serving_spec.annotations)},
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                    "terminationGracePeriodSeconds": 45,
                },
            },
        },
    }


def _rollout_deployment_resource(job: FlashRLJob) -> dict[str, Any]:
    rollout_spec = job.spec.rollout
    selector_labels = job_workload_selector_labels(job, "rollout")
    pod_labels = {**selector_labels, **rollout_spec.labels}
    container: dict[str, Any] = {
        "name": "rollout",
        "image": str(job.spec.images.runtime),
        "imagePullPolicy": str(job.spec.images.pullPolicy),
        "command": ["flashrl", "rollout"],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}],
        "env": _pod_environment(job, "rollout", rollout_spec),
        "resources": rollout_spec.resources.model_dump(mode="json"),
        "volumeMounts": [{"name": "job-config", "mountPath": "/etc/flashrl/job", "readOnly": True}],
        "readinessProbe": _health_probe("/readyz"),
        "livenessProbe": _health_probe("/healthz"),
        "lifecycle": _drain_lifecycle_hook(25),
    }
    shared_storage_mount = _shared_storage_mount_path(job)
    if shared_storage_mount is not None:
        container["volumeMounts"].append(shared_storage_mount)
    volumes = [
        {
            "name": "job-config",
            "configMap": {"name": job_workload_name(job, "config")},
        }
    ]
    shared_storage_volume = _shared_storage_volume(job)
    if shared_storage_volume is not None:
        volumes.append(shared_storage_volume)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _resource_metadata(
            job,
            "rollout",
            name=job_workload_name(job, "rollout"),
            labels=rollout_spec.labels,
        ),
        "spec": {
            "replicas": desired_job_workload_replicas(job, "rollout"),
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": pod_labels, "annotations": dict(rollout_spec.annotations)},
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                    "terminationGracePeriodSeconds": 45,
                },
            },
        },
    }


def _reward_deployment_resource(job: FlashRLJob) -> dict[str, Any]:
    reward_spec = job.spec.reward
    selector_labels = job_workload_selector_labels(job, "reward")
    pod_labels = {**selector_labels, **reward_spec.labels}
    container: dict[str, Any] = {
        "name": "reward",
        "image": str(job.spec.images.runtime),
        "imagePullPolicy": str(job.spec.images.pullPolicy),
        "command": ["flashrl", "reward"],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}],
        "env": _pod_environment(job, "reward", reward_spec),
        "resources": reward_spec.resources.model_dump(mode="json"),
        "volumeMounts": [{"name": "job-config", "mountPath": "/etc/flashrl/job", "readOnly": True}],
        "readinessProbe": _health_probe("/readyz"),
        "livenessProbe": _health_probe("/healthz"),
        "lifecycle": _drain_lifecycle_hook(25),
    }
    shared_storage_mount = _shared_storage_mount_path(job)
    if shared_storage_mount is not None:
        container["volumeMounts"].append(shared_storage_mount)
    volumes = [
        {
            "name": "job-config",
            "configMap": {"name": job_workload_name(job, "config")},
        }
    ]
    shared_storage_volume = _shared_storage_volume(job)
    if shared_storage_volume is not None:
        volumes.append(shared_storage_volume)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _resource_metadata(
            job,
            "reward",
            name=job_workload_name(job, "reward"),
            labels=reward_spec.labels,
        ),
        "spec": {
            "replicas": desired_job_workload_replicas(job, "reward"),
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": pod_labels, "annotations": dict(reward_spec.annotations)},
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                    "terminationGracePeriodSeconds": 45,
                },
            },
        },
    }


def _service_resource(job: FlashRLJob, workload: str, *, headless: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "selector": job_workload_selector_labels(job, workload),
        "ports": [{"name": "http", "port": 80, "targetPort": "http"}],
    }
    if headless:
        spec["clusterIP"] = "None"
        spec["publishNotReadyAddresses"] = True
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": _resource_metadata(job, workload, name=job_workload_name(job, workload)),
        "spec": spec,
    }


def _pod_disruption_budget_resource(job: FlashRLJob, workload: str, *, max_unavailable: int) -> dict[str, Any]:
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": _resource_metadata(
            job,
            workload,
            name=job_workload_name(job, f"{workload}-pdb"),
        ),
        "spec": {
            "maxUnavailable": int(max_unavailable),
            "selector": {"matchLabels": job_workload_selector_labels(job, workload)},
        },
    }


def _shared_storage_claim_resource(job: FlashRLJob) -> dict[str, Any] | None:
    return _persistent_volume_claim_resource(job)
