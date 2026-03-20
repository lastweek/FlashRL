"""Pure Kubernetes manifest rendering for one FlashRLJob."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from flashrl.platform.k8s.job import GROUP, PLURAL, FlashRLJob, WorkloadSpec


HTTP_PORT = 8000
COMPONENTS = ("controller", "learner", "serving", "rollout", "reward")
ELASTIC_COMPONENTS = ("serving", "rollout", "reward")

_COMPONENT_LAYOUT: dict[str, dict[str, Any]] = {
    "controller": {"spec_attr": "controller", "workload_kind": "Deployment", "drainable": False},
    "learner": {
        "spec_attr": "learner",
        "workload_kind": "StatefulSet",
        "drainable": False,
        "headless_service": True,
    },
    "serving": {"spec_attr": "serving", "workload_kind": "Deployment", "drainable": True},
    "rollout": {"spec_attr": "rollout", "workload_kind": "Deployment", "drainable": True},
    "reward": {"spec_attr": "reward", "workload_kind": "Deployment", "drainable": True},
}


def _labels(job: FlashRLJob, component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "flashrl",
        "app.kubernetes.io/component": component,
        "flashrl.dev/job": job.name,
    }


def _selector_labels(job: FlashRLJob, component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "flashrl",
        "app.kubernetes.io/component": component,
        "flashrl.dev/job": job.name,
    }


def _workload_name(job: FlashRLJob, suffix: str) -> str:
    return f"{job.name}-{suffix}"


def _namespace_for(job: FlashRLJob, fallback: str = "default") -> str:
    return str(job.metadata.get("namespace") or fallback)


def _component_spec(job: FlashRLJob, component: str) -> WorkloadSpec:
    return getattr(job.spec, _COMPONENT_LAYOUT[component]["spec_attr"])


def _component_image(job: FlashRLJob, component: str) -> str:
    if component in {"controller", "rollout", "reward"}:
        return str(job.spec.images.runtime)
    if component == "serving":
        return str(job.spec.images.serving)
    if component == "learner":
        return str(job.spec.images.training)
    raise KeyError(component)


def _component_kind(component: str) -> str:
    return str(_COMPONENT_LAYOUT[component]["workload_kind"])


def _component_drainable(component: str) -> bool:
    return bool(_COMPONENT_LAYOUT[component].get("drainable", False))


def _initial_replicas(job: FlashRLJob, component: str) -> int:
    if component == "controller":
        return 0 if job.spec.suspend else 1
    if component == "learner":
        return int(job.spec.framework.actor.dp_size)
    workload = _component_spec(job, component)
    if workload.replicas is None:
        return 1
    return int(workload.replicas.min)


def _owner_references(job: FlashRLJob) -> list[dict[str, Any]]:
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


def _merge_metadata(
    job: FlashRLJob,
    component: str,
    *,
    name: str,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": name,
        "labels": {**_labels(job, component), **(labels or {})},
    }
    if annotations:
        metadata["annotations"] = dict(annotations)
    owner_references = _owner_references(job)
    if owner_references:
        metadata["ownerReferences"] = owner_references
    return metadata


def _job_for_runtime(job: FlashRLJob) -> FlashRLJob:
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


def _render_config_map(job: FlashRLJob) -> dict[str, Any]:
    runtime_job = _job_for_runtime(job)
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": _merge_metadata(job, "config", name=_workload_name(job, "config")),
        "data": {"job.json": runtime_job.model_dump_json(indent=2, by_alias=True)},
    }


def _controller_service_account_name(job: FlashRLJob) -> str:
    return _workload_name(job, "controller")


def _render_controller_rbac(job: FlashRLJob) -> list[dict[str, Any]]:
    namespace = _namespace_for(job)
    service_account_name = _controller_service_account_name(job)
    role_name = _workload_name(job, "controller-status-writer")
    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": _merge_metadata(job, "controller", name=service_account_name),
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": _merge_metadata(job, "controller", name=role_name),
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
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": _merge_metadata(job, "controller", name=_workload_name(job, "controller-binding")),
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": service_account_name,
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": role_name,
            },
        },
    ]


def _common_env(job: FlashRLJob, component: str, workload: WorkloadSpec) -> list[dict[str, Any]]:
    namespace = _namespace_for(job)
    learner_host = (
        f"{_workload_name(job, 'learner')}-0."
        f"{_workload_name(job, 'learner')}.{namespace}.svc.cluster.local"
    )
    env = [
        {"name": "FLASHRL_JOB_NAME", "value": job.name},
        {"name": "FLASHRL_COMPONENT", "value": component},
        {"name": "FLASHRL_JOB_CONFIG_PATH", "value": "/etc/flashrl/job/job.json"},
        {"name": "FLASHRL_CONTROLLER_URL", "value": f"http://{_workload_name(job, 'controller')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_ROLLOUT_URL", "value": f"http://{_workload_name(job, 'rollout')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_REWARD_URL", "value": f"http://{_workload_name(job, 'reward')}.{namespace}.svc.cluster.local"},
        {"name": "FLASHRL_LEARNER_URL", "value": f"http://{learner_host}"},
        {"name": "FLASHRL_SERVING_URL", "value": f"http://{_workload_name(job, 'serving')}.{namespace}.svc.cluster.local"},
        {
            "name": "FLASHRL_NAMESPACE",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
        },
    ]
    env.extend(
        {"name": key, "value": value}
        for key, value in sorted(workload.env.items())
    )
    return env


def _probe(path: str) -> dict[str, Any]:
    return {
        "httpGet": {"path": path, "port": "http"},
        "periodSeconds": 10,
        "timeoutSeconds": 3,
        "failureThreshold": 3,
    }


def _drain_prestop_hook(wait_seconds: int) -> dict[str, Any]:
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
    return _workload_name(job, "shared")


def _shared_storage_volume_name() -> str:
    return "shared-storage"


def _shared_storage_mount(job: FlashRLJob) -> dict[str, Any] | None:
    if not job.spec.sharedStorage.enabled:
        return None
    return {
        "name": _shared_storage_volume_name(),
        "mountPath": job.spec.sharedStorage.mountPath,
    }


def _pod_template(job: FlashRLJob, component: str, workload: WorkloadSpec) -> dict[str, Any]:
    selector_labels = _selector_labels(job, component)
    pod_labels = {**selector_labels, **workload.labels}
    annotations = dict(workload.annotations)
    container: dict[str, Any] = {
        "name": component,
        "image": _component_image(job, component),
        "imagePullPolicy": str(job.spec.images.pullPolicy),
        "command": ["flashrl", "component", "run", "serving-vllm" if component == "serving" else component],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}],
        "env": _common_env(job, component, workload),
        "resources": workload.resources.model_dump(mode="json"),
        "volumeMounts": [{"name": "job-config", "mountPath": "/etc/flashrl/job", "readOnly": True}],
        "readinessProbe": _probe("/readyz"),
        "livenessProbe": _probe("/healthz"),
    }
    pod_spec: dict[str, Any] = {
        "containers": [container],
        "volumes": [
            {
                "name": "job-config",
                "configMap": {"name": _workload_name(job, "config")},
            }
        ],
        "terminationGracePeriodSeconds": 120 if component == "learner" else 45,
    }
    shared_mount = _shared_storage_mount(job)
    if shared_mount is not None:
        container["volumeMounts"].append(shared_mount)
        pod_spec["volumes"].append(
            {
                "name": _shared_storage_volume_name(),
                "persistentVolumeClaim": {
                    "claimName": _shared_storage_claim_name(job),
                },
            }
        )
    if component == "controller":
        pod_spec["serviceAccountName"] = _controller_service_account_name(job)
    if _component_drainable(component):
        container["lifecycle"] = _drain_prestop_hook(25)
    return {
        "metadata": {"labels": pod_labels, "annotations": annotations},
        "spec": pod_spec,
    }


def _deployment(job: FlashRLJob, component: str, *, replicas: int) -> dict[str, Any]:
    workload = _component_spec(job, component)
    selector_labels = _selector_labels(job, component)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _merge_metadata(job, component, name=_workload_name(job, component), labels=workload.labels),
        "spec": {
            "replicas": int(replicas),
            "selector": {"matchLabels": selector_labels},
            "template": _pod_template(job, component, workload),
        },
    }


def _statefulset(job: FlashRLJob, component: str, *, replicas: int) -> dict[str, Any]:
    workload = _component_spec(job, component)
    selector_labels = _selector_labels(job, component)
    name = _workload_name(job, component)
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": _merge_metadata(job, component, name=name, labels=workload.labels),
        "spec": {
            "serviceName": name,
            "replicas": int(replicas),
            "selector": {"matchLabels": selector_labels},
            "podManagementPolicy": "Parallel",
            "template": _pod_template(job, component, workload),
        },
    }


def _service(job: FlashRLJob, component: str) -> dict[str, Any]:
    name = _workload_name(job, component)
    headless = bool(_COMPONENT_LAYOUT[component].get("headless_service"))
    spec: dict[str, Any] = {
        "selector": _selector_labels(job, component),
        "ports": [{"name": "http", "port": 80, "targetPort": "http"}],
    }
    if headless:
        spec["clusterIP"] = "None"
        spec["publishNotReadyAddresses"] = True
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": _merge_metadata(job, component, name=name),
        "spec": spec,
    }


def _pod_disruption_budget(job: FlashRLJob, component: str, *, max_unavailable: int) -> dict[str, Any]:
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": _merge_metadata(job, component, name=_workload_name(job, f"{component}-pdb")),
        "spec": {
            "maxUnavailable": int(max_unavailable),
            "selector": {"matchLabels": _selector_labels(job, component)},
        },
    }


def _persistent_volume_claim(job: FlashRLJob) -> dict[str, Any] | None:
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
        "metadata": _merge_metadata(job, "storage", name=_shared_storage_claim_name(job)),
        "spec": spec,
    }


def render_child_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render the operator-owned child Kubernetes resources for one job."""
    resources: list[dict[str, Any]] = [
        _render_config_map(job),
        *_render_controller_rbac(job),
        _deployment(job, "controller", replicas=_initial_replicas(job, "controller")),
        _statefulset(job, "learner", replicas=_initial_replicas(job, "learner")),
        _deployment(job, "serving", replicas=_initial_replicas(job, "serving")),
        _deployment(job, "rollout", replicas=_initial_replicas(job, "rollout")),
        _deployment(job, "reward", replicas=_initial_replicas(job, "reward")),
        _service(job, "controller"),
        _service(job, "learner"),
        _service(job, "serving"),
        _service(job, "rollout"),
        _service(job, "reward"),
        _pod_disruption_budget(job, "learner", max_unavailable=0),
    ]
    pvc = _persistent_volume_claim(job)
    if pvc is not None:
        resources.insert(1, pvc)
    for component in ELASTIC_COMPONENTS:
        workload = _component_spec(job, component)
        if workload.replicas is not None and workload.replicas.min > 1:
            resources.append(_pod_disruption_budget(job, component, max_unavailable=1))
    return resources
