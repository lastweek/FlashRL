"""Kubernetes resource rendering and operator control loop for FlashRLJob."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
from math import ceil
from time import sleep
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from flashrl.framework.admin.objects import utc_now_iso
from flashrl.framework.distributed.models import ComponentStatus
from flashrl.platform.crd import (
    AutoscalingSpec,
    ConditionStatus,
    FailurePolicySpec,
    FlashRLJob,
    WorkloadSpec,
    WorkloadStatus,
    flashrljob_crd_manifest,
)


GROUP = "platform.flashrl.dev"
VERSION = "v1alpha1"
PLURAL = "flashrljobs"
FINALIZER = "platform.flashrl.dev/finalizer"
HTTP_PORT = 8000
READINESS_TIMEOUT_SECONDS = 3.0
RESYNC_SECONDS = 60
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

_RESOURCE_APIS: dict[str, tuple[str, str, str, str, str]] = {
    "ConfigMap": (
        "CoreV1Api",
        "read_namespaced_config_map",
        "create_namespaced_config_map",
        "replace_namespaced_config_map",
        "delete_namespaced_config_map",
    ),
    "Service": (
        "CoreV1Api",
        "read_namespaced_service",
        "create_namespaced_service",
        "replace_namespaced_service",
        "delete_namespaced_service",
    ),
    "ServiceAccount": (
        "CoreV1Api",
        "read_namespaced_service_account",
        "create_namespaced_service_account",
        "replace_namespaced_service_account",
        "delete_namespaced_service_account",
    ),
    "PersistentVolumeClaim": (
        "CoreV1Api",
        "read_namespaced_persistent_volume_claim",
        "create_namespaced_persistent_volume_claim",
        "replace_namespaced_persistent_volume_claim",
        "delete_namespaced_persistent_volume_claim",
    ),
    "Deployment": (
        "AppsV1Api",
        "read_namespaced_deployment",
        "create_namespaced_deployment",
        "replace_namespaced_deployment",
        "delete_namespaced_deployment",
    ),
    "StatefulSet": (
        "AppsV1Api",
        "read_namespaced_stateful_set",
        "create_namespaced_stateful_set",
        "replace_namespaced_stateful_set",
        "delete_namespaced_stateful_set",
    ),
    "Role": (
        "RbacAuthorizationV1Api",
        "read_namespaced_role",
        "create_namespaced_role",
        "replace_namespaced_role",
        "delete_namespaced_role",
    ),
    "RoleBinding": (
        "RbacAuthorizationV1Api",
        "read_namespaced_role_binding",
        "create_namespaced_role_binding",
        "replace_namespaced_role_binding",
        "delete_namespaced_role_binding",
    ),
    "PodDisruptionBudget": (
        "PolicyV1Api",
        "read_namespaced_pod_disruption_budget",
        "create_namespaced_pod_disruption_budget",
        "replace_namespaced_pod_disruption_budget",
        "delete_namespaced_pod_disruption_budget",
    ),
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
    """Render the child Kubernetes resources for one FlashRLJob."""
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


def render_operator_resources(
    *,
    namespace: str = "flashrl-system",
    image: str = "ghcr.io/flashrl/flashrl-operator:latest",
) -> list[dict[str, Any]]:
    """Render cluster resources for the FlashRL operator itself."""
    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "flashrl-operator", "namespace": namespace},
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": "flashrl-operator"},
            "rules": [
                {
                    "apiGroups": [GROUP],
                    "resources": [PLURAL, f"{PLURAL}/status"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": [""],
                    "resources": [
                        "configmaps",
                        "services",
                        "pods",
                        "serviceaccounts",
                        "persistentvolumeclaims",
                    ],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments", "statefulsets"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": ["rbac.authorization.k8s.io"],
                    "resources": ["roles", "rolebindings"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": ["policy"],
                    "resources": ["poddisruptionbudgets"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": ["apiextensions.k8s.io"],
                    "resources": ["customresourcedefinitions"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch"],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "flashrl-operator"},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "flashrl-operator",
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "flashrl-operator",
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "flashrl-operator", "namespace": namespace},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app.kubernetes.io/name": "flashrl-operator"}},
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": "flashrl-operator"}},
                    "spec": {
                        "serviceAccountName": "flashrl-operator",
                        "containers": [
                            {
                                "name": "operator",
                                "image": image,
                                "command": ["flashrl", "platform", "operator"],
                            }
                        ],
                    },
                },
            },
        },
    ]


def _to_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_plain(to_dict())
    items = getattr(value, "items", None)
    if callable(items):
        return {key: _to_plain(item) for key, item in items()}
    return value


def _get(payload: Any, *path: str, default: Any = None) -> Any:
    current = _to_plain(payload)
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _parse_timestamp(timestamp: str | None) -> datetime | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds_since(now: datetime, timestamp: str | None) -> float:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return float("inf")
    return max((now - parsed).total_seconds(), 0.0)


def _pod_ready(pod: dict[str, Any]) -> bool:
    if _get(pod, "metadata", "deletionTimestamp") is not None:
        return False
    conditions = _get(pod, "status", "conditions", default=[]) or []
    for condition in conditions:
        if condition.get("type") == "Ready":
            return condition.get("status") == "True"
    return False


def _restart_count(pod: dict[str, Any]) -> int:
    statuses = _get(pod, "status", "containerStatuses", default=[]) or []
    return sum(int(item.get("restartCount", 0) or 0) for item in statuses)


def _default_http_get(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = urllib_request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def _aggregate_load(statuses: list[ComponentStatus]) -> dict[str, Any]:
    eligible = [
        status
        for status in statuses
        if not bool(status.metadata.get("draining", False))
    ]
    if not eligible:
        return {"available": False, "queueDepth": 0, "inflightRequests": 0, "p95LatencySeconds": 0.0}
    try:
        queue_depth = sum(int(status.metadata["queueDepth"]) for status in eligible)
        inflight_requests = sum(int(status.metadata["inflightRequests"]) for status in eligible)
        p95_latency = max(float(status.metadata["p95LatencySeconds"]) for status in eligible)
    except (KeyError, TypeError, ValueError):
        return {"available": False, "queueDepth": 0, "inflightRequests": 0, "p95LatencySeconds": 0.0}
    return {
        "available": True,
        "queueDepth": queue_depth,
        "inflightRequests": inflight_requests,
        "p95LatencySeconds": p95_latency,
    }


def _build_condition(
    *,
    previous: dict[str, ConditionStatus],
    condition_type: str,
    status: Literal["True", "False", "Unknown"],
    reason: str,
    message: str | None,
    now_iso: str,
) -> ConditionStatus:
    prior = previous.get(condition_type)
    last_transition = now_iso
    if prior is not None and prior.status == status and prior.reason == reason and prior.message == message:
        last_transition = prior.lastTransitionTime
    return ConditionStatus(
        type=condition_type,
        status=status,
        reason=reason,
        message=message,
        lastTransitionTime=last_transition,
    )


class FlashRLOperator:
    """Long-running Kubernetes operator for FlashRLJob."""

    def __init__(
        self,
        *,
        client_module: Any | None = None,
        watch_factory: Callable[[], Any] | None = None,
        http_getter: Callable[[str], dict[str, Any]] | None = None,
        sleep_fn: Callable[[float], None] = sleep,
        now_fn: Callable[[], str] = utc_now_iso,
    ) -> None:
        self._client = client_module
        self._watch_factory = watch_factory
        self._http_getter = http_getter
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn

    def load_client(self) -> Any:
        """Load the optional Kubernetes client lazily."""
        if self._client is not None:
            return self._client
        try:
            from kubernetes import client, config
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "FlashRL platform operations require the optional `kubernetes` package."
            ) from exc
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self._client = client
        return client

    def load_watch_factory(self) -> Callable[[], Any]:
        """Load the Kubernetes watch factory lazily."""
        if self._watch_factory is not None:
            return self._watch_factory
        try:
            from kubernetes import watch
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "FlashRL operator watch mode requires the optional `kubernetes` package."
            ) from exc
        self._watch_factory = watch.Watch
        return self._watch_factory

    def ensure_crd(self) -> None:
        """Create or update the FlashRLJob CRD."""
        client = self.load_client()
        api = client.ApiextensionsV1Api()
        manifest = flashrljob_crd_manifest()
        try:
            api.read_custom_resource_definition(manifest["metadata"]["name"])
            api.replace_custom_resource_definition(manifest["metadata"]["name"], manifest)
        except Exception as exc:
            if not self._is_not_found(exc):
                raise
            api.create_custom_resource_definition(manifest)

    def render(self, job: FlashRLJob) -> list[dict[str, Any]]:
        """Return the rendered child resources for a validated job."""
        return render_child_resources(job)

    def render_operator(self, *, namespace: str = "flashrl-system", image: str = "ghcr.io/flashrl/flashrl-operator:latest") -> list[dict[str, Any]]:
        """Return the rendered operator deployment resources."""
        return render_operator_resources(namespace=namespace, image=image)

    def submit_job(self, job: FlashRLJob, *, namespace: str = "default") -> dict[str, Any]:
        """Create or replace one FlashRLJob custom resource."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        body = job.model_dump(mode="json", by_alias=True)
        metadata = body.setdefault("metadata", {})
        metadata.setdefault("namespace", namespace)
        try:
            existing = api.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=job.name,
            )
            body["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            return api.replace_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=job.name,
                body=body,
            )
        except Exception as exc:
            if not self._is_not_found(exc):
                raise
            return api.create_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                body=body,
            )

    def get_job(self, name: str, *, namespace: str = "default") -> dict[str, Any]:
        """Fetch one FlashRLJob custom resource."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        return api.get_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=name,
        )

    def delete_job(self, name: str, *, namespace: str = "default") -> dict[str, Any]:
        """Delete one FlashRLJob custom resource."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        return api.delete_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=name,
        )

    def reconcile_job(self, namespace: str, name: str) -> dict[str, Any]:
        """Fetch one job and reconcile it."""
        payload = self.get_job(name, namespace=namespace)
        job = FlashRLJob.model_validate(payload)
        return self.reconcile_payload(job)

    def reconcile_payload(self, job: FlashRLJob) -> dict[str, Any]:
        """Reconcile one validated FlashRLJob payload."""
        namespace = _namespace_for(job)
        if job.metadata.get("deletionTimestamp") is not None:
            return self.finalize_job(job)

        self._ensure_finalizer(job, namespace=namespace)
        for resource in render_child_resources(job):
            self._upsert_resource(resource, namespace=namespace)

        if job.spec.suspend:
            self._set_workload_replicas(job, "controller", replicas=0, namespace=namespace)

        states = self._gather_component_states(job, namespace=namespace)
        if not job.spec.suspend:
            self._apply_autoscaling(job, states, namespace=namespace)
            self._apply_recovery(job, states, namespace=namespace)

        phase, conditions, last_error = self._summarize_job_status(job, states)
        status_payload = self._build_status_patch(
            job,
            phase=phase,
            conditions=conditions,
            components={name: state["status"] for name, state in states.items()},
            last_error=last_error,
        )
        self._patch_job_status(job, namespace=namespace, status=status_payload)
        return status_payload

    def finalize_job(self, job: FlashRLJob) -> dict[str, Any]:
        """Delete operator-owned child resources and clear the finalizer."""
        namespace = _namespace_for(job)
        for resource in reversed(render_child_resources(job)):
            self._delete_resource(resource["kind"], resource["metadata"]["name"], namespace=namespace)
        self._remove_finalizer(job, namespace=namespace)
        return {"status": "finalized", "name": job.name, "namespace": namespace}

    def run_forever(self, *, namespace: str | None = None, resync_seconds: int = RESYNC_SECONDS) -> None:
        """Run the watch loop indefinitely."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        watch_factory = self.load_watch_factory()

        while True:
            watch = watch_factory()
            list_method = (
                api.list_namespaced_custom_object if namespace is not None else api.list_cluster_custom_object
            )
            stream_kwargs: dict[str, Any] = {
                "group": GROUP,
                "version": VERSION,
                "plural": PLURAL,
                "timeout_seconds": max(int(resync_seconds), 1),
            }
            if namespace is not None:
                stream_kwargs["namespace"] = namespace
            try:
                for event in watch.stream(list_method, **stream_kwargs):
                    payload = _to_plain(event.get("object"))
                    if not isinstance(payload, dict):
                        continue
                    self.reconcile_payload(FlashRLJob.model_validate(payload))
            except KeyboardInterrupt:  # pragma: no cover - interactive operator path
                raise
            except Exception:
                self._sleep_fn(5.0)
            finally:
                stop = getattr(watch, "stop", None)
                if callable(stop):
                    stop()
            self._full_resync(namespace=namespace)

    def _full_resync(self, *, namespace: str | None) -> None:
        client = self.load_client()
        api = client.CustomObjectsApi()
        if namespace is None:
            payload = api.list_cluster_custom_object(group=GROUP, version=VERSION, plural=PLURAL)
        else:
            payload = api.list_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
            )
        items = _get(payload, "items", default=[]) or []
        for item in items:
            self.reconcile_payload(FlashRLJob.model_validate(item))

    def _ensure_finalizer(self, job: FlashRLJob, *, namespace: str) -> None:
        finalizers = list(job.metadata.get("finalizers", []) or [])
        if FINALIZER in finalizers:
            return
        finalizers.append(FINALIZER)
        self._patch_job_metadata(job.name, namespace=namespace, metadata={"finalizers": finalizers})

    def _remove_finalizer(self, job: FlashRLJob, *, namespace: str) -> None:
        finalizers = [item for item in list(job.metadata.get("finalizers", []) or []) if item != FINALIZER]
        self._patch_job_metadata(job.name, namespace=namespace, metadata={"finalizers": finalizers})

    def _patch_job_metadata(self, name: str, *, namespace: str, metadata: dict[str, Any]) -> None:
        client = self.load_client()
        api = client.CustomObjectsApi()
        api.patch_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=name,
            body={"metadata": metadata},
        )

    def _patch_job_status(self, job: FlashRLJob, *, namespace: str, status: dict[str, Any]) -> None:
        client = self.load_client()
        api = client.CustomObjectsApi()
        api.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job.name,
            body={"status": status},
        )

    def _build_status_patch(
        self,
        job: FlashRLJob,
        *,
        phase: str,
        conditions: list[ConditionStatus],
        components: dict[str, WorkloadStatus],
        last_error: str | None,
    ) -> dict[str, Any]:
        status = job.status.model_dump(mode="json")
        status.update(
            {
                "observedGeneration": int(job.metadata.get("generation", 0) or 0),
                "phase": phase,
                "components": {
                    name: component.model_dump(mode="json") for name, component in components.items()
                },
                "conditions": [condition.model_dump(mode="json") for condition in conditions],
                "lastError": last_error,
            }
        )
        return status

    def _summarize_job_status(
        self,
        job: FlashRLJob,
        states: dict[str, dict[str, Any]],
    ) -> tuple[str, list[ConditionStatus], str | None]:
        if job.spec.suspend:
            phase = "Suspended"
        elif any(state["status"].phase == "Failed" for state in states.values()):
            phase = "Failed"
        elif self._serving_convergence_blocked(job, states.get("serving", {})):
            phase = "Degraded"
        elif all(state["status"].phase == "Ready" for state in states.values()):
            phase = "Ready"
        elif any(state["status"].phase in {"Recovering", "Degraded"} for state in states.values()):
            phase = "Degraded"
        else:
            phase = "Pending"

        convergence_message = None
        if self._serving_convergence_blocked(job, states.get("serving", {})):
            desired = _get(job.status.weightVersion.desired, "version_id", default=None)
            convergence_message = f"Serving pool has not converged to desired weight version {desired}."

        component_errors = [
            state["status"].lastError
            for state in states.values()
            if state["status"].lastError
        ]
        last_error = convergence_message or (component_errors[0] if component_errors else None)
        now_iso = self._now_fn()
        previous_conditions = {condition.type: condition for condition in job.status.conditions}
        conditions = [
            _build_condition(
                previous=previous_conditions,
                condition_type="Ready",
                status="True" if phase == "Ready" else "False",
                reason="AllComponentsReady" if phase == "Ready" else "ComponentsNotReady",
                message=None if phase == "Ready" else last_error,
                now_iso=now_iso,
            ),
            _build_condition(
                previous=previous_conditions,
                condition_type="Degraded",
                status="True" if phase == "Degraded" else "False",
                reason="ComponentDegraded" if phase == "Degraded" else "NoDegradation",
                message=last_error if phase == "Degraded" else None,
                now_iso=now_iso,
            ),
            _build_condition(
                previous=previous_conditions,
                condition_type="Suspended",
                status="True" if phase == "Suspended" else "False",
                reason="JobSuspended" if phase == "Suspended" else "JobActive",
                message=None,
                now_iso=now_iso,
            ),
            _build_condition(
                previous=previous_conditions,
                condition_type="Failed",
                status="True" if phase == "Failed" else "False",
                reason="RecoveryExhausted" if phase == "Failed" else "NotFailed",
                message=last_error if phase == "Failed" else None,
                now_iso=now_iso,
            ),
        ]
        if convergence_message is not None:
            conditions.append(
                _build_condition(
                    previous=previous_conditions,
                    condition_type="ServingConvergenceBlocked",
                    status="True",
                    reason="ServingConvergenceBlocked",
                    message=convergence_message,
                    now_iso=now_iso,
                )
            )
        return phase, conditions, last_error

    def _serving_convergence_blocked(self, job: FlashRLJob, state: dict[str, Any]) -> bool:
        desired_version = _get(job.status.weightVersion.desired, "version_id", default=None)
        if desired_version is None:
            return False
        pod_statuses: list[ComponentStatus] = state.get("pod_statuses", [])
        active_versions = {
            status.active_weight_version.version_id
            for status in pod_statuses
            if status.active_weight_version is not None and not bool(status.metadata.get("draining", False))
        }
        if not active_versions:
            return False
        return len(active_versions) != 1 or desired_version not in active_versions

    def _gather_component_states(self, job: FlashRLJob, *, namespace: str) -> dict[str, dict[str, Any]]:
        now_iso = self._now_fn()
        states: dict[str, dict[str, Any]] = {}
        for component in COMPONENTS:
            previous = job.status.components.get(component, WorkloadStatus())
            workload = self._read_workload(job, component, namespace=namespace)
            desired = int(_get(workload, "spec", "replicas", default=_initial_replicas(job, component)) or 0)
            ready = int(_get(workload, "status", "readyReplicas", default=0) or 0)
            available = int(_get(workload, "status", "availableReplicas", default=ready) or 0)
            pods = self._list_component_pods(job, component, namespace=namespace)
            restart_count = sum(_restart_count(pod) for pod in pods)
            pod_statuses, status_error = self._collect_pod_statuses(pods)
            active_weight_version = None
            if component == "serving":
                versions = [
                    status.active_weight_version.model_dump(mode="json")
                    for status in pod_statuses
                    if status.active_weight_version is not None
                ]
                if versions and all(version == versions[0] for version in versions):
                    active_weight_version = dict(versions[0])

            phase = self._phase_for_component(
                component=component,
                desired=desired,
                ready=ready,
                available=available,
                suspended=job.spec.suspend and component == "controller",
                recovering=bool(previous.lastRecoveryAt and desired == 0 and component == "learner"),
                failed=previous.phase == "Failed",
            )
            unready_since = previous.unreadySince
            if desired > 0 and ready < desired:
                unready_since = previous.unreadySince or now_iso
            elif ready >= desired:
                unready_since = None

            status = WorkloadStatus(
                phase=phase,
                readyReplicas=ready,
                availableReplicas=available,
                desiredReplicas=desired,
                worldSize=int(job.spec.framework.actor.dp_size) if component == "learner" else None,
                activeWeightVersion=active_weight_version,
                restartCount=restart_count,
                recoveryAttempts=int(previous.recoveryAttempts),
                lastScaleAt=previous.lastScaleAt,
                lastObservedAt=now_iso,
                lastError=status_error or previous.lastError,
                lowLoadSince=previous.lowLoadSince,
                unreadySince=unready_since,
                lastRecoveryAt=previous.lastRecoveryAt,
            )
            states[component] = {
                "workload": workload,
                "pods": pods,
                "pod_statuses": pod_statuses,
                "status": status,
                "load": _aggregate_load(pod_statuses),
            }
        return states

    def _phase_for_component(
        self,
        *,
        component: str,
        desired: int,
        ready: int,
        available: int,
        suspended: bool,
        recovering: bool,
        failed: bool,
    ) -> str:
        if failed:
            return "Failed"
        if suspended:
            return "Suspended"
        if recovering:
            return "Recovering"
        if desired == 0:
            return "ScaledDown"
        if ready >= desired and available >= desired:
            return "Ready"
        if ready > 0:
            return "Degraded"
        return "Pending"

    def _collect_pod_statuses(self, pods: list[dict[str, Any]]) -> tuple[list[ComponentStatus], str | None]:
        statuses: list[ComponentStatus] = []
        errors: list[str] = []
        for pod in pods:
            if not _pod_ready(pod):
                continue
            pod_ip = _get(pod, "status", "podIP", default=None)
            if not pod_ip:
                continue
            try:
                statuses.append(self._fetch_pod_status(str(pod_ip)))
            except Exception as exc:
                errors.append(str(exc))
        return statuses, (errors[0] if errors else None)

    def _fetch_pod_status(self, pod_ip: str) -> ComponentStatus:
        getter = self._http_getter
        if getter is None:
            getter = lambda url: _default_http_get(url, timeout_seconds=READINESS_TIMEOUT_SECONDS)
        payload = getter(f"http://{pod_ip}:{HTTP_PORT}/v1/status")
        return ComponentStatus.model_validate(_get(payload, "status", default=payload))

    def _read_workload(self, job: FlashRLJob, component: str, *, namespace: str) -> dict[str, Any]:
        kind = _component_kind(component)
        name = _workload_name(job, component)
        try:
            return _to_plain(self._read_resource(kind, name, namespace=namespace))
        except Exception as exc:
            if self._is_not_found(exc):
                return {}
            raise

    def _list_component_pods(self, job: FlashRLJob, component: str, *, namespace: str) -> list[dict[str, Any]]:
        client = self.load_client()
        api = client.CoreV1Api()
        selector = ",".join(f"{key}={value}" for key, value in sorted(_selector_labels(job, component).items()))
        payload = _to_plain(api.list_namespaced_pod(namespace=namespace, label_selector=selector))
        return list(_get(payload, "items", default=[]) or [])

    def _apply_autoscaling(self, job: FlashRLJob, states: dict[str, dict[str, Any]], *, namespace: str) -> None:
        now_iso = self._now_fn()
        now = _parse_timestamp(now_iso) or datetime.now(timezone.utc)
        for component in ELASTIC_COMPONENTS:
            workload = _component_spec(job, component)
            policy = workload.autoscaling or AutoscalingSpec()
            if not policy.enabled or workload.replicas is None:
                continue
            status = states[component]["status"]
            load = states[component]["load"]
            current = int(status.desiredReplicas or workload.replicas.min)
            desired, low_load_since = self._desired_scale_target(
                policy=policy,
                current_replicas=current,
                min_replicas=int(workload.replicas.min),
                max_replicas=int(workload.replicas.max),
                load=load,
                previous=status,
                now=now,
                now_iso=now_iso,
            )
            status.lowLoadSince = low_load_since
            if desired != current:
                self._set_workload_replicas(job, component, replicas=desired, namespace=namespace)
                status.desiredReplicas = desired
                status.lastScaleAt = now_iso

    def _desired_scale_target(
        self,
        *,
        policy: AutoscalingSpec,
        current_replicas: int,
        min_replicas: int,
        max_replicas: int,
        load: dict[str, Any],
        previous: WorkloadStatus,
        now: datetime,
        now_iso: str,
    ) -> tuple[int, str | None]:
        if not load.get("available", False):
            return min_replicas, None

        queue_depth = int(load["queueDepth"])
        inflight_requests = int(load["inflightRequests"])
        p95_latency = float(load["p95LatencySeconds"])
        raw_target = ceil((queue_depth + inflight_requests) / max(policy.targetInflightPerReplica, 1))
        desired = max(raw_target, min_replicas)
        if (
            policy.targetP95LatencySeconds is not None
            and queue_depth > 0
            and p95_latency > float(policy.targetP95LatencySeconds)
        ):
            desired += int(policy.scaleUpStep)
        desired = min(max(desired, min_replicas), max_replicas)

        if desired > current_replicas:
            if _seconds_since(now, previous.lastScaleAt) < float(policy.scaleUpCooldownSeconds):
                return current_replicas, None
            return min(current_replicas + int(policy.scaleUpStep), desired), None

        if desired < current_replicas:
            if queue_depth > 0 or inflight_requests > 0:
                return current_replicas, None
            low_load_since = previous.lowLoadSince or now_iso
            if _seconds_since(now, low_load_since) < float(policy.scaleDownStabilizationSeconds):
                return current_replicas, low_load_since
            if _seconds_since(now, previous.lastScaleAt) < float(policy.scaleDownCooldownSeconds):
                return current_replicas, low_load_since
            return max(current_replicas - int(policy.scaleDownStep), desired), low_load_since

        if queue_depth == 0 and inflight_requests == 0:
            return current_replicas, previous.lowLoadSince or now_iso
        return current_replicas, None

    def _apply_recovery(self, job: FlashRLJob, states: dict[str, dict[str, Any]], *, namespace: str) -> None:
        now_iso = self._now_fn()
        now = _parse_timestamp(now_iso) or datetime.now(timezone.utc)
        for component, state in states.items():
            workload = _component_spec(job, component)
            policy = workload.failurePolicy or FailurePolicySpec()
            status = state["status"]
            if component == "controller":
                if status.readyReplicas < 1 and status.desiredReplicas > 0:
                    status.phase = "Degraded"
                    status.lastError = "Controller deployment has no ready replicas."
                else:
                    status.lastError = None
                continue

            if component == "learner":
                self._recover_learner(
                    job,
                    status=status,
                    policy=policy,
                    namespace=namespace,
                    now=now,
                    now_iso=now_iso,
                )
                continue

            if component in ELASTIC_COMPONENTS:
                self._recover_elastic_pool(
                    job,
                    component=component,
                    status=status,
                    policy=policy,
                    namespace=namespace,
                    now=now,
                    now_iso=now_iso,
                )

    def _recover_learner(
        self,
        job: FlashRLJob,
        *,
        status: WorkloadStatus,
        policy: FailurePolicySpec,
        namespace: str,
        now: datetime,
        now_iso: str,
    ) -> None:
        world_size = int(job.spec.framework.actor.dp_size)
        if status.desiredReplicas == 0 and status.lastRecoveryAt is not None:
            if status.readyReplicas == 0 and _seconds_since(now, status.lastRecoveryAt) >= float(policy.backoffSeconds):
                self._set_workload_replicas(job, "learner", replicas=world_size, namespace=namespace)
                status.desiredReplicas = world_size
                status.phase = "Recovering"
                status.lastRecoveryAt = now_iso
            return

        if status.desiredReplicas <= 0 or status.readyReplicas >= status.desiredReplicas:
            status.unreadySince = None
            status.lastError = None
            return
        if _seconds_since(now, status.unreadySince) < float(policy.readinessTimeoutSeconds):
            status.phase = "Degraded"
            status.lastError = "Learner set is below desired readiness."
            return
        if status.recoveryAttempts >= int(policy.maxRecoveryAttempts):
            status.phase = "Failed"
            status.lastError = "Learner recovery attempts exhausted."
            return
        if _seconds_since(now, status.lastRecoveryAt) < float(policy.backoffSeconds):
            status.phase = "Recovering"
            return
        self._set_workload_replicas(job, "learner", replicas=0, namespace=namespace)
        status.desiredReplicas = 0
        status.recoveryAttempts += 1
        status.lastRecoveryAt = now_iso
        status.phase = "Recovering"
        status.lastError = "Restarting learner StatefulSet after readiness timeout."

    def _recover_elastic_pool(
        self,
        job: FlashRLJob,
        *,
        component: str,
        status: WorkloadStatus,
        policy: FailurePolicySpec,
        namespace: str,
        now: datetime,
        now_iso: str,
    ) -> None:
        if status.desiredReplicas <= 0 or status.readyReplicas > 0:
            status.unreadySince = None
            if status.phase != "Failed":
                status.lastError = None
            return
        if _seconds_since(now, status.unreadySince) < float(policy.readinessTimeoutSeconds):
            status.phase = "Degraded"
            status.lastError = f"{component} pool has no ready replicas."
            return
        if status.recoveryAttempts >= int(policy.maxRecoveryAttempts):
            status.phase = "Failed"
            status.lastError = f"{component} recovery attempts exhausted."
            return
        if _seconds_since(now, status.lastRecoveryAt) < float(policy.backoffSeconds):
            status.phase = "Recovering"
            return
        self._restart_workload(job, component, namespace=namespace, now_iso=now_iso)
        status.recoveryAttempts += 1
        status.lastRecoveryAt = now_iso
        status.phase = "Recovering"
        status.lastError = f"Restarting {component} Deployment after readiness timeout."

    def _resource_api(self, kind: str) -> tuple[Any, Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        client = self.load_client()
        api_name, read_name, create_name, replace_name, delete_name = _RESOURCE_APIS[kind]
        api = getattr(client, api_name)()
        return (
            api,
            getattr(api, read_name),
            getattr(api, create_name),
            getattr(api, replace_name),
            getattr(api, delete_name),
        )

    def _read_resource(self, kind: str, name: str, *, namespace: str) -> Any:
        _, read_method, _, _, _ = self._resource_api(kind)
        return read_method(name=name, namespace=namespace)

    def _upsert_resource(self, resource: dict[str, Any], *, namespace: str) -> dict[str, Any]:
        kind = resource["kind"]
        name = resource["metadata"]["name"]
        _, read_method, create_method, replace_method, _ = self._resource_api(kind)
        body = json.loads(json.dumps(resource))
        try:
            existing = _to_plain(read_method(name=name, namespace=namespace))
        except Exception as exc:
            if not self._is_not_found(exc):
                raise
            return _to_plain(create_method(namespace=namespace, body=body))

        if kind in {"Deployment", "StatefulSet"}:
            existing_replicas = _get(existing, "spec", "replicas", default=None)
            if existing_replicas is not None:
                body.setdefault("spec", {})["replicas"] = existing_replicas
        resource_version = _get(existing, "metadata", "resourceVersion", default=None)
        if resource_version is not None:
            body.setdefault("metadata", {})["resourceVersion"] = resource_version
        return _to_plain(replace_method(name=name, namespace=namespace, body=body))

    def _delete_resource(self, kind: str, name: str, *, namespace: str) -> None:
        _, _, _, _, delete_method = self._resource_api(kind)
        try:
            delete_method(name=name, namespace=namespace)
        except Exception as exc:
            if not self._is_not_found(exc):
                raise

    def _set_workload_replicas(self, job: FlashRLJob, component: str, *, replicas: int, namespace: str) -> None:
        client = self.load_client()
        body = {"spec": {"replicas": int(replicas)}}
        name = _workload_name(job, component)
        if _component_kind(component) == "StatefulSet":
            client.AppsV1Api().patch_namespaced_stateful_set(name=name, namespace=namespace, body=body)
            return
        client.AppsV1Api().patch_namespaced_deployment(name=name, namespace=namespace, body=body)

    def _restart_workload(self, job: FlashRLJob, component: str, *, namespace: str, now_iso: str) -> None:
        client = self.load_client()
        name = _workload_name(job, component)
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "flashrl.dev/restartedAt": now_iso,
                        }
                    }
                }
            }
        }
        client.AppsV1Api().patch_namespaced_deployment(name=name, namespace=namespace, body=patch)

    def _is_not_found(self, exc: Exception) -> bool:
        status = getattr(exc, "status", None)
        if status == 404:
            return True
        if isinstance(exc, KeyError):
            return True
        message = str(exc).lower()
        return "not found" in message or "404" in message
