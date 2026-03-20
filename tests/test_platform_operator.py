"""Tests for the FlashRL platform operator reconcile loop."""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse

import pytest

from flashrl.platform.crd import FlashRLJob
from flashrl.platform.operator import FINALIZER, FlashRLOperator


pytestmark = pytest.mark.unit


class NotFoundError(Exception):
    """Fake Kubernetes 404 error."""

    def __init__(self, message: str = "not found") -> None:
        super().__init__(message)
        self.status = 404


class FakeCluster:
    """In-memory Kubernetes object store for operator tests."""

    def __init__(self) -> None:
        self.custom_objects: dict[tuple[str, str], dict[str, object]] = {}
        self.resources: dict[str, dict[tuple[str, str], dict[str, object]]] = {
            "ConfigMap": {},
            "Service": {},
            "ServiceAccount": {},
            "PersistentVolumeClaim": {},
            "Deployment": {},
            "StatefulSet": {},
            "Role": {},
            "RoleBinding": {},
            "PodDisruptionBudget": {},
        }
        self.pods: dict[str, list[dict[str, object]]] = {}
        self.crds: dict[str, dict[str, object]] = {}
        self._resource_version = 1

    def _next_resource_version(self) -> str:
        value = str(self._resource_version)
        self._resource_version += 1
        return value

    def put_custom_object(self, namespace: str, payload: dict[str, object]) -> None:
        body = deepcopy(payload)
        metadata = body.setdefault("metadata", {})
        metadata.setdefault("namespace", namespace)
        metadata.setdefault("generation", 1)
        metadata.setdefault("resourceVersion", self._next_resource_version())
        metadata.setdefault("uid", f"uid-{metadata['name']}")
        body.setdefault("status", {})
        self.custom_objects[(namespace, str(metadata["name"]))] = body

    def get_custom_object(self, namespace: str, name: str) -> dict[str, object]:
        key = (namespace, name)
        if key not in self.custom_objects:
            raise NotFoundError()
        return deepcopy(self.custom_objects[key])

    def patch_custom_object(self, namespace: str, name: str, body: dict[str, object]) -> dict[str, object]:
        payload = self.get_custom_object(namespace, name)
        if "metadata" in body:
            payload.setdefault("metadata", {}).update(deepcopy(body["metadata"]))
        if "status" in body:
            payload.setdefault("status", {}).update(deepcopy(body["status"]))
        payload["metadata"]["resourceVersion"] = self._next_resource_version()
        self.custom_objects[(namespace, name)] = payload
        return deepcopy(payload)

    def create_or_replace_resource(
        self,
        kind: str,
        namespace: str,
        body: dict[str, object],
        *,
        replace: bool,
    ) -> dict[str, object]:
        payload = deepcopy(body)
        metadata = payload.setdefault("metadata", {})
        name = str(metadata["name"])
        metadata.setdefault("namespace", namespace)
        key = (namespace, name)
        existing = deepcopy(self.resources[kind].get(key))
        metadata["resourceVersion"] = self._next_resource_version()
        if kind in {"Deployment", "StatefulSet"}:
            if "status" in payload:
                payload["status"] = deepcopy(payload["status"])
            elif existing is not None and "status" in existing:
                payload["status"] = deepcopy(existing["status"])
            else:
                replicas = int(payload.get("spec", {}).get("replicas", 0) or 0)
                payload["status"] = {
                    "readyReplicas": replicas,
                    "availableReplicas": replicas,
                }
        self.resources[kind][key] = payload
        return deepcopy(payload)

    def get_resource(self, kind: str, namespace: str, name: str) -> dict[str, object]:
        key = (namespace, name)
        if key not in self.resources[kind]:
            raise NotFoundError()
        return deepcopy(self.resources[kind][key])

    def delete_resource(self, kind: str, namespace: str, name: str) -> None:
        key = (namespace, name)
        if key not in self.resources[kind]:
            raise NotFoundError()
        del self.resources[kind][key]

    def patch_workload(self, kind: str, namespace: str, name: str, body: dict[str, object]) -> dict[str, object]:
        payload = self.get_resource(kind, namespace, name)
        spec_patch = body.get("spec", {})
        if "replicas" in spec_patch:
            payload.setdefault("spec", {})["replicas"] = spec_patch["replicas"]
        template_patch = spec_patch.get("template")
        if template_patch is not None:
            annotations = (
                template_patch.get("metadata", {}).get("annotations", {})
            )
            template = payload.setdefault("spec", {}).setdefault("template", {})
            template.setdefault("metadata", {}).setdefault("annotations", {}).update(annotations)
        payload["metadata"]["resourceVersion"] = self._next_resource_version()
        self.resources[kind][(namespace, name)] = payload
        return deepcopy(payload)

    def list_pods(self, namespace: str, label_selector: str) -> dict[str, object]:
        expected = {}
        for part in label_selector.split(","):
            key, value = part.split("=", 1)
            expected[key] = value
        items = []
        for pod in self.pods.get(namespace, []):
            labels = pod.get("metadata", {}).get("labels", {})
            if all(labels.get(key) == value for key, value in expected.items()):
                items.append(deepcopy(pod))
        return {"items": items}


class FakeCustomObjectsApi:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def get_namespaced_custom_object(self, *, namespace: str, name: str, **_: object) -> dict[str, object]:
        return self.cluster.get_custom_object(namespace, name)

    def create_namespaced_custom_object(self, *, namespace: str, body: dict[str, object], **_: object) -> dict[str, object]:
        self.cluster.put_custom_object(namespace, body)
        return self.cluster.get_custom_object(namespace, str(body["metadata"]["name"]))

    def replace_namespaced_custom_object(
        self,
        *,
        namespace: str,
        name: str,
        body: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        self.cluster.put_custom_object(namespace, body)
        return self.cluster.get_custom_object(namespace, name)

    def delete_namespaced_custom_object(self, *, namespace: str, name: str, **_: object) -> dict[str, object]:
        self.cluster.custom_objects.pop((namespace, name), None)
        return {"status": "Success"}

    def patch_namespaced_custom_object(self, *, namespace: str, name: str, body: dict[str, object], **_: object) -> dict[str, object]:
        return self.cluster.patch_custom_object(namespace, name, body)

    def patch_namespaced_custom_object_status(
        self,
        *,
        namespace: str,
        name: str,
        body: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        return self.cluster.patch_custom_object(namespace, name, body)

    def list_namespaced_custom_object(self, *, namespace: str, **_: object) -> dict[str, object]:
        items = [
            deepcopy(payload)
            for (item_namespace, _name), payload in self.cluster.custom_objects.items()
            if item_namespace == namespace
        ]
        return {"items": items}

    def list_cluster_custom_object(self, **_: object) -> dict[str, object]:
        return {"items": [deepcopy(payload) for payload in self.cluster.custom_objects.values()]}


class FakeCoreV1Api:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def read_namespaced_config_map(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("ConfigMap", namespace, name)

    def create_namespaced_config_map(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("ConfigMap", namespace, body, replace=False)

    def replace_namespaced_config_map(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("ConfigMap", namespace, body, replace=True)

    def delete_namespaced_config_map(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("ConfigMap", namespace, name)

    def read_namespaced_service(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("Service", namespace, name)

    def create_namespaced_service(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("Service", namespace, body, replace=False)

    def replace_namespaced_service(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("Service", namespace, body, replace=True)

    def delete_namespaced_service(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("Service", namespace, name)

    def read_namespaced_service_account(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("ServiceAccount", namespace, name)

    def create_namespaced_service_account(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("ServiceAccount", namespace, body, replace=False)

    def replace_namespaced_service_account(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("ServiceAccount", namespace, body, replace=True)

    def delete_namespaced_service_account(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("ServiceAccount", namespace, name)

    def list_namespaced_pod(self, *, namespace: str, label_selector: str) -> dict[str, object]:
        return self.cluster.list_pods(namespace, label_selector)

    def read_namespaced_persistent_volume_claim(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("PersistentVolumeClaim", namespace, name)

    def create_namespaced_persistent_volume_claim(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("PersistentVolumeClaim", namespace, body, replace=False)

    def replace_namespaced_persistent_volume_claim(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("PersistentVolumeClaim", namespace, body, replace=True)

    def delete_namespaced_persistent_volume_claim(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("PersistentVolumeClaim", namespace, name)


class FakeAppsV1Api:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def read_namespaced_deployment(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("Deployment", namespace, name)

    def create_namespaced_deployment(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("Deployment", namespace, body, replace=False)

    def replace_namespaced_deployment(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("Deployment", namespace, body, replace=True)

    def delete_namespaced_deployment(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("Deployment", namespace, name)

    def patch_namespaced_deployment(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.patch_workload("Deployment", namespace, name, body)

    def read_namespaced_stateful_set(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("StatefulSet", namespace, name)

    def create_namespaced_stateful_set(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("StatefulSet", namespace, body, replace=False)

    def replace_namespaced_stateful_set(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("StatefulSet", namespace, body, replace=True)

    def delete_namespaced_stateful_set(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("StatefulSet", namespace, name)

    def patch_namespaced_stateful_set(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.patch_workload("StatefulSet", namespace, name, body)


class FakeRbacAuthorizationV1Api:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def read_namespaced_role(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("Role", namespace, name)

    def create_namespaced_role(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("Role", namespace, body, replace=False)

    def replace_namespaced_role(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("Role", namespace, body, replace=True)

    def delete_namespaced_role(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("Role", namespace, name)

    def read_namespaced_role_binding(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("RoleBinding", namespace, name)

    def create_namespaced_role_binding(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("RoleBinding", namespace, body, replace=False)

    def replace_namespaced_role_binding(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("RoleBinding", namespace, body, replace=True)

    def delete_namespaced_role_binding(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("RoleBinding", namespace, name)


class FakePolicyV1Api:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def read_namespaced_pod_disruption_budget(self, *, name: str, namespace: str) -> dict[str, object]:
        return self.cluster.get_resource("PodDisruptionBudget", namespace, name)

    def create_namespaced_pod_disruption_budget(self, *, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("PodDisruptionBudget", namespace, body, replace=False)

    def replace_namespaced_pod_disruption_budget(self, *, name: str, namespace: str, body: dict[str, object]) -> dict[str, object]:
        return self.cluster.create_or_replace_resource("PodDisruptionBudget", namespace, body, replace=True)

    def delete_namespaced_pod_disruption_budget(self, *, name: str, namespace: str) -> None:
        self.cluster.delete_resource("PodDisruptionBudget", namespace, name)


class FakeApiextensionsV1Api:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def read_custom_resource_definition(self, name: str) -> dict[str, object]:
        if name not in self.cluster.crds:
            raise NotFoundError()
        return deepcopy(self.cluster.crds[name])

    def create_custom_resource_definition(self, body: dict[str, object]) -> dict[str, object]:
        self.cluster.crds[str(body["metadata"]["name"])] = deepcopy(body)
        return deepcopy(body)

    def replace_custom_resource_definition(self, name: str, body: dict[str, object]) -> dict[str, object]:
        self.cluster.crds[name] = deepcopy(body)
        return deepcopy(body)


class FakeClientModule:
    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    def CustomObjectsApi(self) -> FakeCustomObjectsApi:
        return FakeCustomObjectsApi(self.cluster)

    def CoreV1Api(self) -> FakeCoreV1Api:
        return FakeCoreV1Api(self.cluster)

    def AppsV1Api(self) -> FakeAppsV1Api:
        return FakeAppsV1Api(self.cluster)

    def RbacAuthorizationV1Api(self) -> FakeRbacAuthorizationV1Api:
        return FakeRbacAuthorizationV1Api(self.cluster)

    def PolicyV1Api(self) -> FakePolicyV1Api:
        return FakePolicyV1Api(self.cluster)

    def ApiextensionsV1Api(self) -> FakeApiextensionsV1Api:
        return FakeApiextensionsV1Api(self.cluster)


def _job_payload() -> dict[str, object]:
    return {
        "apiVersion": "platform.flashrl.dev/v1alpha1",
        "kind": "FlashRLJob",
        "metadata": {"name": "demo-job", "namespace": "default"},
        "spec": {
            "framework": {
                "actor": {"model_name": "fake/model", "backend": "fsdp2", "dp_size": 2},
                "serving": {"model_name": "fake/model", "backend": "huggingface"},
                "trainer": {"batch_size": 4, "max_epochs": 1},
                "grpo": {"group_size": 2, "kl_coefficient": 0.0},
            },
            "dataset": {"type": "hook"},
            "images": {
                "runtime": "ghcr.io/flashrl/flashrl-runtime:latest",
                "serving": "ghcr.io/flashrl/flashrl-serving-vllm:latest",
                "training": "ghcr.io/flashrl/flashrl-training-fsdp:latest",
                "pullPolicy": "IfNotPresent",
            },
            "userCode": {
                "dataset": {"import": "tests.platform_hooks:build_dataset"},
                "rollout": {"import": "tests.platform_hooks:build_rollout"},
                "reward": {"import": "tests.platform_hooks:build_reward"},
            },
            "sharedStorage": {
                "enabled": True,
                "mountPath": "/var/lib/flashrl/shared",
                "claim": {"size": "2Gi"},
            },
            "serving": {
                "replicas": {"min": 2, "max": 6},
                "autoscaling": {"enabled": True, "targetInflightPerReplica": 2},
            },
            "rollout": {
                "replicas": {"min": 2, "max": 4},
            },
            "reward": {
                "replicas": {"min": 1, "max": 2},
            },
            "storage": {
                "checkpoints": {"uriPrefix": "checkpoints"},
                "weights": {"uriPrefix": "weights"},
            },
        },
        "status": {
            "progress": {"currentEpoch": 1, "currentStep": 7, "lastCompletedStep": 6},
        },
    }


def _ready_pod(*, component: str, ip: str) -> dict[str, object]:
    return {
        "metadata": {
            "name": f"{component}-{ip.rsplit('.', 1)[-1]}",
            "labels": {
                "app.kubernetes.io/name": "flashrl",
                "app.kubernetes.io/component": component,
                "flashrl.dev/job": "demo-job",
            },
        },
        "status": {
            "podIP": ip,
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [{"restartCount": 0}],
        },
    }


def _http_status_map(mapping: dict[str, dict[str, object]]):
    def _getter(url: str) -> dict[str, object]:
        ip = urlparse(url).hostname
        if ip is None or ip not in mapping:
            raise RuntimeError(f"missing status for {url}")
        return deepcopy(mapping[ip])

    return _getter


def test_operator_reconcile_scales_serving_and_preserves_controller_progress() -> None:
    """Reconcile should add a finalizer, scale serving, and keep controller-owned progress."""
    cluster = FakeCluster()
    cluster.put_custom_object("default", _job_payload())
    cluster.pods["default"] = [
        _ready_pod(component="serving", ip="10.0.0.11"),
        _ready_pod(component="serving", ip="10.0.0.12"),
    ]
    status_map = {
        "10.0.0.11": {
            "status": {
                "name": "serving",
                "phase": "Ready",
                "healthy": True,
                "ready_replica_count": 2,
                "desired_replica_count": 2,
                "metadata": {
                    "inflightRequests": 2,
                    "queueDepth": 4,
                    "p95LatencySeconds": 0.4,
                    "draining": False,
                    "lastObservedAt": "2026-03-20T00:00:00Z",
                },
                "active_weight_version": {"version_id": 1, "model_source": "weights://1", "origin": "sync"},
            }
        },
        "10.0.0.12": {
            "status": {
                "name": "serving",
                "phase": "Ready",
                "healthy": True,
                "ready_replica_count": 2,
                "desired_replica_count": 2,
                "metadata": {
                    "inflightRequests": 2,
                    "queueDepth": 4,
                    "p95LatencySeconds": 0.5,
                    "draining": False,
                    "lastObservedAt": "2026-03-20T00:00:00Z",
                },
                "active_weight_version": {"version_id": 1, "model_source": "weights://1", "origin": "sync"},
            }
        },
    }

    operator = FlashRLOperator(
        client_module=FakeClientModule(cluster),
        http_getter=_http_status_map(status_map),
    )
    job = FlashRLJob.model_validate(cluster.get_custom_object("default", "demo-job"))
    status = operator.reconcile_payload(job)

    custom_object = cluster.get_custom_object("default", "demo-job")
    serving = cluster.get_resource("Deployment", "default", "demo-job-serving")
    rollout = cluster.get_resource("Deployment", "default", "demo-job-rollout")
    pvc = cluster.get_resource("PersistentVolumeClaim", "default", "demo-job-shared")
    assert FINALIZER in custom_object["metadata"]["finalizers"]
    assert serving["spec"]["replicas"] == 3
    assert rollout["spec"]["template"]["spec"]["containers"][0]["image"] == "ghcr.io/flashrl/flashrl-runtime:latest"
    assert rollout["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] == "IfNotPresent"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "2Gi"
    assert custom_object["status"]["progress"]["currentStep"] == 7
    assert status["components"]["serving"]["desiredReplicas"] == 3
    assert status["components"]["serving"]["lastScaleAt"] is not None


def test_operator_reconcile_restarts_learner_set_after_timeout() -> None:
    """Learner readiness timeout should trigger a full StatefulSet restart."""
    cluster = FakeCluster()
    payload = _job_payload()
    payload["status"] = {
        "components": {
            "learner": {
                "unreadySince": "2020-01-01T00:00:00Z",
                "recoveryAttempts": 0,
            }
        }
    }
    cluster.put_custom_object("default", payload)
    cluster.create_or_replace_resource(
        "StatefulSet",
        "default",
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": "demo-job-learner"},
            "spec": {"replicas": 2},
            "status": {"readyReplicas": 1, "availableReplicas": 1},
        },
        replace=False,
    )

    operator = FlashRLOperator(client_module=FakeClientModule(cluster), http_getter=_http_status_map({}))
    job = FlashRLJob.model_validate(cluster.get_custom_object("default", "demo-job"))
    status = operator.reconcile_payload(job)

    learner = cluster.get_resource("StatefulSet", "default", "demo-job-learner")
    assert learner["spec"]["replicas"] == 0
    assert status["components"]["learner"]["recoveryAttempts"] == 1
    assert status["components"]["learner"]["phase"] == "Recovering"


def test_operator_reconcile_restarts_zero_ready_serving_pool_after_timeout() -> None:
    """Zero-ready scalable pools should get a rollout restart annotation after timeout."""
    cluster = FakeCluster()
    payload = _job_payload()
    payload["status"] = {
        "components": {
            "serving": {
                "unreadySince": "2020-01-01T00:00:00Z",
                "recoveryAttempts": 0,
            }
        }
    }
    cluster.put_custom_object("default", payload)
    cluster.create_or_replace_resource(
        "Deployment",
        "default",
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "demo-job-serving"},
            "spec": {
                "replicas": 2,
                "template": {"metadata": {"annotations": {}}},
            },
            "status": {"readyReplicas": 0, "availableReplicas": 0},
        },
        replace=False,
    )

    operator = FlashRLOperator(client_module=FakeClientModule(cluster), http_getter=_http_status_map({}))
    job = FlashRLJob.model_validate(cluster.get_custom_object("default", "demo-job"))
    status = operator.reconcile_payload(job)

    serving = cluster.get_resource("Deployment", "default", "demo-job-serving")
    annotations = serving["spec"]["template"]["metadata"]["annotations"]
    assert "flashrl.dev/restartedAt" in annotations
    assert status["components"]["serving"]["recoveryAttempts"] == 1
    assert status["components"]["serving"]["phase"] == "Recovering"
