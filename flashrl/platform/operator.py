"""Kubernetes resource rendering and operator scaffolding for FlashRLJob."""

from __future__ import annotations

from typing import Any

from flashrl.platform.crd import FlashRLJob, flashrljob_crd_manifest


def _labels(job: FlashRLJob, component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "flashrl",
        "app.kubernetes.io/component": component,
        "flashrl.dev/job": job.name,
    }


def _workload_name(job: FlashRLJob, suffix: str) -> str:
    return f"{job.name}-{suffix}"


def render_child_resources(job: FlashRLJob) -> list[dict[str, Any]]:
    """Render the child Kubernetes resources for one FlashRLJob."""
    resources: list[dict[str, Any]] = []
    resources.append(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": _workload_name(job, "config"), "labels": _labels(job, "config")},
            "data": {"job.json": job.model_dump_json(indent=2)},
        }
    )
    resources.append(_deployment(job, "controller", replicas=1))
    resources.append(_statefulset(job, "learner", replicas=job.spec.framework.actor.dp_size))
    resources.append(_deployment(job, "serving", replicas=job.spec.servingPool.replicas.min if job.spec.servingPool.replicas else 1))
    resources.append(_deployment(job, "rollout", replicas=job.spec.rollout.replicas.min if job.spec.rollout.replicas else 1))
    resources.append(_deployment(job, "reward", replicas=job.spec.reward.replicas.min if job.spec.reward.replicas else 1))
    resources.extend(_service(job, component) for component in ("controller", "learner", "serving", "rollout", "reward"))
    return resources


def _deployment(job: FlashRLJob, component: str, *, replicas: int) -> dict[str, Any]:
    workload = getattr(job.spec, component if component != "serving" else "servingPool")
    name = _workload_name(job, component)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": _labels(job, component)},
        "spec": {
            "replicas": int(replicas),
            "selector": {"matchLabels": _labels(job, component)},
            "template": {
                "metadata": {"labels": _labels(job, component)},
                "spec": {
                    "containers": [
                        {
                            "name": component,
                            "image": workload.image,
                            "env": [
                                {"name": key, "value": value}
                                for key, value in sorted(workload.env.items())
                            ],
                            "resources": workload.resources.model_dump(),
                        }
                    ]
                },
            },
        },
    }


def _statefulset(job: FlashRLJob, component: str, *, replicas: int) -> dict[str, Any]:
    workload = getattr(job.spec, component)
    name = _workload_name(job, component)
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": name, "labels": _labels(job, component)},
        "spec": {
            "serviceName": name,
            "replicas": int(replicas),
            "selector": {"matchLabels": _labels(job, component)},
            "template": {
                "metadata": {"labels": _labels(job, component)},
                "spec": {
                    "containers": [
                        {
                            "name": component,
                            "image": workload.image,
                            "env": [
                                {"name": key, "value": value}
                                for key, value in sorted(workload.env.items())
                            ],
                            "resources": workload.resources.model_dump(),
                        }
                    ]
                },
            },
        },
    }


def _service(job: FlashRLJob, component: str) -> dict[str, Any]:
    name = _workload_name(job, component)
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": _labels(job, component)},
        "spec": {
            "selector": _labels(job, component),
            "ports": [{"name": "http", "port": 80, "targetPort": 8000}],
        },
    }


class FlashRLOperator:
    """Thin Python operator wrapper around the Kubernetes client."""

    def __init__(self) -> None:
        self._client = None

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

    def ensure_crd(self) -> None:
        """Create or update the FlashRLJob CRD."""
        client = self.load_client()
        api = client.ApiextensionsV1Api()
        manifest = flashrljob_crd_manifest()
        try:
            api.read_custom_resource_definition(manifest["metadata"]["name"])
            api.replace_custom_resource_definition(manifest["metadata"]["name"], manifest)
        except Exception:
            api.create_custom_resource_definition(manifest)

    def render(self, job: FlashRLJob) -> list[dict[str, Any]]:
        """Return the rendered child resources for a validated job."""
        return render_child_resources(job)

    def submit_job(self, job: FlashRLJob, *, namespace: str = "default") -> dict[str, Any]:
        """Create or replace one FlashRLJob custom resource."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        body = job.model_dump(mode="json")
        metadata = body.setdefault("metadata", {})
        metadata.setdefault("namespace", namespace)
        try:
            existing = api.get_namespaced_custom_object(
                group="platform.flashrl.dev",
                version="v1alpha1",
                namespace=namespace,
                plural="flashrljobs",
                name=job.name,
            )
            body["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            return api.replace_namespaced_custom_object(
                group="platform.flashrl.dev",
                version="v1alpha1",
                namespace=namespace,
                plural="flashrljobs",
                name=job.name,
                body=body,
            )
        except Exception:
            return api.create_namespaced_custom_object(
                group="platform.flashrl.dev",
                version="v1alpha1",
                namespace=namespace,
                plural="flashrljobs",
                body=body,
            )

    def get_job(self, name: str, *, namespace: str = "default") -> dict[str, Any]:
        """Fetch one FlashRLJob custom resource."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        return api.get_namespaced_custom_object(
            group="platform.flashrl.dev",
            version="v1alpha1",
            namespace=namespace,
            plural="flashrljobs",
            name=name,
        )

    def delete_job(self, name: str, *, namespace: str = "default") -> dict[str, Any]:
        """Delete one FlashRLJob custom resource."""
        client = self.load_client()
        api = client.CustomObjectsApi()
        return api.delete_namespaced_custom_object(
            group="platform.flashrl.dev",
            version="v1alpha1",
            namespace=namespace,
            plural="flashrljobs",
            name=name,
        )
