"""Reconcile orchestration for the FlashRL platform operator."""

from __future__ import annotations

import json
from time import sleep
from typing import TYPE_CHECKING, Any, Callable

from flashrl.platform.k8s.job import GROUP, PLURAL, VERSION, FlashRLJob
from flashrl.platform.k8s.job_resources import job_namespace, render_job_resources
from flashrl.platform.k8s.operator.kube import get_path, is_not_found, to_plain

if TYPE_CHECKING:
    from flashrl.platform.k8s.operator.recovery import RecoveryManager
    from flashrl.platform.k8s.operator.scaling import Autoscaler
    from flashrl.platform.k8s.operator.status import StatusCollector


RESYNC_SECONDS = 60


# This module owns the reconcile ordering only. Policy decisions stay in
# scaling.py and recovery.py so the control flow is easy to follow here.
class JobReconciler:
    """Own the reconcile, finalize, and watch sequence for FlashRLJobs."""

    def __init__(
        self,
        client_loader: Callable[[], Any],
        watch_factory_loader: Callable[[], Callable[[], Any]],
        status_collector: StatusCollector,
        autoscaler: Autoscaler,
        recovery_manager: RecoveryManager,
        *,
        finalizer: str,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        self._load_client = client_loader
        self._load_watch_factory = watch_factory_loader
        self._status_collector = status_collector
        self._autoscaler = autoscaler
        self._recovery_manager = recovery_manager
        self._finalizer = finalizer
        self._sleep_fn = sleep_fn

    def reconcile_job(self, namespace: str, name: str) -> dict[str, Any]:
        """Fetch one CR instance from Kubernetes, validate it, then reconcile."""
        payload = to_plain(
            self._load_client().CustomObjectsApi().get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=name,
            )
        )
        return self.reconcile(FlashRLJob.model_validate(payload))

    def reconcile(self, job: FlashRLJob) -> dict[str, Any]:
        """Reconcile one validated FlashRLJob payload."""
        namespace = job_namespace(job)
        # Deletion is the only early-exit path because Kubernetes expects the
        # operator to release owned resources before removing its finalizer.
        if job.metadata.get("deletionTimestamp") is not None:
            return self.finalize(job)

        # The steady-state reconcile sequence is:
        # 1. ensure the operator finalizer exists
        # 2. apply the desired child resources
        # 3. honor job suspension
        # 4. collect observations from Kubernetes and component /v1/status
        # 5. autoscale elastic pools
        # 6. recover unhealthy workloads
        # 7. summarize top-level job health
        # 8. patch CRD status
        self._ensure_finalizer(job, namespace=namespace)
        for resource in render_job_resources(job):
            self._apply_resource(resource, namespace=namespace)

        if job.spec.suspend:
            self._load_client().AppsV1Api().patch_namespaced_deployment(
                name=f"{job.name}-controller",
                namespace=namespace,
                body={"spec": {"replicas": 0}},
            )

        observations = self._status_collector.collect(job, namespace=namespace)
        if not job.spec.suspend:
            self._autoscaler.apply(job, observations, namespace=namespace)
            self._recovery_manager.apply(job, observations, namespace=namespace)

        summary = self._status_collector.summarize(job, observations)
        status_payload = self._status_collector.build_status_patch(
            job,
            summary=summary,
            observations=observations,
        )
        self._patch_job_status(job, namespace=namespace, status=status_payload)
        return status_payload

    def finalize(self, job: FlashRLJob) -> dict[str, Any]:
        """Delete operator-owned children in reverse order, then clear the finalizer."""
        namespace = job_namespace(job)
        # Finalization tears down the resources the operator created for this
        # job, then releases the finalizer so Kubernetes can remove the CR.
        for resource in reversed(render_job_resources(job)):
            self._delete_resource(resource["kind"], resource["metadata"]["name"], namespace=namespace)
        self._remove_finalizer(job, namespace=namespace)
        return {"status": "finalized", "name": job.name, "namespace": namespace}

    def watch(self, *, namespace: str | None = None, resync_seconds: int = RESYNC_SECONDS) -> None:
        """Run the operator watch loop indefinitely."""
        client = self._load_client()
        api = client.CustomObjectsApi()
        watch_factory = self._load_watch_factory()

        while True:
            # Watch events are the fast path. A full resync after each watch
            # window is the safety net for missed events or transient errors.
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
                    payload = to_plain(event.get("object"))
                    if not isinstance(payload, dict):
                        continue
                    self.reconcile(FlashRLJob.model_validate(payload))
            except KeyboardInterrupt:  # pragma: no cover - interactive operator path
                raise
            except Exception:
                # Operator failures should degrade to retry behavior instead of
                # crashing the process and relying on Deployment restarts.
                self._sleep_fn(5.0)
            finally:
                stop = getattr(watch, "stop", None)
                if callable(stop):
                    stop()
            self._full_resync(namespace=namespace)

    def _full_resync(self, *, namespace: str | None) -> None:
        # Resync replays reconcile for every known job so the operator can
        # recover from missed watch events or transient API/watch failures.
        api = self._load_client().CustomObjectsApi()
        if namespace is None:
            payload = api.list_cluster_custom_object(group=GROUP, version=VERSION, plural=PLURAL)
        else:
            payload = api.list_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
            )
        for item in list(get_path(payload, "items", default=[]) or []):
            self.reconcile(FlashRLJob.model_validate(item))

    def _ensure_finalizer(self, job: FlashRLJob, *, namespace: str) -> None:
        # The finalizer makes job deletion go through operator-managed cleanup
        # instead of orphaning child Deployments, Services, PVCs, and RBAC.
        finalizers = list(job.metadata.get("finalizers", []) or [])
        if self._finalizer in finalizers:
            return
        finalizers.append(self._finalizer)
        self._load_client().CustomObjectsApi().patch_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job.name,
            body={"metadata": {"finalizers": finalizers}},
        )

    def _remove_finalizer(self, job: FlashRLJob, *, namespace: str) -> None:
        # Removing the finalizer is the last finalize step because it tells the
        # API server the operator is done cleaning up this job.
        finalizers = [item for item in list(job.metadata.get("finalizers", []) or []) if item != self._finalizer]
        self._load_client().CustomObjectsApi().patch_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job.name,
            body={"metadata": {"finalizers": finalizers}},
        )

    def _patch_job_status(self, job: FlashRLJob, *, namespace: str, status: dict[str, Any]) -> None:
        """Patch the observed CRD status after one reconcile cycle."""
        self._load_client().CustomObjectsApi().patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            name=job.name,
            body={"status": status},
        )

    def _apply_resource(self, resource: dict[str, Any], *, namespace: str) -> dict[str, Any]:
        """Create or replace one rendered child resource using explicit K8s APIs."""
        kind = resource["kind"]
        client = self._load_client()
        if kind == "ConfigMap":
            api = client.CoreV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_config_map(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_config_map(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_config_map(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        if kind == "Service":
            api = client.CoreV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_service(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_service(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_service(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        if kind == "ServiceAccount":
            api = client.CoreV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_service_account(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_service_account(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_service_account(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        if kind == "PersistentVolumeClaim":
            api = client.CoreV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_persistent_volume_claim(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                ),
                create=lambda body: api.create_namespaced_persistent_volume_claim(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_persistent_volume_claim(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        if kind == "Deployment":
            api = client.AppsV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_deployment(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_deployment(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_deployment(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
                preserve_workload_replicas=True,
            )
        if kind == "StatefulSet":
            api = client.AppsV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_stateful_set(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_stateful_set(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_stateful_set(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
                preserve_workload_replicas=True,
            )
        if kind == "Role":
            api = client.RbacAuthorizationV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_role(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_role(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_role(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        if kind == "RoleBinding":
            api = client.RbacAuthorizationV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_role_binding(name=resource["metadata"]["name"], namespace=namespace),
                create=lambda body: api.create_namespaced_role_binding(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_role_binding(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        if kind == "PodDisruptionBudget":
            api = client.PolicyV1Api()
            return self._upsert_namespaced_resource(
                resource,
                namespace=namespace,
                read=lambda: api.read_namespaced_pod_disruption_budget(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                ),
                create=lambda body: api.create_namespaced_pod_disruption_budget(namespace=namespace, body=body),
                replace=lambda body: api.replace_namespaced_pod_disruption_budget(
                    name=resource["metadata"]["name"],
                    namespace=namespace,
                    body=body,
                ),
            )
        raise ValueError(f"Unsupported rendered resource kind: {kind}")

    def _delete_resource(self, kind: str, name: str, *, namespace: str) -> None:
        """Delete one rendered child resource if it still exists."""
        client = self._load_client()
        try:
            if kind == "ConfigMap":
                client.CoreV1Api().delete_namespaced_config_map(name=name, namespace=namespace)
                return
            if kind == "Service":
                client.CoreV1Api().delete_namespaced_service(name=name, namespace=namespace)
                return
            if kind == "ServiceAccount":
                client.CoreV1Api().delete_namespaced_service_account(name=name, namespace=namespace)
                return
            if kind == "PersistentVolumeClaim":
                client.CoreV1Api().delete_namespaced_persistent_volume_claim(name=name, namespace=namespace)
                return
            if kind == "Deployment":
                client.AppsV1Api().delete_namespaced_deployment(name=name, namespace=namespace)
                return
            if kind == "StatefulSet":
                client.AppsV1Api().delete_namespaced_stateful_set(name=name, namespace=namespace)
                return
            if kind == "Role":
                client.RbacAuthorizationV1Api().delete_namespaced_role(name=name, namespace=namespace)
                return
            if kind == "RoleBinding":
                client.RbacAuthorizationV1Api().delete_namespaced_role_binding(name=name, namespace=namespace)
                return
            if kind == "PodDisruptionBudget":
                client.PolicyV1Api().delete_namespaced_pod_disruption_budget(name=name, namespace=namespace)
                return
            raise ValueError(f"Unsupported rendered resource kind: {kind}")
        except Exception as exc:
            if not is_not_found(exc):
                raise

    def _upsert_namespaced_resource(
        self,
        resource: dict[str, Any],
        *,
        namespace: str,
        read: Callable[[], Any],
        create: Callable[[dict[str, Any]], Any],
        replace: Callable[[dict[str, Any]], Any],
        preserve_workload_replicas: bool = False,
    ) -> dict[str, Any]:
        """Apply one resource while preserving workload replica drift from policy code."""
        body = json.loads(json.dumps(resource))
        try:
            existing = to_plain(read())
        except Exception as exc:
            if not is_not_found(exc):
                raise
            return to_plain(create(body))

        if preserve_workload_replicas:
            existing_replicas = get_path(existing, "spec", "replicas", default=None)
            if existing_replicas is not None:
                body.setdefault("spec", {})["replicas"] = existing_replicas
        resource_version = get_path(existing, "metadata", "resourceVersion", default=None)
        if resource_version is not None:
            body.setdefault("metadata", {})["resourceVersion"] = resource_version
        return to_plain(replace(body))
