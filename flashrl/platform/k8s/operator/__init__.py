"""Public operator facade for FlashRL platform jobs."""

from __future__ import annotations

from time import sleep
from typing import Any, Callable

from flashrl.framework.admin.objects import utc_now_iso
from flashrl.platform.k8s.job import FlashRLJob, GROUP, PLURAL, VERSION, flashrljob_crd_manifest
from flashrl.platform.k8s.job_resources import render_job_resources
from flashrl.platform.k8s.operator.kube import is_not_found, load_client, load_watch_factory, to_plain
from flashrl.platform.k8s.operator.reconcile import JobReconciler, RESYNC_SECONDS
from flashrl.platform.k8s.operator.recovery import RecoveryManager
from flashrl.platform.k8s.operator.scaling import Autoscaler
from flashrl.platform.k8s.operator.status import StatusCollector


# This module is intentionally small. New readers should start here, then walk
# downward into reconcile, status, scaling, recovery, and store in that order.
FINALIZER = "platform.flashrl.dev/finalizer"


class FlashRLOperator:
    """Thin public facade around the FlashRL Kubernetes operator internals."""

    def __init__(
        self,
        *,
        client_module: Any | None = None,
        watch_factory: Callable[[], Any] | None = None,
        http_getter: Callable[[str], dict[str, Any]] | None = None,
        sleep_fn: Callable[[float], None] = sleep,
        now_fn: Callable[[], str] = utc_now_iso,
    ) -> None:
        # The facade wires together five responsibilities:
        # - client/watch loading: lazy access to the raw Kubernetes APIs
        # - status: observe workloads and build CRD status payloads
        # - scaling: adjust elastic pools when demand rises or falls
        # - recovery: restart unhealthy workloads after timeout/backoff
        # - reconciler: run one end-to-end reconcile cycle for a job
        self._client_module = client_module
        self._watch_factory = watch_factory
        self._status = StatusCollector(self._load_client, http_getter=http_getter, now_fn=now_fn)
        self._autoscaler = Autoscaler(self._load_client, now_fn=now_fn)
        self._recovery = RecoveryManager(self._load_client, now_fn=now_fn)
        self._reconciler = JobReconciler(
            self._load_client,
            self._load_watch_factory,
            self._status,
            self._autoscaler,
            self._recovery,
            finalizer=FINALIZER,
            sleep_fn=sleep_fn,
        )

    def ensure_crd(self) -> None:
        """Install or refresh the FlashRLJob CRD in the Kubernetes API server.

        The operator and CLI both call this before they create, list, or watch
        FlashRLJob objects so the cluster knows that custom resource type.
        """
        api = self._load_client().ApiextensionsV1Api()
        manifest = flashrljob_crd_manifest()
        try:
            api.read_custom_resource_definition(manifest["metadata"]["name"])
            api.replace_custom_resource_definition(manifest["metadata"]["name"], manifest)
        except Exception as exc:
            if not is_not_found(exc):
                raise
            api.create_custom_resource_definition(manifest)

    def render(self, job: FlashRLJob) -> list[dict[str, Any]]:
        """Render the operator-owned child manifests for one validated job.

        This is a pure manifest step. It does not contact Kubernetes or mutate
        cluster state.
        """
        return render_job_resources(job)

    def apply_job(self, job: FlashRLJob, *, namespace: str = "default") -> dict[str, Any]:
        """Create or replace the FlashRLJob custom resource in the cluster."""
        api = self._load_client().CustomObjectsApi()
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
            return to_plain(
                api.replace_namespaced_custom_object(
                    group=GROUP,
                    version=VERSION,
                    namespace=namespace,
                    plural=PLURAL,
                    name=job.name,
                    body=body,
                )
            )
        except Exception as exc:
            if not is_not_found(exc):
                raise
            return to_plain(
                api.create_namespaced_custom_object(
                    group=GROUP,
                    version=VERSION,
                    namespace=namespace,
                    plural=PLURAL,
                    body=body,
                )
            )

    def get_job(self, name: str, *, namespace: str = "default") -> dict[str, Any]:
        """Fetch one FlashRLJob custom resource from the Kubernetes API."""
        return to_plain(
            self._load_client().CustomObjectsApi().get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=name,
            )
        )

    def delete_job(self, name: str, *, namespace: str = "default") -> dict[str, Any]:
        """Delete one FlashRLJob custom resource from the Kubernetes API."""
        return to_plain(
            self._load_client().CustomObjectsApi().delete_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=name,
            )
        )

    def reconcile_job(self, namespace: str, name: str) -> dict[str, Any]:
        """Fetch one job from Kubernetes, validate it, and run one reconcile."""
        return self._reconciler.reconcile_job(namespace, name)

    def reconcile(self, job: FlashRLJob) -> dict[str, Any]:
        """Run one reconcile cycle for an already validated FlashRLJob model."""
        return self._reconciler.reconcile(job)

    def finalize_job(self, job: FlashRLJob) -> dict[str, Any]:
        """Delete operator-owned children and remove the operator finalizer."""
        return self._reconciler.finalize(job)

    def watch(self, *, namespace: str | None = None, resync_seconds: int = RESYNC_SECONDS) -> None:
        """Run the long-lived operator watch loop for one namespace or cluster-wide."""
        self._reconciler.watch(namespace=namespace, resync_seconds=resync_seconds)

    def submit_job(self, job: FlashRLJob, *, namespace: str = "default") -> dict[str, Any]:
        """Legacy alias for `apply_job`; keep older callers working unchanged."""
        return self.apply_job(job, namespace=namespace)

    def reconcile_payload(self, job: FlashRLJob) -> dict[str, Any]:
        """Legacy alias for `reconcile`; prefer the canonical method name."""
        return self.reconcile(job)

    def run_forever(self, *, namespace: str | None = None, resync_seconds: int = RESYNC_SECONDS) -> None:
        """Legacy alias for `watch`; prefer the canonical method name."""
        self.watch(namespace=namespace, resync_seconds=resync_seconds)

    def _load_client(self) -> Any:
        """Load and cache the Kubernetes client module for operator internals."""
        self._client_module = load_client(self._client_module)
        return self._client_module

    def _load_watch_factory(self) -> Callable[[], Any]:
        """Load and cache the Kubernetes watch factory for the watch loop."""
        self._watch_factory = load_watch_factory(self._watch_factory)
        return self._watch_factory
