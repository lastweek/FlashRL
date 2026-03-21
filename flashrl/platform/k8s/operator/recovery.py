"""Recovery logic for learner and elastic FlashRL platform workloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from flashrl.framework.admin.objects import utc_now_iso
from flashrl.platform.k8s.job import FailurePolicySpec, FlashRLJob, WorkloadStatus
from flashrl.platform.k8s.job_resources import ELASTIC_WORKLOADS, job_workload_name, job_workload_spec
from flashrl.platform.k8s.operator.status import (
    ComponentObservation,
    parse_timestamp,
    seconds_since,
)


# Recovery is readiness-timeout driven. The learner is treated differently from
# elastic pools because its safest reset is a full workload restart, while
# serving/rollout/reward can recover by cycling the Deployment template.
class RecoveryManager:
    """Apply readiness timeout and restart policy decisions."""

    def __init__(self, client_loader: Callable[[], Any], *, now_fn=utc_now_iso) -> None:
        self._load_client = client_loader
        self._now_fn = now_fn

    def apply(
        self,
        job: FlashRLJob,
        observations: dict[str, ComponentObservation],
        *,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """Apply controller, learner, and elastic-pool recovery policies."""
        now_iso = self._now_fn()
        now = parse_timestamp(now_iso) or datetime.now(timezone.utc)
        events: list[dict[str, Any]] = []
        for component, observation in observations.items():
            workload = job_workload_spec(job, component)
            policy = workload.failurePolicy or FailurePolicySpec()
            status = observation.status
            if component == "controller":
                if status.readyReplicas < 1 and status.desiredReplicas > 0:
                    status.phase = "Degraded"
                    status.lastError = "Controller deployment has no ready replicas."
                else:
                    status.lastError = None
                continue

            if component == "learner":
                recovered = self._recover_learner(
                    job,
                    status=status,
                    policy=policy,
                    namespace=namespace,
                    now=now,
                    now_iso=now_iso,
                )
                if recovered is not None:
                    events.append(recovered)
                continue

            if component in ELASTIC_WORKLOADS:
                recovered = self._recover_elastic_pool(
                    job,
                    component=component,
                    status=status,
                    policy=policy,
                    namespace=namespace,
                    now=now,
                    now_iso=now_iso,
                )
                if recovered is not None:
                    events.append(recovered)
        return events

    def _recover_learner(
        self,
        job: FlashRLJob,
        *,
        status: WorkloadStatus,
        policy: FailurePolicySpec,
        namespace: str,
        now: datetime,
        now_iso: str,
    ) -> dict[str, Any] | None:
        """Recover the learner by scaling to zero, backing off, then restoring world size."""
        # Learner recovery is a two-step reset: scale to zero, wait through
        # backoff, then scale back to the configured world size.
        world_size = int(job.spec.framework.actor.dp_size)
        if status.desiredReplicas == 0 and status.lastRecoveryAt is not None:
            if status.readyReplicas == 0 and seconds_since(now, status.lastRecoveryAt) >= float(policy.backoffSeconds):
                self._set_workload_replicas(job, "learner", replicas=world_size, namespace=namespace)
                status.desiredReplicas = world_size
                status.phase = "Recovering"
                status.lastRecoveryAt = now_iso
                return {
                    "event": "recovery_restore",
                    "message": f"Restored learner StatefulSet to {world_size} replicas after backoff.",
                    "component": "learner",
                    "metadata": {"desiredReplicas": world_size},
                }
            return None

        if status.desiredReplicas <= 0 or status.readyReplicas >= status.desiredReplicas:
            status.unreadySince = None
            status.lastError = None
            return None
        if seconds_since(now, status.unreadySince) < float(policy.readinessTimeoutSeconds):
            status.phase = "Degraded"
            status.lastError = "Learner set is below desired readiness."
            return None
        if status.recoveryAttempts >= int(policy.maxRecoveryAttempts):
            status.phase = "Failed"
            status.lastError = "Learner recovery attempts exhausted."
            return None
        if seconds_since(now, status.lastRecoveryAt) < float(policy.backoffSeconds):
            status.phase = "Recovering"
            return None
        self._set_workload_replicas(job, "learner", replicas=0, namespace=namespace)
        status.desiredReplicas = 0
        status.recoveryAttempts += 1
        status.lastRecoveryAt = now_iso
        status.phase = "Recovering"
        status.lastError = "Restarting learner StatefulSet after readiness timeout."
        return {
            "event": "recovery_restart",
            "message": "Restarting learner StatefulSet after readiness timeout.",
            "component": "learner",
            "metadata": {"recoveryAttempts": status.recoveryAttempts},
        }

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
    ) -> dict[str, Any] | None:
        """Recover an elastic pool by forcing a Deployment rollout restart."""
        # Elastic pools recover by forcing a Deployment rollout restart rather
        # than changing their long-term replica target.
        if status.desiredReplicas <= 0 or status.readyReplicas > 0:
            status.unreadySince = None
            if status.phase != "Failed":
                status.lastError = None
            return None
        if seconds_since(now, status.unreadySince) < float(policy.readinessTimeoutSeconds):
            status.phase = "Degraded"
            status.lastError = f"{component} pool has no ready replicas."
            return None
        if status.recoveryAttempts >= int(policy.maxRecoveryAttempts):
            status.phase = "Failed"
            status.lastError = f"{component} recovery attempts exhausted."
            return None
        if seconds_since(now, status.lastRecoveryAt) < float(policy.backoffSeconds):
            status.phase = "Recovering"
            return None
        self._restart_workload(job, component, namespace=namespace, now_iso=now_iso)
        status.recoveryAttempts += 1
        status.lastRecoveryAt = now_iso
        status.phase = "Recovering"
        status.lastError = f"Restarting {component} Deployment after readiness timeout."
        return {
            "event": "recovery_restart",
            "message": f"Restarting {component} Deployment after readiness timeout.",
            "component": component,
            "metadata": {"recoveryAttempts": status.recoveryAttempts},
        }

    def _set_workload_replicas(
        self,
        job: FlashRLJob,
        component: str,
        *,
        replicas: int,
        namespace: str,
    ) -> None:
        """Patch the live replica count for one workload through AppsV1."""
        self._load_client().AppsV1Api().patch_namespaced_stateful_set(
            name=job_workload_name(job, component),
            namespace=namespace,
            body={"spec": {"replicas": int(replicas)}},
        )

    def _restart_workload(
        self,
        job: FlashRLJob,
        component: str,
        *,
        namespace: str,
        now_iso: str,
    ) -> None:
        """Trigger a rollout restart by patching the Deployment pod template."""
        self._load_client().AppsV1Api().patch_namespaced_deployment(
            name=job_workload_name(job, component),
            namespace=namespace,
            body={
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {"flashrl.dev/restartedAt": now_iso}
                        }
                    }
                }
            },
        )
