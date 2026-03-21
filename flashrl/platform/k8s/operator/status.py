"""Observation and status summary logic for the FlashRL platform operator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Callable
from urllib import request as urllib_request

from flashrl.framework.admin.objects import utc_now_iso
from flashrl.framework.distributed.models import ComponentStatus
from flashrl.platform.k8s.job import ConditionStatus, FlashRLJob, WorkloadStatus
from flashrl.platform.k8s.job_resources import (
    HTTP_PORT,
    JOB_WORKLOADS,
    desired_job_workload_replicas,
    job_workload_kind,
    job_workload_name,
    job_workload_selector_labels,
)
from flashrl.platform.k8s.operator.kube import get_path, is_not_found, to_plain


READINESS_TIMEOUT_SECONDS = 3.0


# This module translates two different views of the system into CRD status:
# Kubernetes tells us about pods/workloads, while component /v1/status reports
# queue depth, weight versions, and health from inside each service.
@dataclass
class LoadSnapshot:
    """Aggregated queue and latency metrics for one component pool."""

    available: bool
    queue_depth: int
    inflight_requests: int
    p95_latency_seconds: float


@dataclass
class ComponentObservation:
    """Current observed state for one workload component."""

    workload: dict[str, Any]
    pods: list[dict[str, Any]]
    pod_statuses: list[ComponentStatus]
    status: WorkloadStatus
    load: LoadSnapshot


@dataclass
class JobStatusSummary:
    """Top-level job phase, conditions, and last error."""

    phase: str
    conditions: list[ConditionStatus]
    last_error: str | None


def parse_timestamp(timestamp: str | None) -> datetime | None:
    """Parse one stored timestamp into a datetime used by policy code."""
    if timestamp is None:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def seconds_since(now: datetime, timestamp: str | None) -> float:
    """Return elapsed seconds since a stored timestamp or infinity when missing."""
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return float("inf")
    return max((now - parsed).total_seconds(), 0.0)


def pod_ready(pod: dict[str, Any]) -> bool:
    """Return whether a pod should count as Ready for operator decisions."""
    if get_path(pod, "metadata", "deletionTimestamp") is not None:
        return False
    conditions = get_path(pod, "status", "conditions", default=[]) or []
    for condition in conditions:
        if condition.get("type") == "Ready":
            return condition.get("status") == "True"
    return False


def restart_count(pod: dict[str, Any]) -> int:
    """Return the summed restart count across all containers in one pod."""
    statuses = get_path(pod, "status", "containerStatuses", default=[]) or []
    return sum(int(item.get("restartCount", 0) or 0) for item in statuses)


def default_http_get(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Fetch one component `/v1/status` payload over HTTP."""
    request = urllib_request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def aggregate_load(statuses: list[ComponentStatus]) -> LoadSnapshot:
    """Aggregate pod-level queue and latency metrics into one pool snapshot."""
    eligible = [status for status in statuses if not bool(status.metadata.get("draining", False))]
    if not eligible:
        return LoadSnapshot(
            available=False,
            queue_depth=0,
            inflight_requests=0,
            p95_latency_seconds=0.0,
        )
    try:
        queue_depth = sum(int(status.metadata["queueDepth"]) for status in eligible)
        inflight_requests = sum(int(status.metadata["inflightRequests"]) for status in eligible)
        p95_latency = max(float(status.metadata["p95LatencySeconds"]) for status in eligible)
    except (KeyError, TypeError, ValueError):
        return LoadSnapshot(
            available=False,
            queue_depth=0,
            inflight_requests=0,
            p95_latency_seconds=0.0,
        )
    return LoadSnapshot(
        available=True,
        queue_depth=queue_depth,
        inflight_requests=inflight_requests,
        p95_latency_seconds=p95_latency,
    )


def build_condition(
    *,
    previous: dict[str, ConditionStatus],
    condition_type: str,
    status: str,
    reason: str,
    message: str | None,
    now_iso: str,
) -> ConditionStatus:
    """Build one CRD condition while preserving the prior transition timestamp."""
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


class StatusCollector:
    """Collect workload observations and summarize job status."""

    def __init__(
        self,
        client_loader: Callable[[], Any],
        *,
        http_getter: Any | None = None,
        now_fn: Any = utc_now_iso,
    ) -> None:
        self._load_client = client_loader
        self._http_getter = http_getter
        self._now_fn = now_fn

    def collect(self, job: FlashRLJob, *, namespace: str) -> dict[str, ComponentObservation]:
        """Collect current component observations for one job."""
        now_iso = self._now_fn()
        observations: dict[str, ComponentObservation] = {}
        for component in JOB_WORKLOADS:
            # First gather raw workload state from Kubernetes plus live
            # per-pod service status. That gives us one observation bundle per
            # component before any higher-level job summary is computed.
            previous = job.status.components.get(component, WorkloadStatus())
            workload = self._read_workload(job, component, namespace=namespace)
            desired = int(
                get_path(
                    workload,
                    "spec",
                    "replicas",
                    default=desired_job_workload_replicas(job, component),
                )
                or 0
            )
            ready = int(get_path(workload, "status", "readyReplicas", default=0) or 0)
            available = int(get_path(workload, "status", "availableReplicas", default=ready) or 0)
            pods = self._list_component_pods(job, component, namespace=namespace)
            total_restarts = sum(restart_count(pod) for pod in pods)
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

            # WorkloadStatus is the CRD-facing summary for one component. It is
            # intentionally derived here so scaling and recovery can make
            # decisions on a stable, normalized shape.
            status = WorkloadStatus(
                phase=phase,
                readyReplicas=ready,
                availableReplicas=available,
                desiredReplicas=desired,
                worldSize=int(job.spec.framework.actor.dp_size) if component == "learner" else None,
                activeWeightVersion=active_weight_version,
                restartCount=total_restarts,
                recoveryAttempts=int(previous.recoveryAttempts),
                lastScaleAt=previous.lastScaleAt,
                lastObservedAt=now_iso,
                lastError=status_error or previous.lastError,
                lowLoadSince=previous.lowLoadSince,
                unreadySince=unready_since,
                lastRecoveryAt=previous.lastRecoveryAt,
            )
            observations[component] = ComponentObservation(
                workload=workload,
                pods=pods,
                pod_statuses=pod_statuses,
                status=status,
                load=aggregate_load(pod_statuses),
            )
        return observations

    def summarize(
        self,
        job: FlashRLJob,
        observations: dict[str, ComponentObservation],
    ) -> JobStatusSummary:
        """Summarize top-level phase, conditions, and last error."""
        # Job phase is intentionally computed from component summaries rather
        # than raw pod states so autoscaling/recovery updates are reflected in
        # the same reconcile cycle.
        if job.spec.suspend:
            phase = "Suspended"
        elif any(observation.status.phase == "Failed" for observation in observations.values()):
            phase = "Failed"
        elif self._serving_convergence_blocked(job, observations.get("serving")):
            phase = "Degraded"
        elif all(observation.status.phase == "Ready" for observation in observations.values()):
            phase = "Ready"
        elif any(observation.status.phase in {"Recovering", "Degraded"} for observation in observations.values()):
            phase = "Degraded"
        else:
            phase = "Pending"

        convergence_message = None
        if self._serving_convergence_blocked(job, observations.get("serving")):
            desired = get_path(job.status.weightVersion.desired, "version_id", default=None)
            convergence_message = f"Serving pool has not converged to desired weight version {desired}."

        component_errors = [
            observation.status.lastError
            for observation in observations.values()
            if observation.status.lastError
        ]
        last_error = convergence_message or (component_errors[0] if component_errors else None)
        now_iso = self._now_fn()
        previous_conditions = {condition.type: condition for condition in job.status.conditions}
        conditions = [
            build_condition(
                previous=previous_conditions,
                condition_type="Ready",
                status="True" if phase == "Ready" else "False",
                reason="AllComponentsReady" if phase == "Ready" else "ComponentsNotReady",
                message=None if phase == "Ready" else last_error,
                now_iso=now_iso,
            ),
            build_condition(
                previous=previous_conditions,
                condition_type="Degraded",
                status="True" if phase == "Degraded" else "False",
                reason="ComponentDegraded" if phase == "Degraded" else "NoDegradation",
                message=last_error if phase == "Degraded" else None,
                now_iso=now_iso,
            ),
            build_condition(
                previous=previous_conditions,
                condition_type="Suspended",
                status="True" if phase == "Suspended" else "False",
                reason="JobSuspended" if phase == "Suspended" else "JobActive",
                message=None,
                now_iso=now_iso,
            ),
            build_condition(
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
                build_condition(
                    previous=previous_conditions,
                    condition_type="ServingConvergenceBlocked",
                    status="True",
                    reason="ServingConvergenceBlocked",
                    message=convergence_message,
                    now_iso=now_iso,
                )
            )
        return JobStatusSummary(phase=phase, conditions=conditions, last_error=last_error)

    def build_status_patch(
        self,
        job: FlashRLJob,
        *,
        summary: JobStatusSummary,
        observations: dict[str, ComponentObservation],
    ) -> dict[str, Any]:
        """Build one CRD status patch from current observations.

        The patch starts from the current status so controller-owned fields such
        as progress and weight version tracking survive operator updates.
        """
        status = job.status.model_dump(mode="json")
        status.update(
            {
                "observedGeneration": int(job.metadata.get("generation", 0) or 0),
                "phase": summary.phase,
                "components": {
                    name: observation.status.model_dump(mode="json")
                    for name, observation in observations.items()
                },
                "conditions": [condition.model_dump(mode="json") for condition in summary.conditions],
                "lastError": summary.last_error,
            }
        )
        return status

    def _read_workload(self, job: FlashRLJob, component: str, *, namespace: str) -> dict[str, Any]:
        # Workload observation reads the concrete Deployment or StatefulSet
        # directly from Kubernetes instead of going through an operator repository.
        apps_api = self._load_client().AppsV1Api()
        workload_name = job_workload_name(job, component)
        try:
            if job_workload_kind(component) == "StatefulSet":
                return to_plain(apps_api.read_namespaced_stateful_set(name=workload_name, namespace=namespace))
            return to_plain(apps_api.read_namespaced_deployment(name=workload_name, namespace=namespace))
        except Exception as exc:
            if is_not_found(exc):
                return {}
            raise

    def _list_component_pods(self, job: FlashRLJob, component: str, *, namespace: str) -> list[dict[str, Any]]:
        # Pod discovery is selector-based so status logic follows the same labels
        # Kubernetes uses to back the rendered workloads and services.
        selector = ",".join(
            f"{key}={value}" for key, value in sorted(job_workload_selector_labels(job, component).items())
        )
        payload = to_plain(
            self._load_client().CoreV1Api().list_namespaced_pod(
                namespace=namespace,
                label_selector=selector,
            )
        )
        return list(get_path(payload, "items", default=[]) or [])

    def _phase_for_component(
        self,
        *,
        desired: int,
        ready: int,
        available: int,
        suspended: bool,
        recovering: bool,
        failed: bool,
    ) -> str:
        # Component phase is a normalized operator view used by scaling,
        # recovery, and top-level job summarization.
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
        # Only Ready pods are queried for /v1/status; unready pods already show
        # up through Kubernetes readiness and should not block status fetches.
        statuses: list[ComponentStatus] = []
        errors: list[str] = []
        for pod in pods:
            if not pod_ready(pod):
                continue
            pod_ip = get_path(pod, "status", "podIP", default=None)
            if not pod_ip:
                continue
            try:
                statuses.append(self._fetch_pod_status(str(pod_ip)))
            except Exception as exc:
                errors.append(str(exc))
        return statuses, (errors[0] if errors else None)

    def _fetch_pod_status(self, pod_ip: str) -> ComponentStatus:
        # Components expose a stable /v1/status contract so the operator can
        # reason about queue depth, inflight load, and active weights.
        getter = self._http_getter
        if getter is None:
            getter = lambda url: default_http_get(url, timeout_seconds=READINESS_TIMEOUT_SECONDS)
        payload = getter(f"http://{pod_ip}:{HTTP_PORT}/v1/status")
        return ComponentStatus.model_validate(get_path(payload, "status", default=payload))

    def _serving_convergence_blocked(
        self,
        job: FlashRLJob,
        observation: ComponentObservation | None,
    ) -> bool:
        # Serving convergence is a cluster-level signal: even if pods are Ready,
        # the job is still degraded when the active serving weight does not
        # match the desired learner-published version.
        if observation is None:
            return False
        desired_version = get_path(job.status.weightVersion.desired, "version_id", default=None)
        if desired_version is None:
            return False
        active_versions = {
            status.active_weight_version.version_id
            for status in observation.pod_statuses
            if status.active_weight_version is not None and not bool(status.metadata.get("draining", False))
        }
        if not active_versions:
            return False
        return len(active_versions) != 1 or desired_version not in active_versions
