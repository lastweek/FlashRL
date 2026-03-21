"""Autoscaling logic for elastic FlashRL platform workloads."""

from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from typing import Any, Callable

from flashrl.framework.admin.objects import utc_now_iso
from flashrl.platform.k8s.job import AutoscalingSpec, FlashRLJob, WorkloadStatus
from flashrl.platform.k8s.job_resources import ELASTIC_WORKLOADS, job_workload_name, job_workload_spec
from flashrl.platform.k8s.operator.status import (
    ComponentObservation,
    LoadSnapshot,
    parse_timestamp,
    seconds_since,
)


# Scaling is demand-driven but intentionally conservative. Queue depth and
# latency increase capacity, while cooldown and stabilization windows avoid
# thrashing when traffic briefly spikes or falls to zero.
class Autoscaler:
    """Compute and apply replica changes for elastic pools."""

    def __init__(self, client_loader: Callable[[], Any], *, now_fn=utc_now_iso) -> None:
        self._load_client = client_loader
        self._now_fn = now_fn

    def apply(
        self,
        job: FlashRLJob,
        observations: dict[str, ComponentObservation],
        *,
        namespace: str,
    ) -> None:
        """Apply autoscaling policy to the elastic workload pools only."""
        now_iso = self._now_fn()
        now = parse_timestamp(now_iso) or datetime.now(timezone.utc)
        for component in ELASTIC_WORKLOADS:
            workload = job_workload_spec(job, component)
            policy = workload.autoscaling or AutoscalingSpec()
            if not policy.enabled or workload.replicas is None:
                continue
            observation = observations[component]
            current = int(observation.status.desiredReplicas or workload.replicas.min)
            desired, low_load_since = self.desired_scale_target(
                policy=policy,
                current_replicas=current,
                min_replicas=int(workload.replicas.min),
                max_replicas=int(workload.replicas.max),
                load=observation.load,
                previous=observation.status,
                now=now,
                now_iso=now_iso,
            )
            observation.status.lowLoadSince = low_load_since
            if desired != current:
                self._patch_replicas(job, component, replicas=desired, namespace=namespace)
                observation.status.desiredReplicas = desired
                observation.status.lastScaleAt = now_iso

    def desired_scale_target(
        self,
        *,
        policy: AutoscalingSpec,
        current_replicas: int,
        min_replicas: int,
        max_replicas: int,
        load: LoadSnapshot,
        previous: WorkloadStatus,
        now: datetime,
        now_iso: str,
    ) -> tuple[int, str | None]:
        """Compute a raw load target, then constrain it through scaling guards."""
        if not load.available:
            return min_replicas, None

        # First compute the demand-based target, then clamp it through
        # cooldown/stabilization rules before the operator patches replicas.
        queue_depth = int(load.queue_depth)
        inflight_requests = int(load.inflight_requests)
        p95_latency = float(load.p95_latency_seconds)
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
            if seconds_since(now, previous.lastScaleAt) < float(policy.scaleUpCooldownSeconds):
                return current_replicas, None
            return min(current_replicas + int(policy.scaleUpStep), desired), None

        if desired < current_replicas:
            if queue_depth > 0 or inflight_requests > 0:
                return current_replicas, None
            low_load_since = previous.lowLoadSince or now_iso
            if seconds_since(now, low_load_since) < float(policy.scaleDownStabilizationSeconds):
                return current_replicas, low_load_since
            if seconds_since(now, previous.lastScaleAt) < float(policy.scaleDownCooldownSeconds):
                return current_replicas, low_load_since
            return max(current_replicas - int(policy.scaleDownStep), desired), low_load_since

        if queue_depth == 0 and inflight_requests == 0:
            return current_replicas, previous.lowLoadSince or now_iso
        return current_replicas, None

    def _patch_replicas(self, job: FlashRLJob, component: str, *, replicas: int, namespace: str) -> None:
        """Patch one workload's live replica count through the explicit AppsV1 API."""
        apps_api = self._load_client().AppsV1Api()
        body = {"spec": {"replicas": int(replicas)}}
        apps_api.patch_namespaced_deployment(
            name=job_workload_name(job, component),
            namespace=namespace,
            body=body,
        )
