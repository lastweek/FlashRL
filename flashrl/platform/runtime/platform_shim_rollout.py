"""Platform shim for the rollout pod."""

from __future__ import annotations

from flashrl.framework import runtime_support
from flashrl.framework.distributed import ServingClient
from flashrl.framework.rollout import RolloutService, build_rollout_generator, create_rollout_service_app
from flashrl.framework.serving import RemoteServingBackend
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_pod_logging import PlatformPodLogger
from flashrl.platform.runtime.platform_shim_common import (
    component_log_metadata,
    load_mounted_job,
    service_url_for,
)


def _attach_platform_observability(service, *, job, component: str, event_logger):
    try:
        setattr(service, "_status_metadata", component_log_metadata(job, component))
        setattr(service, "_event_logger", event_logger)
    except Exception:
        pass
    return service


class PlatformShimRollout(PlatformShim):
    """Load the mounted job, then wire rollout logic into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
        self._pod_logger = PlatformPodLogger(job=job, component="rollout")
        self._pod_logger.configure_python_logging()
        self._pod_logger.emit("shim_startup", message="Bootstrapping rollout pod.")
        rollout_hook = runtime_support.instantiate_hook(job.spec.userCode.rollout)
        remote_serving_backend = RemoteServingBackend(
            config=job.spec.framework.serving.model_copy(deep=True),
            client=ServingClient(service_url_for("serving")),
        )
        rollout_generator = build_rollout_generator(
            rollout_fn=rollout_hook,
            serving_backend=remote_serving_backend,
            config=runtime_support.build_rollout_config(job.spec.framework.grpo),
        )
        service = _attach_platform_observability(
            RolloutService(rollout_generator),
            job=job,
            component="rollout",
            event_logger=self._pod_logger.emit,
        )
        return create_rollout_service_app(service)
