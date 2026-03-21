"""Platform shim for the serving pod."""

from __future__ import annotations

from flashrl.framework.serving import ServingService, create_serving_backend, create_serving_service_app
from flashrl.platform.runtime.platform_pod_logging import PlatformPodLogger
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import (
    component_log_metadata,
    load_mounted_job,
    storage_path_from_uri,
)


def _attach_platform_observability(service, *, job, event_logger):
    try:
        setattr(service, "_status_metadata", component_log_metadata(job, "serving"))
        setattr(service, "_event_logger", event_logger)
    except Exception:
        pass
    return service


class PlatformShimServing(PlatformShim):
    """Load the mounted job, then wire the serving backend into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
        self._pod_logger = PlatformPodLogger(job=job, component="serving")
        self._pod_logger.configure_python_logging()
        self._pod_logger.emit("shim_startup", message="Bootstrapping serving pod.")
        serving_artifact_dir = storage_path_from_uri(
            job.spec.storage.weights.uriPrefix,
            purpose="serving-artifacts",
        )
        serving_backend = create_serving_backend(
            job.spec.framework.serving.model_copy(deep=True),
            log_dir=serving_artifact_dir,
        )
        service = _attach_platform_observability(
            ServingService(serving_backend),
            job=job,
            event_logger=self._pod_logger.emit,
        )
        return create_serving_service_app(service)
