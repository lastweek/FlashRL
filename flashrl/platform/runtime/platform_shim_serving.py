"""Platform shim for the serving pod."""

from __future__ import annotations

from flashrl.framework.serving import ServingService, create_serving_backend, create_serving_service_app
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import load_mounted_job, storage_path_from_uri


class PlatformShimServing(PlatformShim):
    """Load the mounted job, then wire the serving backend into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
        serving_artifact_dir = storage_path_from_uri(
            job.spec.storage.weights.uriPrefix,
            purpose="serving-artifacts",
        )
        serving_backend = create_serving_backend(
            job.spec.framework.serving.model_copy(deep=True),
            log_dir=serving_artifact_dir,
        )
        return create_serving_service_app(ServingService(serving_backend))
