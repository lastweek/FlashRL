"""Explicit serving pod bootstrap for FlashRL platform mode."""

from __future__ import annotations

import uvicorn

from flashrl.framework.distributed import ServingService, create_serving_service_app
from flashrl.framework.serving import create_serving_backend
from flashrl.platform.runtime.pod import load_mounted_job, storage_path_from_uri


def create_serving_pod_app():
    """Build the serving service app from serving backend plus shared artifact path."""
    job = load_mounted_job()
    serving_artifact_dir = storage_path_from_uri(
        job.spec.storage.weights.uriPrefix,
        purpose="serving-artifacts",
    )
    serving_backend = create_serving_backend(
        job.spec.framework.serving.model_copy(deep=True),
        log_dir=serving_artifact_dir,
    )
    return create_serving_service_app(ServingService(serving_backend))


def run_serving_pod(*, host: str = "0.0.0.0", port: int = 8000) -> int:
    """Run the serving pod HTTP server."""
    uvicorn.run(create_serving_pod_app(), host=host, port=port, log_level="info")
    return 0
