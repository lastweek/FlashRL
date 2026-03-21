"""Explicit rollout pod bootstrap for FlashRL platform mode."""

from __future__ import annotations

import uvicorn

from flashrl.framework import runtime_support
from flashrl.framework.distributed import (
    RolloutService,
    ServingClient,
    create_rollout_service_app,
)
from flashrl.framework.distributed.remote_serving_backend import RemoteServingBackend
from flashrl.framework.rollout.base import build_rollout_generator
from flashrl.platform.runtime.pod import load_mounted_job, service_url_for


def create_rollout_pod_app():
    """Build the rollout service app from the mounted job plus serving client."""
    job = load_mounted_job()
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
    return create_rollout_service_app(RolloutService(rollout_generator))


def run_rollout_pod(*, host: str = "0.0.0.0", port: int = 8000) -> int:
    """Run the rollout pod HTTP server."""
    uvicorn.run(create_rollout_pod_app(), host=host, port=port, log_level="info")
    return 0
