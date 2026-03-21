"""Platform shim for the rollout pod."""

from __future__ import annotations

from flashrl.framework import runtime_support
from flashrl.framework.distributed import ServingClient
from flashrl.framework.rollout import RolloutService, build_rollout_generator, create_rollout_service_app
from flashrl.framework.serving import RemoteServingBackend
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import load_mounted_job, service_url_for


class PlatformShimRollout(PlatformShim):
    """Load the mounted job, then wire rollout logic into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
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
