"""Platform shim for the reward pod."""

from __future__ import annotations

from flashrl.framework import runtime_support
from flashrl.framework.reward import RewardService, UserDefinedReward, create_reward_service_app
from flashrl.platform.runtime.platform_pod_logging import PlatformPodLogger
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import component_log_metadata, load_mounted_job


def _attach_platform_observability(service, *, job, event_logger):
    try:
        setattr(service, "_status_metadata", component_log_metadata(job, "reward"))
        setattr(service, "_event_logger", event_logger)
    except Exception:
        pass
    return service


class PlatformShimReward(PlatformShim):
    """Load the mounted job, then wire the reward hook into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
        self._pod_logger = PlatformPodLogger(job=job, component="reward")
        self._pod_logger.configure_python_logging()
        self._pod_logger.emit("shim_startup", message="Bootstrapping reward pod.")
        reward_hook = runtime_support.instantiate_hook(job.spec.userCode.reward)
        reward = UserDefinedReward(reward_fn=reward_hook, config=job.spec.framework.grpo)
        service = _attach_platform_observability(
            RewardService(reward),
            job=job,
            event_logger=self._pod_logger.emit,
        )
        return create_reward_service_app(service)
