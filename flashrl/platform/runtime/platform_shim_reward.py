"""Platform shim for the reward pod."""

from __future__ import annotations

from flashrl.framework import runtime_support
from flashrl.framework.reward import RewardService, UserDefinedReward, create_reward_service_app
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import load_mounted_job


class PlatformShimReward(PlatformShim):
    """Load the mounted job, then wire the reward hook into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
        reward_hook = runtime_support.instantiate_hook(job.spec.userCode.reward)
        reward = UserDefinedReward(reward_fn=reward_hook, config=job.spec.framework.grpo)
        return create_reward_service_app(RewardService(reward))
