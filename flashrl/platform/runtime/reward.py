"""Explicit reward pod bootstrap for FlashRL platform mode."""

from __future__ import annotations

import uvicorn

from flashrl.framework import runtime_support
from flashrl.framework.distributed import RewardService, create_reward_service_app
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.platform.runtime.pod import load_mounted_job


def create_reward_pod_app():
    """Build the reward service app from the mounted job plus reward hook."""
    job = load_mounted_job()
    reward_hook = runtime_support.instantiate_hook(job.spec.userCode.reward)
    reward = UserDefinedReward(reward_fn=reward_hook, config=job.spec.framework.grpo)
    return create_reward_service_app(RewardService(reward))


def run_reward_pod(*, host: str = "0.0.0.0", port: int = 8000) -> int:
    """Run the reward pod HTTP server."""
    uvicorn.run(create_reward_pod_app(), host=host, port=port, log_level="info")
    return 0
