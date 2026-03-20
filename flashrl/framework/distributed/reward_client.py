"""Reward RPC clients."""

from __future__ import annotations

from flashrl.framework.distributed.client_common import _HttpJsonClient
from flashrl.framework.distributed.models import (
    ComponentStatus,
    RewardBatchRequest,
    RewardBatchResponse,
    StatusResponse,
)
from flashrl.framework.reward.user_defined import UserDefinedReward


class LocalRewardClient:
    """In-process controller adapter over one user-defined reward."""

    def __init__(self, reward: UserDefinedReward) -> None:
        self._reward = reward

    def reward_batch(self, request: RewardBatchRequest) -> RewardBatchResponse:
        rewards = self._reward.compute_batch(request.rollouts)
        reward_values = [reward.reward for reward in rewards]
        reward_mean = sum(reward_values) / len(reward_values) if reward_values else 0.0
        return RewardBatchResponse(
            api_version=request.api_version,
            job_id=request.job_id,
            run_id=request.run_id,
            step_id=request.step_id,
            trace_id=request.trace_id,
            deadline_ms=request.deadline_ms,
            rewards=rewards,
            metrics={"reward_mean": reward_mean, "sample_count": len(rewards)},
        )

    def status(self) -> StatusResponse:
        return StatusResponse(
            status=ComponentStatus(
                name="reward",
                phase="Ready",
                healthy=True,
                ready_replica_count=1,
                desired_replica_count=1,
            )
        )


class HttpRewardClient(_HttpJsonClient):
    """HTTP reward client."""

    def reward_batch(self, request: RewardBatchRequest) -> RewardBatchResponse:
        return self._post("/v1/reward-batches", request, RewardBatchResponse)

    def status(self) -> StatusResponse:
        return self._get("/v1/status", StatusResponse)
