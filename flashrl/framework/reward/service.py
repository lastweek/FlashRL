"""Domain-owned reward service implementation and HTTP app builder."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI

from flashrl.framework.distributed.http_common import install_common_routes
from flashrl.framework.distributed.models import (
    ComponentStatus,
    RewardBatchRequest,
    RewardBatchResponse,
    StatusResponse,
)
from flashrl.framework.memory import capture_memory_snapshot
from flashrl.framework.reward.user_defined import UserDefinedReward


class RewardService:
    """In-process reward service over one user-defined reward."""

    def __init__(
        self,
        reward: UserDefinedReward,
        *,
        status_metadata: dict[str, Any] | None = None,
        event_logger: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._reward = reward
        self._status_metadata = dict(status_metadata or {})
        self._event_logger = event_logger

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
                metadata={
                    "memory": capture_memory_snapshot(getattr(self._reward, "device", None)),
                    **dict(self._status_metadata),
                },
            )
        )


def create_reward_service_app(service: RewardService) -> FastAPI:
    """Create a reward RPC app around one local reward service."""
    app = FastAPI(title="FlashRL Reward Service")
    install_common_routes(
        app,
        status_getter=lambda: service.status().status,
        kind="RewardService",
        name="reward",
        drainable=True,
        event_logger=service._event_logger,
    )

    @app.post("/v1/reward-batches")
    def reward_batches(request: RewardBatchRequest) -> RewardBatchResponse:
        return service.reward_batch(request)

    return app
