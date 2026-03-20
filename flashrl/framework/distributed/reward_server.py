"""FastAPI app for the reward service."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.distributed.models import RewardBatchRequest, RewardBatchResponse
from flashrl.framework.distributed.reward_client import LocalRewardClient
from flashrl.framework.distributed.server_common import install_common_routes


def create_reward_app(client: LocalRewardClient) -> FastAPI:
    """Create a reward RPC app around one local reward adapter."""
    app = FastAPI(title="FlashRL Reward Service")
    install_common_routes(
        app,
        status_getter=lambda: client.status().status,
        kind="RewardService",
        name="reward",
        drainable=True,
    )

    @app.post("/v1/reward-batches")
    def reward_batches(request: RewardBatchRequest) -> RewardBatchResponse:
        return client.reward_batch(request)

    return app
