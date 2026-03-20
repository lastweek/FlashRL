"""Protocols for distributed FlashRL controller dependencies."""

from __future__ import annotations

from typing import Protocol

from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    ActivateWeightVersionResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    OptimizeStepRequest,
    OptimizeStepResponse,
    RewardBatchRequest,
    RewardBatchResponse,
    RolloutBatchRequest,
    RolloutBatchResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    StatusResponse,
)


class RolloutClient(Protocol):
    """Controller-facing rollout RPC contract."""

    def rollout_batch(self, request: RolloutBatchRequest) -> RolloutBatchResponse:
        """Generate grouped rollout batches."""

    def status(self) -> StatusResponse:
        """Return the current rollout component status."""


class RewardClient(Protocol):
    """Controller-facing reward RPC contract."""

    def reward_batch(self, request: RewardBatchRequest) -> RewardBatchResponse:
        """Score a rollout batch."""

    def status(self) -> StatusResponse:
        """Return the current reward component status."""


class LearnerClient(Protocol):
    """Controller-facing learner RPC contract."""

    def optimize_step(self, request: OptimizeStepRequest) -> OptimizeStepResponse:
        """Optimize one learner batch and publish a weight artifact."""

    def save_checkpoint(self, request: SaveCheckpointRequest) -> SaveCheckpointResponse:
        """Persist one learner checkpoint."""

    def load_checkpoint(self, request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        """Load one learner checkpoint."""

    def status(self) -> StatusResponse:
        """Return the current learner component status."""


class ServingClient(Protocol):
    """Controller-facing serving RPC contract."""

    def activate_weight_version(
        self,
        request: ActivateWeightVersionRequest,
    ) -> ActivateWeightVersionResponse:
        """Activate a published weight version across the serving pool."""

    def status(self) -> StatusResponse:
        """Return the current serving component status."""
