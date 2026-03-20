"""Shared serving backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.admin import utc_now_iso
from flashrl.framework.config import ServingConfig
from flashrl.framework.data_models import WeightVersionInfo
from flashrl.framework.distributed.models import WeightVersionRef
from flashrl.framework.models.actor import ActorModel


class ServingBackend(ABC):
    """Shared serving surface used by the rest of the framework."""

    config: ServingConfig
    device: Any
    generation_defaults: dict[str, Any]

    @abstractmethod
    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text responses from prompts."""

    @abstractmethod
    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        """Generate one structured sample per prompt."""

    @abstractmethod
    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[Any]]:
        """Generate prompt-major grouped candidates."""

    @abstractmethod
    def set_generation_defaults(self, **kwargs: Any) -> None:
        """Update default generation kwargs used by rollout code."""

    def set_live_rollout_debug(self, callback: Any, context: dict[str, Any]) -> None:
        """Install a serving-side live-rollout debug callback when supported."""
        del callback, context

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        """Update the framework-owned candidate index for debug streaming."""
        del candidate_index

    def clear_live_rollout_debug(self) -> None:
        """Clear any serving-side live-rollout debug hooks."""
        return None

    def set_log_dir(self, log_dir: str | Any | None) -> None:
        """Update the backend-owned artifact directory when supported."""
        del log_dir
        return None

    def list_admin_objects(self) -> list[dict[str, Any]]:
        """Return backend-owned admin objects when available."""
        return []

    @abstractmethod
    def sync_from_training_actor(
        self,
        training_actor: ActorModel,
        *,
        source_training_step: int | None = None,
        source_epoch: int | None = None,
        origin: str = "sync",
    ) -> WeightVersionInfo:
        """Refresh the serving copy from the training actor."""

    def current_weight_version(self) -> WeightVersionInfo:
        """Return the currently active serving weight version."""
        active = getattr(self, "_active_weight_version", None)
        if not isinstance(active, WeightVersionInfo):
            raise RuntimeError("Serving backend weight version state is not initialized.")
        return active.model_copy(deep=True)

    def weight_sync_status(self) -> dict[str, Any]:
        """Return admin-facing sync/version state when available."""
        active = getattr(self, "_active_weight_version", None)
        pending = getattr(self, "_pending_weight_version", None)
        return {
            "activeWeightVersion": (
                active.model_dump() if isinstance(active, WeightVersionInfo) else None
            ),
            "pendingWeightVersion": (
                pending.model_dump() if isinstance(pending, WeightVersionInfo) else None
            ),
            "lastSuccessfulSyncAt": getattr(self, "_last_successful_sync_at", None),
            "syncHealthy": bool(getattr(self, "_sync_healthy", active is not None)),
            "lastSyncError": getattr(self, "_last_sync_error", None),
        }

    def export_weight_version_state(self) -> dict[str, Any]:
        """Serialize the minimum state required to keep version ids monotonic."""
        active = getattr(self, "_active_weight_version", None)
        next_version_id = getattr(self, "_next_weight_version_id", None)
        if next_version_id is None:
            if isinstance(active, WeightVersionInfo):
                next_version_id = active.version_id + 1
            else:
                next_version_id = 0
        return {
            "schema_version": 1,
            "next_version_id": int(next_version_id),
            "active_weight_version": (
                active.model_dump() if isinstance(active, WeightVersionInfo) else None
            ),
            "last_successful_sync_at": getattr(self, "_last_successful_sync_at", None),
        }

    def restore_weight_version_state(self, state: dict[str, Any] | None) -> None:
        """Restore the next version id from checkpoint metadata before a resume sync."""
        if not isinstance(state, dict):
            return
        raw_next_version_id = state.get("next_version_id")
        if not isinstance(raw_next_version_id, int):
            return
        current_active = getattr(self, "_active_weight_version", None)
        minimum_next_version_id = (
            current_active.version_id + 1 if isinstance(current_active, WeightVersionInfo) else 0
        )
        self._next_weight_version_id = max(int(raw_next_version_id), int(minimum_next_version_id))

    def activate_weight_version_ref(self, weight_version: WeightVersionRef) -> WeightVersionInfo:
        """Activate a previously published weight artifact when supported."""
        current = self.current_weight_version()
        if current.version_id == weight_version.version_id:
            return current
        raise NotImplementedError(
            f"{type(self).__name__} does not support activate_weight_version_ref()."
        )

    def _initialize_weight_version(
        self,
        *,
        model_source: str,
        origin: str = "startup",
    ) -> WeightVersionInfo:
        """Initialize serving version state after backend startup."""
        activated_at = utc_now_iso()
        active = WeightVersionInfo(
            version_id=0,
            source_training_step=None,
            source_epoch=None,
            activated_at=activated_at,
            model_source=str(model_source),
            origin=origin,
        )
        self._active_weight_version = active
        self._pending_weight_version = None
        self._next_weight_version_id = 1
        self._last_successful_sync_at = activated_at
        self._sync_healthy = True
        self._last_sync_error = None
        return active.model_copy(deep=True)

    def _begin_pending_weight_version(
        self,
        *,
        model_source: str,
        source_training_step: int | None,
        source_epoch: int | None,
        origin: str,
    ) -> WeightVersionInfo:
        """Create one pending serving version before activation."""
        pending = WeightVersionInfo(
            version_id=int(getattr(self, "_next_weight_version_id", 0)),
            source_training_step=source_training_step,
            source_epoch=source_epoch,
            activated_at=None,
            model_source=str(model_source),
            origin=origin,
        )
        self._pending_weight_version = pending
        return pending.model_copy(deep=True)

    def _activate_published_weight_version(self, weight_version: WeightVersionRef) -> WeightVersionInfo:
        """Mark one externally published weight version as active."""
        activated = WeightVersionInfo(
            version_id=weight_version.version_id,
            source_training_step=weight_version.source_training_step,
            source_epoch=weight_version.source_epoch,
            activated_at=utc_now_iso(),
            model_source=str(weight_version.artifact_uri),
            origin=weight_version.origin,
        )
        self._active_weight_version = activated
        self._pending_weight_version = None
        self._next_weight_version_id = max(
            int(getattr(self, "_next_weight_version_id", 1)),
            activated.version_id + 1,
        )
        self._last_successful_sync_at = activated.activated_at
        self._sync_healthy = True
        self._last_sync_error = None
        return activated.model_copy(deep=True)

    def _commit_pending_weight_version(self) -> WeightVersionInfo:
        """Activate the currently pending serving version."""
        pending = getattr(self, "_pending_weight_version", None)
        if not isinstance(pending, WeightVersionInfo):
            raise RuntimeError("No pending weight version is available to activate.")
        activated = pending.model_copy(update={"activated_at": utc_now_iso()})
        self._active_weight_version = activated
        self._pending_weight_version = None
        self._next_weight_version_id = activated.version_id + 1
        self._last_successful_sync_at = activated.activated_at
        self._sync_healthy = True
        self._last_sync_error = None
        return activated.model_copy(deep=True)

    def _clear_pending_weight_version(
        self,
        *,
        sync_healthy: bool,
        last_sync_error: str | None,
    ) -> None:
        """Clear pending sync state after a failed or rolled-back attempt."""
        self._pending_weight_version = None
        self._sync_healthy = bool(sync_healthy)
        self._last_sync_error = last_sync_error

    @abstractmethod
    def close(self) -> None:
        """Release any backend-owned resources."""
