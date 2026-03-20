"""Common helpers for component service apps."""

from __future__ import annotations

from fastapi import FastAPI

from flashrl.framework.admin.objects import build_admin_object, build_admin_object_list, utc_now_iso
from flashrl.framework.distributed.models import ComponentStatus


def install_common_routes(
    app: FastAPI,
    *,
    status_getter,
    kind: str,
    name: str,
) -> None:
    """Install shared health, status, and admin routes."""

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        status = status_getter()
        return {"status": "ok" if status.healthy else "degraded"}

    @app.get("/v1/status")
    def status():
        return {"status": status_getter().model_dump()}

    @app.get("/admin/v1/objects")
    def admin_objects():
        status = status_getter()
        item = build_admin_object(
            kind,
            name,
            uid=f"{kind.lower()}:{name}",
            created_at=utc_now_iso(),
            spec={"component": name},
            status={
                "phase": status.phase,
                "healthy": status.healthy,
                "readyReplicaCount": status.ready_replica_count,
                "desiredReplicaCount": status.desired_replica_count,
                "activeWeightVersion": (
                    status.active_weight_version.model_dump()
                    if status.active_weight_version is not None
                    else None
                ),
                "lastError": status.last_error,
                **dict(status.metadata),
            },
        )
        return build_admin_object_list([item])


def unwrap_status(payload: dict[str, object]) -> ComponentStatus:
    """Parse a shared ``/v1/status`` payload."""
    return ComponentStatus.model_validate(payload["status"])
