"""FastAPI application factory for the FlashRL admin API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from flashrl.framework.admin.objects import build_admin_object_list
from flashrl.framework.admin.registry import AdminRegistry


def create_admin_app(registry: AdminRegistry) -> FastAPI:
    """Build the read-only admin API application."""
    app = FastAPI(
        title="FlashRL Admin API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/admin/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/v1/objects")
    def list_objects() -> dict[str, object]:
        return build_admin_object_list(registry.list_objects())

    @app.get("/admin/v1/objects/{kind}")
    def list_objects_by_kind(kind: str) -> dict[str, object]:
        return build_admin_object_list(registry.list_objects(kind))

    @app.get("/admin/v1/objects/{kind}/{name}")
    def get_object(kind: str, name: str) -> dict[str, object]:
        item = registry.get_object(kind, name)
        if item is None:
            raise HTTPException(status_code=404, detail="not_found")
        return item

    return app
