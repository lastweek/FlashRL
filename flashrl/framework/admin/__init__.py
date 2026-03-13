"""Live admin API primitives for FlashRL runtimes."""

from flashrl.framework.admin.app import create_admin_app
from flashrl.framework.admin.objects import ADMIN_API_VERSION, build_admin_object, build_admin_object_list, utc_now_iso
from flashrl.framework.admin.registry import AdminRegistry
from flashrl.framework.admin.server import AdminServer

__all__ = [
    "ADMIN_API_VERSION",
    "AdminRegistry",
    "AdminServer",
    "build_admin_object",
    "build_admin_object_list",
    "create_admin_app",
    "utc_now_iso",
]
