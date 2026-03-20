"""Platform layer: Orchestration, scaling, and observability."""

from flashrl.platform.cli import build_argument_parser
from flashrl.platform.crd import (
    FlashRLJob,
    FlashRLJobSpec,
    FlashRLJobStatus,
    FrameworkSpec,
    flashrljob_crd_manifest,
)
from flashrl.platform.operator import FlashRLOperator, render_child_resources

__all__ = [
    "FlashRLJob",
    "FlashRLJobSpec",
    "FlashRLJobStatus",
    "FrameworkSpec",
    "flashrljob_crd_manifest",
    "render_child_resources",
    "FlashRLOperator",
    "build_argument_parser",
]
