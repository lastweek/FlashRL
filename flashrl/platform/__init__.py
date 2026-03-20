"""Platform layer: Orchestration, scaling, and observability."""

from flashrl.platform.cli import build_argument_parser
from flashrl.platform.config import PlatformConfig, build_flashrl_job, load_flashrl_config
from flashrl.platform.crd import (
    FlashRLJob,
    FlashRLJobSpec,
    FlashRLJobStatus,
    FrameworkSpec,
    flashrljob_crd_manifest,
    flashrljob_openapi_schema,
)
from flashrl.platform.operator import (
    FlashRLOperator,
    render_child_resources,
    render_operator_resources,
)
from flashrl.platform.minikube_e2e import run_minikube_math_e2e

__all__ = [
    "FlashRLJob",
    "FlashRLJobSpec",
    "FlashRLJobStatus",
    "FrameworkSpec",
    "PlatformConfig",
    "load_flashrl_config",
    "build_flashrl_job",
    "flashrljob_crd_manifest",
    "flashrljob_openapi_schema",
    "render_child_resources",
    "render_operator_resources",
    "FlashRLOperator",
    "build_argument_parser",
    "run_minikube_math_e2e",
]
