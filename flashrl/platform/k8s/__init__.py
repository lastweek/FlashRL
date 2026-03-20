"""Kubernetes-facing platform helpers."""

from flashrl.platform.k8s.operator import FINALIZER, FlashRLOperator
from flashrl.platform.k8s.renderer import render_child_resources

__all__ = ["FINALIZER", "FlashRLOperator", "render_child_resources"]
