"""Compatibility shim for the relocated agent tool runtime."""

from flashrl.framework.agent.tools import SubprocessToolRuntime, Tool

__all__ = ["Tool", "SubprocessToolRuntime"]
