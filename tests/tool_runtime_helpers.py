"""Lightweight tool helpers imported by subprocess runtime tests."""

from __future__ import annotations

import time

from flashrl.framework.data_models import Prompt


def echo_tool(arguments: dict[str, object], prompt: Prompt) -> str:
    """Return one echoed argument for subprocess runtime tests."""
    del prompt
    return str(arguments["text"])


def slow_tool(arguments: dict[str, object], prompt: Prompt) -> str:
    """Sleep briefly and return one tagged result."""
    del prompt
    time.sleep(float(arguments.get("delay", 0.05)))
    return str(arguments["text"])


def failing_tool(arguments: dict[str, object], prompt: Prompt) -> str:
    """Raise a deterministic tool failure."""
    del arguments, prompt
    raise RuntimeError("boom")
