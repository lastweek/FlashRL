"""Subprocess entrypoint for built-in tool execution."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashrl.framework.data_models import Prompt, ToolResult


def resolve_import(import_string: str) -> Any:
    """Resolve one ``module:attribute`` import string."""
    module_name, separator, attr_path = import_string.partition(":")
    if separator == "" or not module_name or not attr_path:
        raise ValueError(
            "Tool entrypoints must use the format 'module.submodule:function'."
        )

    module = importlib.import_module(module_name)
    resolved = module
    for attr_name in attr_path.split("."):
        resolved = getattr(resolved, attr_name)
    return resolved


def _normalize_result(result: Any) -> ToolResult:
    """Normalize arbitrary tool returns into one ToolResult."""
    if isinstance(result, ToolResult):
        return result
    if isinstance(result, str):
        return ToolResult(content=result)
    return ToolResult(content=str(result))


def main() -> int:
    """Read one tool payload from stdin, execute it, and emit JSON."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        result = ToolResult(
            content=f"Invalid tool payload: {exc}",
            error=True,
            metadata={"status": "invalid_payload"},
        )
        print(result.model_dump_json())
        return 0

    try:
        tool_fn = resolve_import(str(payload["entrypoint"]))
        prompt = Prompt.model_validate(payload.get("prompt", {}))
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must deserialize to a JSON object.")
        result = _normalize_result(tool_fn(dict(arguments), prompt))
    except Exception as exc:
        result = ToolResult(
            content=f"Tool execution failed: {exc}",
            error=True,
            metadata={
                "status": "exception",
                "exception_type": type(exc).__name__,
            },
        )

    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
