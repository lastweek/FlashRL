"""Thin CLI entrypoint for the reasoning-code example."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = Path(__file__).resolve().parent / "workflow.py"
_original_name = __name__
try:
    globals()["__name__"] = f"{_original_name}._workflow"
    exec(compile(WORKFLOW_PATH.read_text(encoding="utf-8"), str(WORKFLOW_PATH), "exec"), globals())
finally:
    globals()["__name__"] = _original_name


if __name__ == "__main__":
    raise SystemExit(main())
