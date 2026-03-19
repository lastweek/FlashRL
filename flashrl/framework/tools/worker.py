"""Compatibility shim for the relocated agent tool worker."""

from flashrl.framework.agent.tools.worker import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
