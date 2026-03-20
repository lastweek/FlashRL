"""Top-level CLI entrypoint for FlashRL."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch top-level subcommands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: flashrl <platform|component> <subcommand> [...]", file=sys.stderr)
        return 2
    if args[0] == "platform":
        from flashrl.platform.cli import main as platform_main

        return int(platform_main(["platform", *args[1:]]))
    if args[0] == "component":
        from flashrl.platform.component_cli import main as component_main

        return int(component_main(args[1:]))
    print(f"Unknown command: {args[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
