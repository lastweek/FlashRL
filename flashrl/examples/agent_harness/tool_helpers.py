"""Repo-oriented tool helpers used by the reference agent harness example."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from flashrl.framework.data_models import Prompt


def _repo_root(prompt: Prompt) -> Path:
    repo_root = prompt.metadata.get("repo_root")
    if not isinstance(repo_root, str) or not repo_root.strip():
        raise ValueError("Prompt metadata must include a non-empty repo_root.")
    path = Path(repo_root).resolve()
    if not path.exists():
        raise ValueError(f"Repo root does not exist: {path}")
    return path


def list_repo_files(arguments: dict[str, Any], prompt: Prompt) -> str:
    root = _repo_root(prompt)
    max_files = int(arguments.get("max_files", 200))
    files = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )
    return "\n".join(files[:max_files])


def read_repo_file(arguments: dict[str, Any], prompt: Prompt) -> str:
    root = _repo_root(prompt)
    relative_path = str(arguments.get("path", "")).strip()
    if not relative_path:
        raise ValueError("read_repo_file requires a `path` argument.")
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Requested path escapes the repo root.")
    if not target.exists() or not target.is_file():
        raise ValueError(f"File not found: {relative_path}")
    text = target.read_text(encoding="utf-8")
    max_chars = int(arguments.get("max_chars", 4000))
    return text[:max_chars]


def search_repo_text(arguments: dict[str, Any], prompt: Prompt) -> str:
    root = _repo_root(prompt)
    needle = str(arguments.get("query", "")).strip()
    if not needle:
        raise ValueError("search_repo_text requires a non-empty `query` argument.")
    max_matches = int(arguments.get("max_matches", 20))
    matches: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_index, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                matches.append(f"{path.relative_to(root)}:{line_index}: {line.strip()}")
                if len(matches) >= max_matches:
                    return "\n".join(matches)
    return "\n".join(matches)


def run_repo_shell(arguments: dict[str, Any], prompt: Prompt) -> str:
    root = _repo_root(prompt)
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError("run_repo_shell requires a non-empty `argv` list.")
    argv = [str(item) for item in argv]
    allowed = {"ls", "pwd", "cat", "head", "tail", "wc"}
    if argv[0] not in allowed:
        raise ValueError(f"Shell command not allowed: {argv[0]}")
    completed = subprocess.run(
        argv,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise ValueError(stderr or f"Command failed with code {completed.returncode}")
    return output
