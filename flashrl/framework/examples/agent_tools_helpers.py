"""Importable tool entrypoints for the agent-tools example."""

from __future__ import annotations

from flashrl.framework.data_models import Prompt


def add_tool(arguments: dict[str, int], prompt: Prompt) -> str:
    """Return the sum of two integers."""
    del prompt
    return str(int(arguments["a"]) + int(arguments["b"]))


def multiply_tool(arguments: dict[str, int], prompt: Prompt) -> str:
    """Return the product of two integers."""
    del prompt
    return str(int(arguments["a"]) * int(arguments["b"]))


def lookup_note_tool(arguments: dict[str, str], prompt: Prompt) -> str:
    """Return one canned note body by id."""
    del prompt
    notes = {
        "alpha": "Alpha has the strongest reliability score in the comparison set.",
        "beta": "Beta is cheaper to operate, but it trails Alpha on reliability.",
        "gamma": "Gamma is fastest, but it uses the most energy under load.",
    }
    note_id = str(arguments.get("note_id", "")).strip().lower()
    if note_id not in notes:
        raise ValueError(f"Unknown note_id: {note_id}")
    return notes[note_id]
