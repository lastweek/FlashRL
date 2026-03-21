"""Tests for runtime memory snapshot helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flashrl.framework.memory as memory_module


pytestmark = pytest.mark.unit


def test_capture_memory_snapshot_uses_psutil_process_and_system(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU snapshots should include process RSS and system memory counters."""

    class _FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=123456)

    monkeypatch.setattr(memory_module.psutil, "Process", lambda: _FakeProcess())
    monkeypatch.setattr(
        memory_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=999999, available=111111),
    )

    snapshot = memory_module.capture_memory_snapshot("cpu")

    assert snapshot["device_type"] == "cpu"
    assert snapshot["process"]["rss_bytes"] == 123456
    assert snapshot["system"]["total_bytes"] == 999999
    assert snapshot["system"]["available_bytes"] == 111111


def test_capture_memory_snapshot_reads_mps_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """MPS snapshots should include backend-native device counters when present."""
    monkeypatch.setattr(
        memory_module.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)),
    )
    monkeypatch.setattr(
        memory_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=200, available=50),
    )
    monkeypatch.setattr(
        memory_module.torch,
        "mps",
        SimpleNamespace(
            current_allocated_memory=lambda: 300,
            driver_allocated_memory=lambda: 400,
            recommended_max_memory=lambda: 500,
        ),
        raising=False,
    )

    snapshot = memory_module.capture_memory_snapshot("mps")

    assert snapshot["device_type"] == "mps"
    assert snapshot["device"]["current_allocated_bytes"] == 300
    assert snapshot["device"]["driver_allocated_bytes"] == 400
    assert snapshot["device"]["recommended_max_bytes"] == 500


def test_capture_memory_snapshot_gracefully_handles_missing_backend_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable backend counters should not make memory capture fail."""
    monkeypatch.setattr(
        memory_module.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)),
    )
    monkeypatch.setattr(
        memory_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=200, available=50),
    )
    monkeypatch.setattr(
        memory_module.torch,
        "mps",
        SimpleNamespace(
            current_allocated_memory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            driver_allocated_memory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            recommended_max_memory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
        raising=False,
    )

    snapshot = memory_module.capture_memory_snapshot("mps")

    assert snapshot["device_type"] == "mps"
    assert snapshot["device"] == {}
