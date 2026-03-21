"""Unit tests for device and thread utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flashrl.framework.models.device as device_module

pytestmark = pytest.mark.unit


def reset_device_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level device and thread globals between tests."""
    monkeypatch.setattr(device_module, "DEFAULT_DEVICE", None)
    monkeypatch.setattr(device_module, "_INTEROP_THREADS_SET", False)


def test_get_device_prefers_explicit_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit device should override autodetection."""
    reset_device_state(monkeypatch)
    assert device_module.get_device("cpu").type == "cpu"


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "cpu"),
        (False, False, "cpu"),
    ],
)
def test_get_device_autodetect_precedence(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    """Autodetection should prefer cuda, then cpu, with MPS as explicit opt-in."""
    reset_device_state(monkeypatch)
    monkeypatch.setattr(
        device_module.torch,
        "cuda",
        SimpleNamespace(is_available=lambda: cuda_available),
    )
    monkeypatch.setattr(
        device_module.torch,
        "backends",
        SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        raising=False,
    )

    assert device_module.get_device().type == expected


def test_get_device_supports_explicit_mps_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit MPS selection should remain available even though autodetect prefers CPU."""
    reset_device_state(monkeypatch)
    assert device_module.get_device("mps").type == "mps"


def test_set_num_threads_updates_env_and_sets_interop_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interop threads should only be set on the first call for the process state."""
    reset_device_state(monkeypatch)
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        device_module.torch,
        "set_num_threads",
        lambda value: calls.append(("intra", value)),
    )
    monkeypatch.setattr(
        device_module.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )

    device_module.set_num_threads(6)
    device_module.set_num_threads(4)

    assert calls == [("intra", 6), ("interop", 3), ("intra", 4)]
    assert device_module._INTEROP_THREADS_SET is True
    assert device_module.os.environ["OMP_NUM_THREADS"] == "4"
    assert device_module.os.environ["MKL_NUM_THREADS"] == "4"
