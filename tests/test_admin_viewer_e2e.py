"""Browser E2E coverage for the unified static viewer."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING

import pytest

import flashrl.framework.flashrl as flashrl_module
from flashrl.framework.config import (
    GrpoConfig,
    LoggingConfig,
    MetricsConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)
from flashrl.framework.flashrl import FlashRL
from tests.test_admin import AdminServingBackend, StubTrainingBackend, build_rollout_fn, reward_fn

pytest.importorskip("playwright.sync_api")

if TYPE_CHECKING:
    from playwright.sync_api import Page


pytestmark = pytest.mark.integration


class _QuietStaticHandler(SimpleHTTPRequestHandler):
    """Silence noisy access logs from the viewer test's file server."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        del format, args


def _start_static_server(directory: Path) -> tuple[ThreadingHTTPServer, Thread, str]:
    handler = partial(_QuietStaticHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, base_url


def test_viewer_renders_live_runtime_and_run_history_workspaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    page: Page,
) -> None:
    """The unified viewer should render live admin data and expose run history UI."""
    monkeypatch.setattr(
        flashrl_module,
        "create_training_backend",
        lambda config, role, learning_rate=None: StubTrainingBackend(
            config,
            learning_rate=float(learning_rate or 1e-5),
        ),
    )
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None: AdminServingBackend(config),
    )
    trainer = FlashRL(
        actor_config=TrainingConfig(model_name="fake/model", device="cpu"),
        serving_config=ServingConfig(model_name="fake/model", backend="vllm"),
        trainer_config=TrainerConfig(batch_size=2, max_epochs=1),
        grpo_config=GrpoConfig(group_size=2),
        rollout_fn=build_rollout_fn,
        reward_fn=reward_fn,
        logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
        metrics_config=MetricsConfig(enabled=False),
    )
    viewer_server, viewer_thread, viewer_base_url = _start_static_server(Path("docs").resolve())

    try:
        assert trainer.admin_base_url is not None
        page.goto(f"{viewer_base_url}/viewer.html")
        page.locator("#workspace-live-button").click()
        if page.locator("#auto-refresh").is_checked():
            page.locator("#auto-refresh").uncheck()

        page.locator("#base-url").fill(trainer.admin_base_url)
        page.locator("#refresh-button").click()
        page.wait_for_function(
            "() => document.getElementById('status-pill').textContent.includes('Loaded 4 objects')"
        )

        assert page.locator("#object-count").inner_text() == "4 objects"
        assert page.locator("#instance-count").inner_text() == "1 instance"
        assert "vllm-instance-0" in page.locator("#vllm-body").inner_text()
        assert "healthy" in page.locator("#vllm-body").inner_text()

        page.locator(".object-item", has_text="VLLMInstance/vllm-instance-0").click()
        page.wait_for_function(
            "() => document.getElementById('json-output').textContent.includes('\"kind\": \"VLLMInstance\"')"
        )
        assert '"healthy": true' in page.locator("#json-output").inner_text()

        page.locator("#workspace-runs-button").click()
        assert page.locator("#open-root-button").is_visible()
        assert page.locator("#run-search").is_visible()
        assert page.locator("#empty-state").is_visible()
        assert "Chrome and Edge" in page.locator("#empty-state").inner_text()
        assert "Open run folder" in page.locator("#workspace-runs").inner_text()

        page.locator("#workspace-live-button").click()
        page.locator("#base-url").fill("http://127.0.0.1:1")
        page.locator("#refresh-button").click()
        page.wait_for_function(
            "() => document.getElementById('status-pill').textContent.startsWith('Failed:')"
        )
        assert "Unable to reach admin API." in page.locator("#vllm-body").inner_text()
    finally:
        trainer.close()
        viewer_server.shutdown()
        viewer_server.server_close()
        viewer_thread.join(timeout=2.0)
