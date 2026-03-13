"""Tests for the live admin API and viewer."""

from __future__ import annotations

import json
from pathlib import Path
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest
import torch
from fastapi.testclient import TestClient

import flashrl.framework.flashrl as flashrl_module
from flashrl.framework.admin import AdminRegistry, build_admin_object, create_admin_app
from flashrl.framework.config import LoggingConfig, MetricsConfig, RuntimeConfig, ServingConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.flashrl import FlashRL
from tests.conftest import TinyActor


pytestmark = pytest.mark.unit


class StubTrainingBackend:
    """Training backend stub for admin API tests."""

    def __init__(self, config, learning_rate: float = 1e-5) -> None:
        del config
        self.actor = TinyActor(bias_shift=0.25)
        self.actor.train()
        self.optimizer = torch.optim.SGD(self.actor.model.parameters(), lr=learning_rate)

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.actor.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.actor.model.load_state_dict(torch.load(path, weights_only=False))

    def sync_weights_to(self, serving_backend) -> None:
        serving_backend.sync_from_training_actor(self.actor)


class AdminServingBackend:
    """Serving backend stub that exposes one managed vLLM object."""

    def __init__(self, config) -> None:
        self.config = config
        self._actor = TinyActor(bias_shift=0.1)
        self.device = self._actor.device
        self.generation_defaults: dict[str, object] = {}
        self.closed = False

    def generate(self, prompts: list[str], **kwargs):
        return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)
        self._actor.set_generation_defaults(**kwargs)

    def sync_from_training_actor(self, training_actor) -> None:
        self._actor.model.load_state_dict(training_actor.model.state_dict())

    def set_live_rollout_debug(self, callback, context) -> None:
        del callback, context

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        del candidate_index

    def clear_live_rollout_debug(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def list_admin_objects(self) -> list[dict[str, object]]:
        return [
            build_admin_object(
                "VLLMInstance",
                "vllm-instance-0",
                uid="test-runtime:vllm:0",
                created_at="2026-03-13T00:00:00Z",
                labels={"flashrl.dev/serving-backend": "vllm"},
                spec={
                    "replicaIndex": 0,
                    "host": "127.0.0.1",
                    "port": 8100,
                    "modelSource": "fake/model",
                    "servedModelName": self.config.model_name,
                    "pythonExecutable": "/tmp/fake-python",
                    "command": ["/tmp/fake-vllm", "serve", "fake/model"],
                },
                status={
                    "phase": "Closed" if self.closed else "Ready",
                    "pid": 4321,
                    "healthy": not self.closed,
                    "startedAt": "2026-03-13T00:00:00Z",
                    "readyAt": "2026-03-13T00:00:01Z",
                    "exitCode": 0 if self.closed else None,
                    "stderrTail": "",
                    "lastError": None,
                },
            )
        ]


def build_rollout_fn(prompts: list[Prompt], serving_backend) -> list[RolloutOutput]:
    """Construct one rollout per prompt."""
    samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
    return [
        RolloutOutput(
            text=sample.text,
            log_prob=sample.log_prob,
            prompt_token_ids=sample.prompt_token_ids,
            response_token_ids=sample.response_token_ids,
            response_token_logprobs=sample.response_token_logprobs,
            metadata=dict(sample.metadata),
            conversation=Conversation(
                messages=[
                    Message(role="user", content=prompt.text),
                    Message(role="assistant", content=sample.text),
                ]
            ),
        )
        for prompt, sample in zip(prompts, samples, strict=True)
    ]


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Return a deterministic reward."""
    return RewardOutput(reward=float(len(rollout.text)))


def failing_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Raise to keep the admin API alive in the failed state."""
    raise ValueError(f"reward failure for {rollout.text}")


def _get_json(url: str) -> dict[str, object]:
    with urllib_request.urlopen(url, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_json(url: str) -> dict[str, object]:
    last_error: Exception | None = None
    for _ in range(20):
        try:
            return _get_json(url)
        except Exception as exc:  # pragma: no cover - retry helper
            last_error = exc
            time.sleep(0.05)
    assert last_error is not None
    raise last_error


def _build_test_registry() -> AdminRegistry:
    registry = AdminRegistry()
    registry.register(
        lambda: [
            build_admin_object(
                "FlashRLRuntime",
                "flashrl-runtime",
                uid="runtime:flashrl-runtime",
                created_at="2026-03-13T00:00:00Z",
                labels={"flashrl.dev/runtime": "flashrl"},
                spec={
                    "modelName": "fake/model",
                    "referenceEnabled": False,
                    "adminBaseUrl": "http://127.0.0.1:9999",
                },
                status={
                    "phase": "Ready",
                    "startedAt": "2026-03-13T00:00:00Z",
                    "currentEpoch": 0,
                    "currentStep": 0,
                    "lastError": None,
                },
            ),
            build_admin_object(
                "VLLMInstance",
                "vllm-instance-0",
                uid="runtime:vllm:0",
                created_at="2026-03-13T00:00:00Z",
                labels={"flashrl.dev/serving-backend": "vllm"},
                spec={
                    "replicaIndex": 0,
                    "host": "127.0.0.1",
                    "port": 8100,
                    "modelSource": "fake/model",
                    "servedModelName": "fake/model",
                    "pythonExecutable": "/tmp/fake-python",
                    "command": ["/tmp/fake-vllm", "serve", "fake/model"],
                },
                status={
                    "phase": "Ready",
                    "pid": 4321,
                    "healthy": True,
                    "startedAt": "2026-03-13T00:00:00Z",
                    "readyAt": "2026-03-13T00:00:01Z",
                    "exitCode": None,
                    "stderrTail": "",
                    "lastError": None,
                },
            ),
        ]
    )
    return registry


def test_runtime_config_parses_admin_fields() -> None:
    """Runtime config should parse the admin endpoint fields."""
    config = RuntimeConfig(
        admin_enabled=False,
        admin_host="127.0.0.2",
        admin_port=12345,
    )

    assert config.admin_enabled is False
    assert config.admin_host == "127.0.0.2"
    assert config.admin_port == 12345


def test_create_admin_app_exposes_expected_routes_and_shapes() -> None:
    """FastAPI admin app should preserve the current route contract."""
    client = TestClient(create_admin_app(_build_test_registry()))

    health = client.get("/admin/healthz")
    objects = client.get("/admin/v1/objects")
    runtime = client.get("/admin/v1/objects/FlashRLRuntime/flashrl-runtime")
    vllm = client.get("/admin/v1/objects/VLLMInstance")
    missing = client.get("/admin/v1/objects/VLLMInstance/missing")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert objects.status_code == 200
    assert objects.json()["kind"] == "ObjectList"
    assert {item["kind"] for item in objects.json()["items"]} == {
        "FlashRLRuntime",
        "VLLMInstance",
    }
    assert runtime.status_code == 200
    assert runtime.json()["metadata"]["name"] == "flashrl-runtime"
    assert vllm.status_code == 200
    assert vllm.json()["items"][0]["spec"]["port"] == 8100
    assert missing.status_code == 404
    assert missing.json() == {"detail": "not_found"}


def test_create_admin_app_disables_docs_and_enables_cors() -> None:
    """FastAPI admin app should stay minimal and browser-fetch friendly."""
    client = TestClient(create_admin_app(_build_test_registry()))

    docs = client.get("/docs")
    openapi = client.get("/openapi.json")
    preflight = client.options(
        "/admin/v1/objects",
        headers={
            "Origin": "http://viewer.local",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert docs.status_code == 404
    assert openapi.status_code == 404
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert "GET" in preflight.headers["access-control-allow-methods"]


def test_flashrl_admin_server_exposes_runtime_backend_and_vllm_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FlashRL should expose live runtime and VLLM admin objects over HTTP."""
    monkeypatch.setattr(flashrl_module, "TrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None: AdminServingBackend(config),
    )

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=build_rollout_fn,
        reward_fn=reward_fn,
        serving_config=ServingConfig(model_name="fake/model", backend="vllm"),
        logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
        metrics_config=MetricsConfig(enabled=False),
    )
    assert trainer.admin_base_url is not None

    try:
        health = _wait_for_json(f"{trainer.admin_base_url}/admin/healthz")
        object_list = _wait_for_json(f"{trainer.admin_base_url}/admin/v1/objects")
        runtime_object = _wait_for_json(
            f"{trainer.admin_base_url}/admin/v1/objects/FlashRLRuntime/flashrl-runtime"
        )
        vllm_list = _wait_for_json(f"{trainer.admin_base_url}/admin/v1/objects/VLLMInstance")
    finally:
        trainer.close()

    kinds = {item["kind"] for item in object_list["items"]}
    assert health == {"status": "ok"}
    assert "FlashRLRuntime" in kinds
    assert "TrainingBackend" in kinds
    assert "ServingBackend" in kinds
    assert "VLLMInstance" in kinds
    assert runtime_object["spec"]["adminBaseUrl"] == trainer.admin_base_url
    assert vllm_list["items"][0]["spec"]["port"] == 8100
    assert vllm_list["items"][0]["status"]["phase"] == "Ready"

    with pytest.raises(urllib_error.URLError):
        _get_json(f"{trainer.admin_base_url}/admin/healthz")


def test_flashrl_admin_server_keeps_failed_state_visible_until_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Training failures should remain visible through the admin API until close."""
    monkeypatch.setattr(flashrl_module, "TrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None: AdminServingBackend(config),
    )

    trainer = FlashRL(
        model="fake/model",
        rollout_fn=build_rollout_fn,
        reward_fn=failing_reward_fn,
        batch_size=2,
        max_epochs=1,
        serving_config=ServingConfig(model_name="fake/model", backend="vllm"),
        logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
        metrics_config=MetricsConfig(enabled=False),
    )
    assert trainer.admin_base_url is not None

    with pytest.raises(ValueError, match="reward failure"):
        trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    runtime_object = _wait_for_json(
        f"{trainer.admin_base_url}/admin/v1/objects/FlashRLRuntime/flashrl-runtime"
    )
    vllm_list = _wait_for_json(f"{trainer.admin_base_url}/admin/v1/objects/VLLMInstance")

    assert runtime_object["status"]["phase"] == "Failed"
    assert "reward failure" in runtime_object["status"]["lastError"]
    assert vllm_list["items"][0]["status"]["phase"] == "Closed"

    trainer.close()


def test_unified_viewer_exists_and_contains_live_runtime_sections() -> None:
    """The unified viewer should expose the live runtime workspace."""
    viewer_path = Path("docs/viewer.html")
    html = viewer_path.read_text(encoding="utf-8")

    assert viewer_path.exists()
    assert "FlashRL Viewer" in html
    assert "Live Runtime" in html
    assert "Run History" in html
    assert "/admin/v1/objects" in html
    assert "VLLMInstance" in html
    assert "localStorage" in html
    assert "status-pill" in html
    assert "object-list" in html
    assert "vllm-body" in html
