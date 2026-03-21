"""Tests for distributed framework contracts, adapters, and service apps."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

import flashrl.framework.serving.huggingface as serving_module
import flashrl.framework.training.huggingface as training_module
from flashrl.framework.config import GrpoConfig, RolloutConfig, ServingConfig, TrainingConfig
from flashrl.framework.data_models import Conversation, LearnerBatch, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.distributed import (
    ActivateWeightVersionRequest,
    LearnerService,
    OptimizeStepRequest,
    RewardService,
    RolloutService,
    RewardBatchRequest,
    RolloutBatchRequest,
    ServingClient,
    ServingService,
    StatusResponse,
    create_learner_service_app,
    create_reward_service_app,
    create_rollout_service_app,
    create_serving_service_app,
)
from flashrl.framework.rollout.base import build_rollout_generator
from flashrl.framework.training import HuggingFaceTrainingBackend
from flashrl.framework.serving import HuggingFaceServingBackend
from tests.conftest import TinyActor


pytestmark = pytest.mark.unit


class BackendActor(TinyActor):
    """Tiny actor subclass used for service tests."""

    def __init__(self, config) -> None:
        self.config_snapshot = config
        super().__init__(bias_shift=0.1)


def _configure_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(serving_module, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training_module, "ActorModel", BackendActor)
    monkeypatch.setattr(serving_module, "ActorModel", BackendActor)


def _build_rollout_fn():
    def rollout_fn(prompts: list[Prompt], serving_backend) -> list[RolloutOutput]:
        samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
        return [
            RolloutOutput(
                text=sample.text,
                log_prob=sample.log_prob,
                prompt_token_ids=sample.prompt_token_ids,
                response_token_ids=sample.response_token_ids,
                response_token_logprobs=sample.response_token_logprobs,
                conversation=Conversation(
                    messages=[
                        Message(role="user", content=prompt.text),
                        Message(role="assistant", content=sample.text),
                    ]
                ),
                metadata=dict(sample.metadata),
            )
            for prompt, sample in zip(prompts, samples, strict=True)
        ]

    return rollout_fn


def _build_learner_batch() -> LearnerBatch:
    return LearnerBatch(
        prompt_token_ids=[[1, 2], [1, 2]],
        response_token_ids=[[3, 4], [3, 5]],
        response_token_logprobs=[[-0.1, -0.2], [-0.1, -0.3]],
        advantages=[0.5, -0.5],
        group_size=2,
        prompt_count=1,
        prompt_indices=[0, 0],
        candidate_indices=[0, 1],
    )


def test_learner_and_serving_service_apps_publish_and_activate_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Learner publish and serving activation should work through the new service contracts."""
    _configure_backends(monkeypatch)
    actor_backend = HuggingFaceTrainingBackend(
        TrainingConfig(model_name="fake/model", num_threads=1),
        learning_rate=1e-3,
    )
    serving_backend = HuggingFaceServingBackend(
        ServingConfig(model_name="fake/model", num_threads=1)
    )

    learner_service = LearnerService(
        actor_backend,
        None,
        grpo_config=GrpoConfig(kl_coefficient=0.0),
        publish_dir=tmp_path,
        synchronize_serving=False,
    )
    learner_app = create_learner_service_app(learner_service)
    learner_http = TestClient(learner_app)

    optimize_request = OptimizeStepRequest(
        step_id=1,
        epoch=1,
        learner_batch=_build_learner_batch(),
    )
    optimize_response = learner_http.post(
        "/v1/optimize-steps",
        json=optimize_request.model_dump(mode="json"),
    )
    assert optimize_response.status_code == 200
    published = optimize_response.json()["weight_version"]
    assert published["version_id"] == 1
    assert Path(published["artifact_uri"]).exists()

    serving_service = ServingService(serving_backend)
    serving_app = create_serving_service_app(serving_service)
    serving_http = TestClient(serving_app)

    activate_response = serving_http.post(
        "/v1/activate-weight-version",
        json=ActivateWeightVersionRequest(weight_version=published).model_dump(mode="json"),
    )
    assert activate_response.status_code == 200
    payload = activate_response.json()
    assert payload["converged"] is True
    assert payload["active_weight_version"]["version_id"] == 1
    assert serving_backend.current_weight_version().version_id == 1


def test_rollout_and_reward_service_apps_round_trip_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout and reward service apps should preserve batch payloads and weight versions."""
    _configure_backends(monkeypatch)
    serving_backend = HuggingFaceServingBackend(
        ServingConfig(model_name="fake/model", num_threads=1)
    )
    rollout_generator = build_rollout_generator(
        rollout_fn=_build_rollout_fn(),
        serving_backend=serving_backend,
        config=RolloutConfig(max_new_tokens=16),
    )
    rollout_service = RolloutService(rollout_generator)
    reward_service = RewardService(
        reward=type(
            "_Reward",
            (),
            {
                "compute_batch": staticmethod(
                    lambda rollouts: [RewardOutput(reward=float(len(item.text))) for item in rollouts]
                )
            },
        )()
    )

    rollout_http = TestClient(create_rollout_service_app(rollout_service))
    reward_http = TestClient(create_reward_service_app(reward_service))

    rollout_response = rollout_http.post(
        "/v1/rollout-batches",
        json=RolloutBatchRequest(
            step_id=3,
            prompts=[Prompt(text="one"), Prompt(text="two")],
            group_size=2,
        ).model_dump(mode="json"),
    )
    assert rollout_response.status_code == 200
    rollout_payload = rollout_response.json()
    assert len(rollout_payload["rollouts"]) == 4
    assert rollout_payload["weight_version"]["version_id"] == 0

    reward_response = reward_http.post(
        "/v1/reward-batches",
        json=RewardBatchRequest(rollouts=rollout_payload["rollouts"]).model_dump(mode="json"),
    )
    assert reward_response.status_code == 200
    reward_payload = reward_response.json()
    assert len(reward_payload["rewards"]) == 4
    assert reward_payload["metrics"]["sample_count"] == 4


def test_serving_client_parses_remote_status_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote clients should parse shared status payloads from JSON endpoints."""
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self) -> bytes:
            return json.dumps(
                StatusResponse(
                    status={
                        "name": "serving",
                        "phase": "Ready",
                        "healthy": True,
                        "ready_replica_count": 2,
                        "desired_replica_count": 2,
                    }
                ).model_dump(mode="json")
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        "flashrl.framework.distributed.client_common.urllib_request.urlopen",
        fake_urlopen,
    )

    client = ServingClient("http://serving.internal", timeout_seconds=9.0)
    response = client.status()
    assert captured == {
        "url": "http://serving.internal/v1/status",
        "timeout": 9.0,
    }
    assert response.status.name == "serving"
    assert response.status.ready_replica_count == 2


def test_rollout_service_reports_load_and_drains_during_inflight_requests() -> None:
    """Drain should flip readiness and expose load metadata while requests are still running."""
    request_started = threading.Event()
    release_request = threading.Event()

    class _BlockingRolloutGenerator:
        def __init__(self) -> None:
            self.config = RolloutConfig(max_new_tokens=8)

        def generate_grouped(self, prompts, group_size):
            request_started.set()
            release_request.wait(timeout=2.0)
            rollouts = [
                RolloutOutput(
                    text=f"reply-{index}",
                    log_prob=-0.1,
                    prompt_token_ids=[1],
                    response_token_ids=[2],
                    response_token_logprobs=[-0.1],
                    conversation=Conversation(
                        messages=[
                            Message(role="user", content=prompt.text),
                            Message(role="assistant", content=f"reply-{index}"),
                        ]
                    ),
                    metadata={"weight_version": {"version_id": 0, "origin": "startup", "model_source": "local://startup"}},
                )
                for index, prompt in enumerate(prompts)
                for _ in range(group_size)
            ]
            prompt_indices = [index for index, _prompt in enumerate(prompts) for _ in range(group_size)]
            candidate_indices = [candidate for _index, _prompt in enumerate(prompts) for candidate in range(group_size)]
            return prompts, rollouts, prompt_indices, candidate_indices

    app = create_rollout_service_app(RolloutService(_BlockingRolloutGenerator()))
    client = TestClient(app)

    response_holder: dict[str, object] = {}

    def _request_rollout() -> None:
        response_holder["response"] = client.post(
            "/v1/rollout-batches",
            json=RolloutBatchRequest(
                step_id=9,
                prompts=[Prompt(text="slow")],
                group_size=2,
            ).model_dump(mode="json"),
        )

    worker = threading.Thread(target=_request_rollout)
    worker.start()
    assert request_started.wait(timeout=1.0) is True

    status_payload = client.get("/v1/status").json()["status"]
    assert status_payload["metadata"]["inflightRequests"] == 1
    assert status_payload["metadata"]["queueDepth"] == 0
    assert "p95LatencySeconds" in status_payload["metadata"]

    drain_response = client.post("/v1/lifecycle/drain?wait_seconds=0.1")
    assert drain_response.status_code == 200
    assert drain_response.json()["status"] == "draining"

    ready = client.get("/readyz")
    assert ready.status_code == 503

    release_request.set()
    worker.join(timeout=2.0)
    response = response_holder["response"]
    assert response.status_code == 200

    drained_status = client.get("/v1/status").json()["status"]
    assert drained_status["metadata"]["inflightRequests"] == 0
    assert drained_status["metadata"]["draining"] is True


def test_distributed_package_reexports_services_clients_and_app_builders() -> None:
    """The distributed package should expose services, clients, and app builders directly."""
    from flashrl.framework.distributed import (
        LearnerService as ImportedLearnerService,
        ServingClient as ImportedServingClient,
        create_reward_service_app as imported_create_reward_service_app,
    )

    assert ImportedServingClient is ServingClient
    assert ImportedLearnerService is LearnerService
    assert imported_create_reward_service_app is create_reward_service_app


def test_distributed_package_no_longer_exports_protocols_or_legacy_names() -> None:
    """The legacy Local/Http/protocol surface should stay removed."""
    import flashrl.framework.distributed as distributed_module

    assert importlib.import_module("flashrl.framework.distributed.http_common") is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashrl.framework.distributed.protocols")
    for module_name in (
        "flashrl.framework.distributed.reward_server",
        "flashrl.framework.distributed.rollout_server",
        "flashrl.framework.distributed.learner_server",
        "flashrl.framework.distributed.serving_server",
        "flashrl.framework.distributed.server_common",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)

    for name in (
        "LocalRolloutClient",
        "LocalRewardClient",
        "LocalLearnerClient",
        "LocalServingClient",
        "HttpRolloutClient",
        "HttpRewardClient",
        "HttpLearnerClient",
        "HttpServingClient",
        "create_rollout_app",
        "create_reward_app",
        "create_learner_app",
        "create_serving_app",
    ):
        assert not hasattr(distributed_module, name), f"legacy export still present: {name}"


def test_repo_imports_use_consolidated_distributed_layout() -> None:
    """Repo Python imports should not reference the removed service or split client modules."""
    repo_root = Path(__file__).resolve().parents[1]
    legacy_paths = [
        ".".join(["flashrl", "framework", "services"]),
        ".".join(["flashrl", "framework", "distributed", "local"]),
    ]

    offenders: list[str] = []
    for base in (repo_root / "flashrl", repo_root / "tests"):
        for path in base.rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            content = path.read_text(encoding="utf-8")
            for legacy_path in legacy_paths:
                if legacy_path in content:
                    offenders.append(f"{path.relative_to(repo_root)}:{legacy_path}")

    assert offenders == []
