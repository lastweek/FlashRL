"""Tests for distributed framework contracts, adapters, and service apps."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import flashrl.framework.serving.huggingface as serving_module
import flashrl.framework.training.huggingface as training_module
from flashrl.framework.config import GrpoConfig, RolloutConfig, ServingConfig, TrainingConfig
from flashrl.framework.data_models import Conversation, LearnerBatch, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.distributed.http import HttpServingClient
from flashrl.framework.distributed.local import (
    LocalLearnerClient,
    LocalRewardClient,
    LocalRolloutClient,
    LocalServingClient,
)
from flashrl.framework.distributed.models import (
    ActivateWeightVersionRequest,
    OptimizeStepRequest,
    RewardBatchRequest,
    RolloutBatchRequest,
    StatusResponse,
)
from flashrl.framework.rollout.base import build_rollout_generator
from flashrl.framework.services import (
    create_learner_app,
    create_reward_app,
    create_rollout_app,
    create_serving_app,
)
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

    learner_client = LocalLearnerClient(
        actor_backend,
        None,
        grpo_config=GrpoConfig(kl_coefficient=0.0),
        publish_dir=tmp_path,
        synchronize_serving=False,
    )
    learner_app = create_learner_app(learner_client)
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

    serving_client = LocalServingClient(serving_backend)
    serving_app = create_serving_app(serving_client)
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
    rollout_client = LocalRolloutClient(rollout_generator)
    reward_client = LocalRewardClient(
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

    rollout_http = TestClient(create_rollout_app(rollout_client))
    reward_http = TestClient(create_reward_app(reward_client))

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


def test_http_serving_client_parses_status_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP adapters should parse shared status payloads from JSON endpoints."""
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
        "flashrl.framework.distributed.http.urllib_request.urlopen",
        fake_urlopen,
    )

    client = HttpServingClient("http://serving.internal", timeout_seconds=9.0)
    response = client.status()
    assert captured == {
        "url": "http://serving.internal/v1/status",
        "timeout": 9.0,
    }
    assert response.status.name == "serving"
    assert response.status.ready_replica_count == 2
