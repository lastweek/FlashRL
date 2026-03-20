"""Component pod app factories for the platform runtime."""

from __future__ import annotations

from flashrl.framework import runtime_support
from flashrl.framework.distributed import (
    HttpServingClient,
    LocalLearnerClient,
    LocalRewardClient,
    LocalRolloutClient,
    LocalServingClient,
    create_learner_app,
    create_reward_app,
    create_rollout_app,
    create_serving_app,
)
from flashrl.framework.distributed.remote_serving_backend import HttpServingBackend
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.base import build_rollout_generator
from flashrl.framework.serving import create_serving_backend
from flashrl.framework.training import create_training_backend
from flashrl.platform.k8s.job import FlashRLJob
from flashrl.platform.runtime.common import build_rollout_config, service_url, shared_path


def create_rollout_component_app(job: FlashRLJob):
    """Build the rollout FastAPI app for one platform job."""
    rollout_impl = runtime_support.instantiate_hook(job.spec.userCode.rollout)
    serving_backend = HttpServingBackend(
        config=job.spec.framework.serving.model_copy(deep=True),
        client=HttpServingClient(service_url("serving")),
    )
    rollout_generator = build_rollout_generator(
        rollout_fn=rollout_impl,
        serving_backend=serving_backend,
        config=build_rollout_config(job),
    )
    return create_rollout_app(LocalRolloutClient(rollout_generator))


def create_reward_component_app(job: FlashRLJob):
    """Build the reward FastAPI app for one platform job."""
    reward_impl = runtime_support.instantiate_hook(job.spec.userCode.reward)
    reward = UserDefinedReward(reward_fn=reward_impl, config=job.spec.framework.grpo)
    return create_reward_app(LocalRewardClient(reward))


def create_learner_component_app(job: FlashRLJob):
    """Build the learner FastAPI app for one platform job."""
    actor_backend = create_training_backend(
        job.spec.framework.actor.model_copy(deep=True),
        role="actor",
        learning_rate=job.spec.framework.trainer.learning_rate,
    )
    reference_backend = (
        create_training_backend(
            job.spec.framework.reference.model_copy(deep=True),
            role="reference",
        )
        if job.spec.framework.reference is not None
        else None
    )
    publish_dir = shared_path(job.spec.storage.weights.uriPrefix, purpose="weights")
    publish_dir.mkdir(parents=True, exist_ok=True)
    client = LocalLearnerClient(
        actor_backend,
        reference_backend,
        grpo_config=job.spec.framework.grpo,
        publish_dir=publish_dir,
        synchronize_serving=False,
    )
    return create_learner_app(client)


def create_serving_component_app(job: FlashRLJob):
    """Build the serving FastAPI app for one platform job."""
    serving_backend = create_serving_backend(
        job.spec.framework.serving.model_copy(deep=True),
        log_dir=shared_path(job.spec.storage.weights.uriPrefix, purpose="serving-cache"),
    )
    return create_serving_app(LocalServingClient(serving_backend))
