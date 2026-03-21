"""Explicit learner pod bootstrap for FlashRL platform mode."""

from __future__ import annotations

import uvicorn

from flashrl.framework.distributed import LearnerService, create_learner_service_app
from flashrl.framework.training import create_training_backend
from flashrl.platform.runtime.pod import load_mounted_job, storage_path_from_uri


def create_learner_pod_app():
    """Build the learner service app from training backends plus shared storage."""
    job = load_mounted_job()
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
    published_weights_dir = storage_path_from_uri(job.spec.storage.weights.uriPrefix, purpose="weights")
    published_weights_dir.mkdir(parents=True, exist_ok=True)
    learner_service = LearnerService(
        actor_backend,
        reference_backend,
        grpo_config=job.spec.framework.grpo,
        publish_dir=published_weights_dir,
        synchronize_serving=False,
    )
    return create_learner_service_app(learner_service)


def run_learner_pod(*, host: str = "0.0.0.0", port: int = 8000) -> int:
    """Run the learner pod HTTP server."""
    uvicorn.run(create_learner_pod_app(), host=host, port=port, log_level="info")
    return 0
