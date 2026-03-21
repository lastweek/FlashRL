"""Platform shim for the learner pod."""

from __future__ import annotations

from flashrl.framework.training import LearnerService, create_learner_service_app, create_training_backend
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import load_mounted_job, storage_path_from_uri


class PlatformShimLearner(PlatformShim):
    """Load the mounted job, then wire training backends into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
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
