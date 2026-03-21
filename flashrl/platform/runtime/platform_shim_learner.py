"""Platform shim for the learner pod."""

from __future__ import annotations

from flashrl.framework.training import LearnerService, create_learner_service_app, create_training_backend
from flashrl.platform.runtime.platform_pod_logging import PlatformPodLogger
from flashrl.platform.runtime.platform_shim_base import PlatformShim
from flashrl.platform.runtime.platform_shim_common import (
    component_log_metadata,
    load_mounted_job,
    storage_path_from_uri,
)


def _attach_platform_observability(service, *, job, event_logger):
    try:
        setattr(service, "_status_metadata", component_log_metadata(job, "learner"))
        setattr(service, "_event_logger", event_logger)
    except Exception:
        pass
    return service


class PlatformShimLearner(PlatformShim):
    """Load the mounted job, then wire training backends into the framework service."""

    def create_app(self):
        job = load_mounted_job(self._job_path)
        self._pod_logger = PlatformPodLogger(job=job, component="learner")
        self._pod_logger.configure_python_logging()
        self._pod_logger.emit("shim_startup", message="Bootstrapping learner pod.")
        actor_backend = create_training_backend(
            job.spec.framework.actor.model_copy(deep=True),
            role="actor",
            learning_rate=job.spec.framework.controller.learning_rate,
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
        learner_service = _attach_platform_observability(
            learner_service,
            job=job,
            event_logger=self._pod_logger.emit,
        )
        return create_learner_service_app(learner_service)
