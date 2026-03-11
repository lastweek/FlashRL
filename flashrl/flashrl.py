"""FlashRL: Unified RL training API."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Callable

from flashrl.framework.backends.serving import ServingBackend
from flashrl.framework.backends.training import TrainingBackend
from flashrl.framework.config import (
    LoggingConfig,
    ModelConfig,
    RewardConfig,
    RolloutConfig,
    TrainerConfig,
)
from flashrl.framework.data_models import (
    Prompt,
    RewardOutput,
    RolloutOutput,
)
from flashrl.framework.models.reference import ReferenceModel
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.run_logger import RunLogger
from flashrl.framework.trainer.grpo import GRPOTrainer

if TYPE_CHECKING:
    from flashrl.framework.models.actor import ActorModel


class FlashRL:
    """Unified FlashRL trainer with a simple RL training API."""

    def __init__(
        self,
        model: str,
        rollout_fn: Callable[[list[Prompt], "ActorModel"], list[RolloutOutput]],
        reward_fn: Callable[[RolloutOutput], RewardOutput],
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        max_epochs: int = 10,
        kl_coefficient: float = 0.0,
        device: str | None = None,
        max_length: int = 2048,
        num_threads: int = 1,
        logging_config: LoggingConfig | None = None,
        reference_enabled: bool = False,
        reference_device: str | None = None,
    ) -> None:
        """Initialize FlashRL trainer.

        The local path keeps separate training and serving model copies, each on
        exactly one device. A frozen reference model is optional and is only
        needed when the user wants KL-regularized GRPO.
        """
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.reference_enabled = reference_enabled
        self.reference_device = reference_device

        self.model_config = ModelConfig(
            model_name=model,
            device=device,
            max_length=max_length,
            num_threads=num_threads,
        )
        self.trainer_config = TrainerConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            kl_coefficient=kl_coefficient,
        )
        self.rollout_config = RolloutConfig()
        self.reward_config = RewardConfig()
        self.logging_config = logging_config or LoggingConfig()

        self._training_backend: TrainingBackend | None = None
        self._serving_backend: ServingBackend | None = None
        self._reference: ReferenceModel | None = None
        self._rollout_generator: UserDefinedRollout | None = None
        self._reward: UserDefinedReward | None = None
        self._trainer: GRPOTrainer | None = None
        self._run_logger: RunLogger | None = None
        self._run_lifecycle_totals: dict[str, float] = {}
        self._runtime_bootstrap_events: list[dict[str, Any]] = []
        self._runtime_bootstrap_totals: dict[str, float] = {}
        self._resume_from_checkpoint = False

        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        """Build the one-time runtime object graph eagerly."""
        startup_started_at = time.perf_counter()
        bootstrap_events: list[dict[str, Any]] = [
            {
                "component": "training_backend",
                "status": "started",
                "cpu_threads": self.model_config.num_threads,
            }
        ]
        bootstrap_totals: dict[str, float] = {}

        training_started_at = time.perf_counter()
        self._training_backend = TrainingBackend(
            self.model_config,
            learning_rate=self.trainer_config.learning_rate,
        )
        training_duration = time.perf_counter() - training_started_at
        bootstrap_totals["startup_training_backend_seconds"] = training_duration
        bootstrap_events.append(
            {
                "component": "training_backend",
                "status": "completed",
                "device": str(self._training_backend.actor.device),
                "cpu_threads": self.model_config.num_threads,
                "duration_seconds": training_duration,
            }
        )
        bootstrap_events.append(
            {
                "component": "serving_backend",
                "status": "started",
                "cpu_threads": self.model_config.num_threads,
            }
        )

        serving_started_at = time.perf_counter()
        self._serving_backend = ServingBackend(self.model_config)
        serving_duration = time.perf_counter() - serving_started_at
        bootstrap_totals["startup_serving_backend_seconds"] = serving_duration
        bootstrap_events.append(
            {
                "component": "serving_backend",
                "status": "completed",
                "device": str(self._serving_backend.actor.device),
                "cpu_threads": self.model_config.num_threads,
                "duration_seconds": serving_duration,
            }
        )

        if self.reference_enabled:
            reference_config = self.model_config.model_copy(
                update={
                    "device": self.reference_device or self.model_config.device,
                }
            )
            bootstrap_events.append(
                {
                    "component": "reference_model",
                    "status": "started",
                    "cpu_threads": reference_config.num_threads,
                }
            )
            reference_started_at = time.perf_counter()
            self._reference = ReferenceModel(reference_config)
            reference_duration = time.perf_counter() - reference_started_at
            bootstrap_totals["startup_reference_model_seconds"] = reference_duration
            bootstrap_events.append(
                {
                    "component": "reference_model",
                    "status": "completed",
                    "device": str(self._reference.device),
                    "cpu_threads": reference_config.num_threads,
                    "duration_seconds": reference_duration,
                }
            )
        else:
            self._reference = None

        bootstrap_totals["startup_total_seconds"] = time.perf_counter() - startup_started_at
        self._runtime_bootstrap_events = bootstrap_events
        self._runtime_bootstrap_totals = bootstrap_totals

        assert self._serving_backend is not None
        assert self._training_backend is not None

        self._rollout_generator = UserDefinedRollout(
            rollout_fn=self.rollout_fn,
            actor=self._serving_backend.actor,
            config=self.rollout_config,
        )
        self._reward = UserDefinedReward(
            reward_fn=self.reward_fn,
            config=self.reward_config,
        )
        self._trainer = GRPOTrainer(
            config=self.trainer_config,
            training_backend=self._training_backend,
            serving_backend=self._serving_backend,
            reference=self._reference,
            reward_fn=self._reward,
            rollout_generator=self._rollout_generator,
            run_logger=None,
        )

    def _replay_runtime_bootstrap_logs(self) -> None:
        """Replay cached runtime bootstrap events into the current run logger."""
        if self._run_logger is None:
            return
        for event in self._runtime_bootstrap_events:
            self._run_logger.log_model_load(
                event["component"],
                event["status"],
                {
                    key: value
                    for key, value in event.items()
                    if key not in {"component", "status"}
                },
            )

    def train(self, dataset: list[Prompt]) -> None:
        """Train on dataset."""
        if self._run_logger is not None:
            self._run_logger.close()

        self._run_lifecycle_totals = dict(self._runtime_bootstrap_totals)
        total_batches = (
            math.ceil(len(dataset) / self.trainer_config.batch_size) * self.trainer_config.max_epochs
            if dataset
            else 0
        )
        self._run_logger = RunLogger(
            self.logging_config,
            model_name=self.model_config.model_name,
        )
        self._run_logger.start_run(
            dataset_size=len(dataset),
            batch_size=self.trainer_config.batch_size,
            max_epochs=self.trainer_config.max_epochs,
            total_batches=total_batches,
            device=self.model_config.device or "auto",
            dtype=self.model_config.dtype,
            cpu_threads=self.model_config.num_threads,
            runtime_shape="single-device-per-backend",
            reference_enabled=self.reference_enabled,
            reference_device=self.reference_device or self.model_config.device or "auto",
        )
        self._replay_runtime_bootstrap_logs()

        status = "completed"
        assert self._trainer is not None
        if not self._resume_from_checkpoint:
            self._trainer.reset_state()
        self._trainer.attach_run_logger(self._run_logger)
        training_loop_started_at = time.perf_counter()
        try:
            training_loop_started_at = time.perf_counter()
            self._trainer.train(dataset)
        except Exception as exc:
            status = "failed"
            if self._run_logger is not None:
                epoch = self._trainer.current_epoch + 1
                step = self._trainer.total_steps
                context = {
                    "stage": "train",
                    "step": step,
                }
                context["epoch"] = epoch
                self._run_logger.log_exception(exc, context=context)
            raise
        finally:
            self._run_lifecycle_totals["training_loop_seconds"] = (
                time.perf_counter() - training_loop_started_at
            )
            self._trainer.attach_run_logger(None)
            self._resume_from_checkpoint = False
            if self._run_logger is not None:
                self._run_logger.finish_run(
                    status=status,
                    total_steps=self._trainer.total_steps,
                    lifecycle_totals=self._run_lifecycle_totals,
                )

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        assert self._trainer is not None

        started_at = time.perf_counter()
        self._trainer.save_checkpoint(path)
        duration_seconds = time.perf_counter() - started_at
        self._run_lifecycle_totals["checkpoint_save_seconds"] = (
            self._run_lifecycle_totals.get("checkpoint_save_seconds", 0.0)
            + duration_seconds
        )
        if self._run_logger is not None:
            self._run_logger.log_checkpoint(
                "save",
                path,
                epoch=self._trainer.current_epoch + 1,
                step=self._trainer.total_steps,
                duration_seconds=duration_seconds,
            )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        assert self._trainer is not None

        started_at = time.perf_counter()
        self._trainer.load_checkpoint(path)
        self._resume_from_checkpoint = True
        duration_seconds = time.perf_counter() - started_at
        self._run_lifecycle_totals["checkpoint_load_seconds"] = (
            self._run_lifecycle_totals.get("checkpoint_load_seconds", 0.0)
            + duration_seconds
        )
        if self._run_logger is not None:
            self._run_logger.log_checkpoint(
                "load",
                path,
                epoch=self._trainer.current_epoch + 1,
                step=self._trainer.total_steps,
                duration_seconds=duration_seconds,
            )
