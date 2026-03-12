"""FlashRL: Unified RL training API."""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .backends.serving import ServingBackend
from .backends.training import TrainingBackend
from .config import (
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    RewardConfig,
    RunConfig,
    RolloutConfig,
    ServingConfig,
    TrainerConfig,
)
from .data_models import Prompt, RewardOutput, RolloutOutput
from .models.reference import ReferenceModel
from .metrics import PrometheusMetricsSink
from .reward.user_defined import UserDefinedReward
from .rollout.user_defined import UserDefinedRollout
from .run_logger import RunLogger
from .trainer.grpo import GRPOTrainer

if TYPE_CHECKING:
    from .models.actor import ActorModel


def _resolve_import_string(import_string: str) -> Any:
    """Resolve a ``module:attribute`` import string."""
    module_name, separator, attr_path = import_string.partition(":")
    if separator == "" or not module_name or not attr_path:
        raise ValueError(
            "Hook import strings must use the format 'module.submodule:attribute'."
        )

    module = importlib.import_module(module_name)
    resolved = module
    for attr_name in attr_path.split("."):
        resolved = getattr(resolved, attr_name)
    return resolved


def _coerce_dataset(dataset: list[Prompt] | list[str]) -> list[Prompt]:
    """Normalize string datasets into Prompt objects."""
    normalized: list[Prompt] = []
    for item in dataset:
        if isinstance(item, Prompt):
            normalized.append(item)
        else:
            normalized.append(Prompt(text=str(item)))
    return normalized


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
        clip_epsilon: float = 0.2,
        kl_coefficient: float = 0.0,
        gamma: float = 1.0,
        device: str | None = None,
        dtype: str = "float32",
        max_length: int = 2048,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        num_threads: int = 1,
        serving_config: ServingConfig | None = None,
        logging_config: LoggingConfig | None = None,
        metrics_config: MetricsConfig | None = None,
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
            dtype=dtype,
            max_length=max_length,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            num_threads=num_threads,
        )
        self.serving_config = (
            serving_config
            if serving_config is not None
            else ServingConfig(**self.model_config.model_dump())
        )
        self.trainer_config = TrainerConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            clip_epsilon=clip_epsilon,
            kl_coefficient=kl_coefficient,
            gamma=gamma,
        )
        self.rollout_config = RolloutConfig()
        self.reward_config = RewardConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.metrics_config = metrics_config or MetricsConfig()

        self._training_backend: TrainingBackend | None = None
        self._serving_backend: ServingBackend | None = None
        self._reference: ReferenceModel | None = None
        self._rollout_generator: UserDefinedRollout | None = None
        self._reward: UserDefinedReward | None = None
        self._trainer: GRPOTrainer | None = None
        self._run_logger: RunLogger | None = None
        self._metrics_sink: PrometheusMetricsSink | None = None
        self._run_lifecycle_totals: dict[str, float] = {}
        self._runtime_bootstrap_events: list[dict[str, Any]] = []
        self._runtime_bootstrap_totals: dict[str, float] = {}
        self._resume_from_checkpoint = False
        self._dataset_loader: Callable[[], list[Prompt] | list[str]] | None = None

        if self.metrics_config.enabled:
            self._metrics_sink = PrometheusMetricsSink(
                self.metrics_config,
                model_name=self.model_config.model_name,
            )

        self._initialize_runtime()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlashRL":
        """Construct FlashRL from a YAML run config."""
        config_path = Path(path)
        run_config = RunConfig.from_yaml(config_path)
        rollout_fn = _resolve_import_string(run_config.hooks.rollout_fn)
        reward_fn = _resolve_import_string(run_config.hooks.reward_fn)
        dataset_fn = _resolve_import_string(run_config.hooks.dataset_fn)

        instance = cls(
            model=run_config.model.model_name,
            rollout_fn=rollout_fn,
            reward_fn=reward_fn,
            learning_rate=run_config.trainer.learning_rate,
            batch_size=run_config.trainer.batch_size,
            max_epochs=run_config.trainer.max_epochs,
            clip_epsilon=run_config.trainer.clip_epsilon,
            kl_coefficient=run_config.trainer.kl_coefficient,
            gamma=run_config.trainer.gamma,
            device=run_config.model.device,
            dtype=run_config.model.dtype,
            max_length=run_config.model.max_length,
            load_in_8bit=run_config.model.load_in_8bit,
            trust_remote_code=run_config.model.trust_remote_code,
            num_threads=run_config.model.num_threads,
            serving_config=(
                run_config.serving
                if run_config.serving is not None
                else ServingConfig(**run_config.model.model_dump())
            ),
            logging_config=run_config.logging,
            metrics_config=run_config.metrics,
            reference_enabled=run_config.runtime.reference_enabled,
            reference_device=run_config.runtime.reference_device,
        )
        instance._dataset_loader = dataset_fn
        return instance

    def _initialize_runtime(self) -> None:
        """Initialize training and serving backends, and reference model if needed."""
        self._runtime_bootstrap_events = []
        self._runtime_bootstrap_totals = {}
        startup_total_seconds = 0.0

        started_at = time.perf_counter()
        self._training_backend = TrainingBackend(
            self.model_config,
            learning_rate=self.trainer_config.learning_rate,
        )
        duration_seconds = time.perf_counter() - started_at
        startup_total_seconds += duration_seconds
        self._runtime_bootstrap_totals["startup_training_backend_seconds"] = duration_seconds
        self._runtime_bootstrap_events.append(
            self._make_model_load_event(
                component="training_backend",
                duration_seconds=duration_seconds,
                device=self._training_backend.actor.device,
            )
        )

        started_at = time.perf_counter()
        self._serving_backend = ServingBackend(self.serving_config)
        duration_seconds = time.perf_counter() - started_at
        startup_total_seconds += duration_seconds
        self._runtime_bootstrap_totals["startup_serving_backend_seconds"] = duration_seconds
        self._runtime_bootstrap_events.append(
            self._make_model_load_event(
                component="serving_backend",
                duration_seconds=duration_seconds,
                device=self._serving_backend.actor.device,
            )
        )

        if self.reference_enabled:
            started_at = time.perf_counter()
            reference_config = self.model_config.model_copy(
                update={"device": self.reference_device or self.model_config.device}
            )
            self._reference = ReferenceModel(reference_config)
            duration_seconds = time.perf_counter() - started_at
            startup_total_seconds += duration_seconds
            self._runtime_bootstrap_totals["startup_reference_model_seconds"] = duration_seconds
            self._runtime_bootstrap_events.append(
                self._make_model_load_event(
                    component="reference_model",
                    duration_seconds=duration_seconds,
                    device=self._reference.device,
                )
            )
        else:
            self._reference = None

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
            metrics_sink=self._metrics_sink,
        )
        self._runtime_bootstrap_totals["startup_total_seconds"] = startup_total_seconds

    def _make_model_load_event(
        self,
        component: str,
        duration_seconds: float,
        device: Any,
    ) -> dict[str, Any]:
        """Build one cached model-load event for replay into each run logger."""
        return {
            "component": component,
            "status": "completed",
            "metadata": {
                "device": str(device),
                "cpu_threads": self.model_config.num_threads,
                "duration_seconds": duration_seconds,
            },
        }

    def train(self, dataset: list[Prompt] | list[str] | None = None) -> None:
        """Train on dataset or use the configured dataset loader."""
        if dataset is None:
            if self._dataset_loader is None:
                raise ValueError(
                    "FlashRL.train() requires a dataset unless the trainer was created with "
                    "FlashRL.from_yaml(...) and hooks.dataset_fn is configured."
                )
            dataset = self._dataset_loader()

        dataset = _coerce_dataset(dataset)
        if self._run_logger is not None:
            self._run_logger.close()

        checkpoint_totals = {
            key: value
            for key, value in self._run_lifecycle_totals.items()
            if key.startswith("checkpoint_")
        }
        self._run_lifecycle_totals = {
            **self._runtime_bootstrap_totals,
            **checkpoint_totals,
        }
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
        for event in self._runtime_bootstrap_events:
            self._run_logger.log_model_load(
                event["component"],
                event["status"],
                event["metadata"],
            )

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
                self._run_logger.close()

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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for YAML-driven runs."""
    parser = argparse.ArgumentParser(description="Run FlashRL from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the FlashRL YAML config file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m flashrl.framework.flashrl``."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        flashrl = FlashRL.from_yaml(args.config)
        flashrl.train()
    except Exception as exc:
        print(f"FlashRL YAML run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
