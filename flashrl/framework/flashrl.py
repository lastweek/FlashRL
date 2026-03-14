"""FlashRL: Unified RL training API."""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

from .admin import AdminRegistry, AdminServer, build_admin_object, utc_now_iso
from .backends.training import TrainingBackend
from .config import (
    CommonConfig,
    GrpoConfig,
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
from .serving import ServingBackend, create_serving_backend
from .trainer.grpo import GRPOTrainer


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


COMMON_MODEL_SECTION_FIELDS = {
    "model_name",
    "device",
    "dtype",
    "max_length",
    "load_in_8bit",
    "trust_remote_code",
    "num_threads",
    "metadata",
}

SERVING_ONLY_SECTION_FIELDS = {
    "backend",
    "runtime_python",
    "num_replicas",
    "vllm_args",
    "debug_live_rollout",
}


def _resolve_model_config(
    *,
    common: CommonConfig | None,
    section: CommonConfig,
    config_cls: type[ModelConfig] | type[ServingConfig],
    section_name: str,
) -> ModelConfig | ServingConfig:
    """Merge optional common defaults with one model-copy section."""
    merged = {}
    if common is not None:
        merged.update(common.model_dump(exclude_none=True))
    include_fields = set(COMMON_MODEL_SECTION_FIELDS)
    if config_cls is ServingConfig:
        include_fields.update(SERVING_ONLY_SECTION_FIELDS)
    merged.update(section.model_dump(include=include_fields, exclude_none=True))

    if not merged.get("model_name"):
        raise ValueError(
            f"Run config requires '{section_name}.model_name' or 'common.model_name' after merge."
        )
    return config_cls(**merged)


def _rollout_config_from_grpo(grpo_config: GrpoConfig) -> RolloutConfig:
    """Build the internal rollout config from GRPO sampling knobs."""
    return RolloutConfig(
        max_new_tokens=grpo_config.max_new_tokens,
        temperature=grpo_config.temperature,
        top_p=grpo_config.top_p,
        top_k=grpo_config.top_k,
        do_sample=grpo_config.do_sample,
    )


class FlashRL:
    """Unified FlashRL trainer with a simple RL training API."""

    def __init__(
        self,
        model: str,
        rollout_fn: Callable[[list[Prompt], ServingBackend], list[RolloutOutput]],
        reward_fn: Callable[[RolloutOutput], RewardOutput],
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        max_epochs: int = 10,
        seed: int = 42,
        shuffle_each_epoch: bool = True,
        clip_ratio: float = 0.2,
        kl_coefficient: float = 0.0,
        gamma: float = 1.0,
        device: str | None = None,
        dtype: str = "float32",
        max_length: int = 2048,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        num_threads: int = 1,
        grpo_config: GrpoConfig | None = None,
        serving_config: ServingConfig | None = None,
        logging_config: LoggingConfig | None = None,
        metrics_config: MetricsConfig | None = None,
        reference_enabled: bool = False,
        reference_device: str | None = None,
        admin_enabled: bool = True,
        admin_host: str = "127.0.0.1",
        admin_port: int = 0,
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
        self.admin_enabled = admin_enabled
        self.admin_host = admin_host
        self.admin_port = admin_port
        self.admin_base_url: str | None = None

        self.training_model_config = ModelConfig(
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
            else ServingConfig(**self.training_model_config.model_dump())
        )
        self.trainer_config = TrainerConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            seed=seed,
            shuffle_each_epoch=shuffle_each_epoch,
        )
        self.grpo_config = grpo_config or GrpoConfig(
            clip_ratio=clip_ratio,
            kl_coefficient=kl_coefficient,
            metadata={"gamma": gamma},
        )
        if self.trainer_config.batch_size % self.grpo_config.group_size != 0:
            raise ValueError(
                "training.batch_size must be divisible by grpo.group_size "
                f"(got batch_size={self.trainer_config.batch_size}, "
                f"group_size={self.grpo_config.group_size})."
            )
        self.rollout_config = _rollout_config_from_grpo(self.grpo_config)
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
        self._admin_registry: AdminRegistry | None = None
        self._admin_server: AdminServer | None = None
        self._runtime_uid = uuid4().hex
        self._runtime_created_at = utc_now_iso()
        self._runtime_phase = "Pending"
        self._last_runtime_error: str | None = None
        self._closed = False

        if self.metrics_config.enabled:
            self._metrics_sink = PrometheusMetricsSink(
                self.metrics_config,
                model_name=self.training_model_config.model_name,
            )

        self._initialize_runtime()

    def _apply_random_seed(self) -> None:
        """Apply the configured host-process RNG seed."""
        random.seed(self.trainer_config.seed)
        try:
            import torch

            torch.manual_seed(self.trainer_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.trainer_config.seed)
        except Exception:
            return None

    def _bootstrap_console_enabled(self) -> bool:
        """Return whether live bootstrap console output should be emitted."""
        return bool(self.logging_config.console)

    def _emit_bootstrap_console(self, line: str) -> None:
        """Write one live bootstrap line to stdout immediately."""
        if not self._bootstrap_console_enabled():
            return
        print(line, flush=True)

    def _format_bootstrap_duration(self, duration_seconds: float) -> str:
        """Format one startup duration for the console."""
        if duration_seconds < 1.0:
            return f"{duration_seconds * 1000.0:.1f}ms"
        return f"{duration_seconds:.3f}s"

    def _serving_replica_summary(self) -> str:
        """Return the serving replica suffix for replicated backends."""
        if self.serving_config.backend != "vllm":
            return ""
        return f" replicas={self.serving_config.num_replicas}"

    def _emit_bootstrap_banner(self) -> None:
        """Print the immediate startup banner before any long-running work."""
        reference_state = "enabled" if self.reference_enabled else "disabled"
        self._emit_bootstrap_console(
            "FlashRL startup  "
            f"model={self.training_model_config.model_name}  "
            f"serving={self.serving_config.backend}"
            f"{self._serving_replica_summary()}  "
            f"reference={reference_state}"
        )

    def _emit_bootstrap_stage(self, component: str, message: str) -> None:
        """Print one concise bootstrap stage line."""
        self._emit_bootstrap_console(f"startup {component:<17} {message}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlashRL":
        """Construct FlashRL from a YAML run config."""
        config_path = Path(path)
        run_config = RunConfig.from_yaml(config_path)
        training_model_config = _resolve_model_config(
            common=run_config.common,
            section=run_config.training,
            config_cls=ModelConfig,
            section_name="training",
        )
        serving_config = _resolve_model_config(
            common=run_config.common,
            section=run_config.serving,
            config_cls=ServingConfig,
            section_name="serving",
        )
        trainer_config = TrainerConfig(
            learning_rate=run_config.training.learning_rate,
            batch_size=run_config.training.batch_size,
            max_epochs=run_config.training.max_epochs,
            seed=run_config.training.seed,
            shuffle_each_epoch=run_config.training.shuffle_each_epoch,
        )
        rollout_fn = _resolve_import_string(run_config.hooks.rollout_fn)
        reward_fn = _resolve_import_string(run_config.hooks.reward_fn)
        dataset_fn = _resolve_import_string(run_config.hooks.dataset_fn)

        instance = cls(
            model=training_model_config.model_name,
            rollout_fn=rollout_fn,
            reward_fn=reward_fn,
            learning_rate=trainer_config.learning_rate,
            batch_size=trainer_config.batch_size,
            max_epochs=trainer_config.max_epochs,
            seed=trainer_config.seed,
            shuffle_each_epoch=trainer_config.shuffle_each_epoch,
            device=training_model_config.device,
            dtype=training_model_config.dtype,
            max_length=training_model_config.max_length,
            load_in_8bit=training_model_config.load_in_8bit,
            trust_remote_code=training_model_config.trust_remote_code,
            num_threads=training_model_config.num_threads,
            grpo_config=run_config.grpo,
            serving_config=serving_config,
            logging_config=run_config.logging,
            metrics_config=run_config.metrics,
            reference_enabled=run_config.runtime.reference_enabled,
            reference_device=run_config.runtime.reference_device,
            admin_enabled=run_config.runtime.admin_enabled,
            admin_host=run_config.runtime.admin_host,
            admin_port=run_config.runtime.admin_port,
        )
        instance._dataset_loader = dataset_fn
        return instance

    def _initialize_runtime(self) -> None:
        """Initialize training and serving backends, and reference model if needed."""
        self._runtime_phase = "Starting"
        self._runtime_bootstrap_events = []
        self._runtime_bootstrap_totals = {}
        startup_total_seconds = 0.0
        self._apply_random_seed()
        self._emit_bootstrap_banner()

        self._emit_bootstrap_stage("training_backend", "starting")
        started_at = time.perf_counter()
        self._training_backend = TrainingBackend(
            self.training_model_config,
            learning_rate=self.trainer_config.learning_rate,
        )
        duration_seconds = time.perf_counter() - started_at
        startup_total_seconds += duration_seconds
        self._emit_bootstrap_stage(
            "training_backend",
            "ready "
            f"device={self._training_backend.actor.device} "
            f"cpu={self.training_model_config.num_threads} "
            f"{self._format_bootstrap_duration(duration_seconds)}",
        )
        self._runtime_bootstrap_totals["startup_training_backend_seconds"] = duration_seconds
        self._runtime_bootstrap_events.append(
            self._make_model_load_event(
                component="training_backend",
                duration_seconds=duration_seconds,
                device=self._training_backend.actor.device,
                cpu_threads=self.training_model_config.num_threads,
            )
        )

        self._emit_bootstrap_stage(
            "serving_backend",
            "starting "
            f"backend={self.serving_config.backend}"
            f"{self._serving_replica_summary()}",
        )
        started_at = time.perf_counter()
        self._serving_backend = create_serving_backend(
            self.serving_config,
            startup_logger=(
                self._emit_bootstrap_console if self._bootstrap_console_enabled() else None
            ),
        )
        duration_seconds = time.perf_counter() - started_at
        startup_total_seconds += duration_seconds
        serving_ready_line = (
            "ready "
            f"backend={self.serving_config.backend} "
            f"device={self._serving_backend.device}"
        )
        if self.serving_config.backend == "vllm":
            serving_ready_line += f" replicas={self.serving_config.num_replicas}"
        serving_ready_line += f" {self._format_bootstrap_duration(duration_seconds)}"
        self._emit_bootstrap_stage("serving_backend", serving_ready_line)
        self._runtime_bootstrap_totals["startup_serving_backend_seconds"] = duration_seconds
        self._runtime_bootstrap_events.append(
            self._make_model_load_event(
                component="serving_backend",
                duration_seconds=duration_seconds,
                device=self._serving_backend.device,
                cpu_threads=self.serving_config.num_threads,
            )
        )

        if self.reference_enabled:
            self._emit_bootstrap_stage("reference_model", "starting")
            started_at = time.perf_counter()
            reference_config = self.training_model_config.model_copy(
                update={
                    "device": self.reference_device or self.training_model_config.device
                }
            )
            self._reference = ReferenceModel(reference_config)
            duration_seconds = time.perf_counter() - started_at
            startup_total_seconds += duration_seconds
            self._emit_bootstrap_stage(
                "reference_model",
                "ready "
                f"device={self._reference.device} "
                f"cpu={reference_config.num_threads} "
                f"{self._format_bootstrap_duration(duration_seconds)}",
            )
            self._runtime_bootstrap_totals["startup_reference_model_seconds"] = duration_seconds
            self._runtime_bootstrap_events.append(
                self._make_model_load_event(
                    component="reference_model",
                    duration_seconds=duration_seconds,
                    device=self._reference.device,
                    cpu_threads=reference_config.num_threads,
                )
            )
        else:
            self._reference = None

        self._rollout_generator = UserDefinedRollout(
            rollout_fn=self.rollout_fn,
            serving_backend=self._serving_backend,
            config=self.rollout_config,
        )
        self._reward = UserDefinedReward(
            reward_fn=self.reward_fn,
            config=self.reward_config,
        )

        self._trainer = GRPOTrainer(
            config=self.trainer_config,
            grpo_config=self.grpo_config,
            training_backend=self._training_backend,
            serving_backend=self._serving_backend,
            reference=self._reference,
            reward_fn=self._reward,
            rollout_generator=self._rollout_generator,
            run_logger=None,
            metrics_sink=self._metrics_sink,
        )
        self._initialize_admin()
        self._runtime_bootstrap_totals["startup_total_seconds"] = startup_total_seconds
        self._runtime_phase = "Ready"

    def _make_model_load_event(
        self,
        component: str,
        duration_seconds: float,
        device: Any,
        cpu_threads: int,
    ) -> dict[str, Any]:
        """Build one cached model-load event for replay into each run logger."""
        return {
            "component": component,
            "status": "completed",
            "metadata": {
                "device": str(device),
                "cpu_threads": cpu_threads,
                "duration_seconds": duration_seconds,
            },
        }

    def _initialize_admin(self) -> None:
        """Start the runtime-owned admin registry and HTTP server when enabled."""
        self._admin_registry = AdminRegistry()
        self._admin_registry.register(self._runtime_admin_objects)
        self._admin_registry.register(self._training_backend_admin_objects)
        self._admin_registry.register(self._serving_backend_admin_objects)
        self._admin_registry.register(self._reference_admin_objects)
        self._admin_registry.register(self._serving_child_admin_objects)

        if not self.admin_enabled:
            return

        self._admin_server = AdminServer(
            self._admin_registry,
            host=self.admin_host,
            port=self.admin_port,
        )
        self.admin_base_url = self._admin_server.start()
        self._emit_bootstrap_stage("admin", f"ready url={self.admin_base_url}")

    def _runtime_labels(self) -> dict[str, str]:
        """Build shared labels for all runtime-owned admin objects."""
        return {
            "flashrl.dev/runtime": "flashrl",
            "flashrl.dev/model-name": self.training_model_config.model_name,
            "flashrl.dev/serving-backend": self.serving_config.backend,
        }

    def _runtime_object_uid(self, suffix: str) -> str:
        """Build one stable UID string for a runtime-owned object."""
        return f"{self._runtime_uid}:{suffix}"

    def _runtime_admin_objects(self) -> list[dict[str, Any]]:
        """Return the parent runtime admin object."""
        return [
            build_admin_object(
                "FlashRLRuntime",
                "flashrl-runtime",
                uid=self._runtime_object_uid("runtime"),
                created_at=self._runtime_created_at,
                labels=self._runtime_labels(),
                spec={
                    "modelName": self.training_model_config.model_name,
                    "referenceEnabled": self.reference_enabled,
                    "adminBaseUrl": self.admin_base_url,
                },
                status={
                    "phase": self._runtime_phase,
                    "startedAt": self._runtime_created_at,
                    "currentEpoch": self._admin_current_epoch(),
                    "currentStep": self._admin_current_step(),
                    "lastError": self._last_runtime_error,
                },
            )
        ]

    def _training_backend_admin_objects(self) -> list[dict[str, Any]]:
        """Return the training backend admin object."""
        if self._training_backend is None:
            return []
        return [
            build_admin_object(
                "TrainingBackend",
                "training-backend",
                uid=self._runtime_object_uid("training-backend"),
                created_at=self._runtime_created_at,
                labels=self._runtime_labels(),
                spec={
                    "modelName": self.training_model_config.model_name,
                    "device": self.training_model_config.device or "auto",
                    "dtype": self.training_model_config.dtype,
                    "numThreads": self.training_model_config.num_threads,
                },
                status={
                    "phase": self._component_phase(),
                    "device": str(self._training_backend.actor.device),
                    "optimizer": type(self._training_backend.optimizer).__name__,
                    "loaded": True,
                },
            )
        ]

    def _serving_backend_admin_objects(self) -> list[dict[str, Any]]:
        """Return the serving backend admin object."""
        if self._serving_backend is None:
            return []

        child_objects = self._serving_child_admin_objects()
        active_replica_count = sum(
            1
            for item in child_objects
            if item.get("kind") == "VLLMInstance"
            and item.get("status", {}).get("healthy") is True
        )
        backend_phase = self._component_phase()
        if self.serving_config.backend == "vllm" and child_objects:
            expected = self.serving_config.num_replicas
            if active_replica_count == expected:
                backend_phase = "Ready"
            elif active_replica_count > 0:
                backend_phase = "Degraded"
            elif self._runtime_phase not in {"Closing", "Closed", "Failed"}:
                backend_phase = "Starting"

        return [
            build_admin_object(
                "ServingBackend",
                "serving-backend",
                uid=self._runtime_object_uid("serving-backend"),
                created_at=self._runtime_created_at,
                labels=self._runtime_labels(),
                spec={
                    "backend": self.serving_config.backend,
                    "modelName": self.serving_config.model_name,
                    "device": self.serving_config.device or "auto",
                    "numReplicas": self.serving_config.num_replicas,
                },
                status={
                    "phase": backend_phase,
                    "device": str(self._serving_backend.device),
                    "activeReplicaCount": active_replica_count,
                },
            )
        ]

    def _reference_admin_objects(self) -> list[dict[str, Any]]:
        """Return the optional reference model admin object."""
        if self._reference is None:
            return []
        return [
            build_admin_object(
                "ReferenceModel",
                "reference-model",
                uid=self._runtime_object_uid("reference-model"),
                created_at=self._runtime_created_at,
                labels=self._runtime_labels(),
                spec={
                    "modelName": self.training_model_config.model_name,
                    "device": self.reference_device or self.training_model_config.device or "auto",
                },
                status={
                    "phase": self._component_phase(),
                    "loaded": True,
                },
            )
        ]

    def _serving_child_admin_objects(self) -> list[dict[str, Any]]:
        """Return backend-owned child admin objects."""
        if self._serving_backend is None:
            return []
        list_objects = getattr(self._serving_backend, "list_admin_objects", None)
        if list_objects is None:
            return []
        return list_objects()

    def _component_phase(self) -> str:
        """Map the runtime phase to a generic component phase."""
        if self._runtime_phase == "Training":
            return "Ready"
        return self._runtime_phase

    def _admin_current_epoch(self) -> int:
        """Return the user-facing epoch number for the runtime object."""
        if self._trainer is None:
            return 0
        if self._trainer.total_steps == 0 and self._runtime_phase == "Ready":
            return 0
        return self._trainer.current_epoch + 1

    def _admin_current_step(self) -> int:
        """Return the current training step for the runtime object."""
        if self._trainer is None:
            return 0
        return self._trainer.total_steps

    def close(self) -> None:
        """Release runtime-owned resources."""
        if self._closed:
            return
        self._closed = True
        if self._run_logger is not None:
            self._run_logger.close()
            self._run_logger = None
        self._runtime_phase = "Closing"
        try:
            if self._serving_backend is not None:
                self._serving_backend.close()
        finally:
            self._runtime_phase = "Closed"
            if self._admin_server is not None:
                self._admin_server.close()
                self._admin_server = None
        self._serving_backend = None

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
        prompts_per_step = self.trainer_config.batch_size // self.grpo_config.group_size
        total_batches = (
            math.ceil(len(dataset) / prompts_per_step) * self.trainer_config.max_epochs
            if dataset
            else 0
        )
        self._run_logger = RunLogger(
            self.logging_config,
            model_name=self.training_model_config.model_name,
        )
        self._run_logger.start_run(
            dataset_size=len(dataset),
            batch_size=self.trainer_config.batch_size,
            max_epochs=self.trainer_config.max_epochs,
            total_batches=total_batches,
            device=self.training_model_config.device or "auto",
            dtype=self.training_model_config.dtype,
            cpu_threads=self.training_model_config.num_threads,
            runtime_shape="single-device-per-backend",
            reference_enabled=self.reference_enabled,
            reference_device=self.reference_device
            or self.training_model_config.device
            or "auto",
            group_size=self.grpo_config.group_size,
            clip_ratio=self.grpo_config.clip_ratio,
            prompts_per_step=prompts_per_step,
            steps_per_epoch=(math.ceil(len(dataset) / prompts_per_step) if dataset else 0),
            total_planned_steps=total_batches,
            serving_backend=self.serving_config.backend,
            serving_device=str(self._serving_backend.device),
            serving_num_replicas=self.serving_config.num_replicas,
            admin_base_url=self.admin_base_url,
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
        self._last_runtime_error = None
        self._runtime_phase = "Training"
        try:
            training_loop_started_at = time.perf_counter()
            self._trainer.train(dataset)
        except Exception as exc:
            status = "failed"
            self._runtime_phase = "Failed"
            self._last_runtime_error = f"{type(exc).__name__}: {exc}"
            if self._run_logger is not None:
                epoch = self._trainer.current_epoch + 1
                step = self._trainer.total_steps
                context = {
                    "stage": "train",
                    "step": step,
                }
                context["epoch"] = epoch
                self._run_logger.log_exception(exc, context=context)
            if self._serving_backend is not None:
                self._serving_backend.close()
            raise
        finally:
            self._run_lifecycle_totals["training_loop_seconds"] = (
                time.perf_counter() - training_loop_started_at
            )
            self._trainer.attach_run_logger(None)
            self._resume_from_checkpoint = False
            if status == "completed":
                self._runtime_phase = "Ready"
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
    flashrl: FlashRL | None = None

    try:
        flashrl = FlashRL.from_yaml(args.config)
        flashrl.train()
    except Exception as exc:
        print(f"FlashRL YAML run failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if flashrl is not None:
            flashrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
