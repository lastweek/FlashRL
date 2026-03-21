"""FlashRL: Unified RL training API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence
from uuid import uuid4

from . import runtime_support
from .admin import AdminRegistry, AdminServer, build_admin_object, utc_now_iso
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    RestoredCheckpoint,
)
from .config import (
    AdminConfig,
    CheckpointingConfig,
    ControllerConfig,
    GrpoConfig,
    LoggingConfig,
    MetricsConfig,
    RewardConfig,
    RunConfig,
    ServingConfig,
    TrainingConfig,
)
from .data_models import Prompt, RewardOutput, RolloutOutput, WeightVersionInfo
from .memory import capture_memory_snapshot
from .metrics import MetricsSink, build_metrics_sink
from .observability import timed_call
from .reward.user_defined import UserDefinedReward
from .rollout.base import BaseRolloutGenerator, build_rollout_generator
from .run_logger import RunLogger
from .serving import ServingBackend, create_serving_backend
from .train_runtime import (
    TrainRunState,
    build_train_run_state,
    finish_run_observers,
    open_run_logger as open_shared_run_logger,
    start_run_metrics,
)
from .training import ActorTrainingBackend, ReferenceTrainingBackend, TrainingBackend, create_training_backend
from .controller.grpo.controller import GRPOController

if TYPE_CHECKING:
    from .agent import Agent


class FlashRL:
    """Unified FlashRL controller with a simple RL training API."""

    def __init__(
        self,
        *,
        actor_config: TrainingConfig | None = None,
        reference_config: TrainingConfig | None = None,
        serving_config: ServingConfig | None = None,
        controller_config: ControllerConfig | None = None,
        grpo_config: GrpoConfig | None = None,
        rollout_fn: Callable[[list[Prompt], ServingBackend], list[RolloutOutput]] | Agent | None = None,
        reward_fn: Callable[[RolloutOutput], RewardOutput] | None = None,
        logging_config: LoggingConfig | None = None,
        metrics_config: MetricsConfig | None = None,
        checkpointing_config: CheckpointingConfig | None = None,
        admin_config: AdminConfig | None = None,
        config_path: str | Path | None = None,
        run_config: RunConfig | dict[str, Any] | None = None,
        dataset_loader: Callable[[], list[Prompt] | list[str]] | None = None,
    ) -> None:
        """Initialize FlashRL controller.

        The runtime keeps separate actor, optional reference, and serving model
        copies with explicit role assignment.
        """
        resolved_run_config = runtime_support.load_run_config(
            config_path=config_path,
            run_config=run_config,
        )
        if resolved_run_config is not None:
            if rollout_fn is None or reward_fn is None:
                raise ValueError(
                    "FlashRL config-based construction requires explicit rollout_fn and reward_fn."
                )
            if (
                actor_config is not None
                or reference_config is not None
                or serving_config is not None
                or controller_config is not None
                or grpo_config is not None
                or logging_config is not None
                or metrics_config is not None
                or checkpointing_config is not None
                or admin_config is not None
            ):
                raise ValueError(
                    "FlashRL config-based construction cannot be combined with explicit "
                    "actor/reference/serving/controller/admin/checkpointing overrides."
                )

            (
                actor_config,
                reference_config,
                serving_config,
                controller_config,
                admin_config,
            ) = runtime_support.build_runtime_role_configs(resolved_run_config)
            grpo_config = resolved_run_config.grpo
            logging_config = resolved_run_config.logging
            metrics_config = resolved_run_config.metrics
            checkpointing_config = resolved_run_config.checkpointing

        if actor_config is None:
            raise ValueError(
                "FlashRL(...) requires actor_config unless config_path or run_config is provided."
            )
        if serving_config is None:
            raise ValueError(
                "FlashRL(...) requires serving_config unless config_path or run_config is provided."
            )
        if controller_config is None:
            raise ValueError(
                "FlashRL(...) requires controller_config unless config_path or run_config is provided."
            )
        if grpo_config is None:
            raise ValueError(
                "FlashRL(...) requires grpo_config unless config_path or run_config is provided."
            )
        if rollout_fn is None or reward_fn is None:
            raise ValueError("FlashRL(...) requires explicit rollout_fn and reward_fn.")
        if grpo_config.kl_coefficient > 0.0 and reference_config is None:
            raise ValueError("reference_config is required when grpo_config.kl_coefficient > 0.")
        if grpo_config.kl_coefficient <= 0.0 and reference_config is not None:
            raise ValueError("reference_config must be omitted when grpo_config.kl_coefficient <= 0.")

        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.admin_config = admin_config or AdminConfig()
        self.admin_enabled = self.admin_config.admin_enabled
        self.admin_host = self.admin_config.admin_host
        self.admin_port = self.admin_config.admin_port
        self.admin_base_url: str | None = None

        self.actor_config = actor_config
        self.reference_config = reference_config
        self.serving_config = serving_config
        self.controller_config = controller_config
        self.grpo_config = grpo_config
        if self.controller_config.batch_size % self.grpo_config.group_size != 0:
            raise ValueError(
                "controller.batch_size must be divisible by grpo.group_size "
                f"(got batch_size={self.controller_config.batch_size}, "
                f"group_size={self.grpo_config.group_size})."
            )
        self.rollout_config = runtime_support.build_rollout_config(self.grpo_config)
        self.reward_config = RewardConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.metrics_config = metrics_config or MetricsConfig()
        self.checkpointing_config = checkpointing_config or CheckpointingConfig()

        self._actor_backend: ActorTrainingBackend | None = None
        self._reference_backend: ReferenceTrainingBackend | None = None
        self._serving_backend: ServingBackend | None = None
        self._rollout_generator: BaseRolloutGenerator | None = None
        self._reward: UserDefinedReward | None = None
        self._controller: GRPOController | None = None
        self._run_logger: RunLogger | None = None
        self._metrics_sink: MetricsSink | None = None
        self._run_lifecycle_totals: dict[str, float] = {}
        self._restored_run_lifecycle_totals: dict[str, float] = {}
        self._runtime_bootstrap_events: list[dict[str, Any]] = []
        self._runtime_bootstrap_totals: dict[str, float] = {}
        self._startup_artifact_dir: Path | None = None
        self._resume_from_checkpoint = False
        self._managed_resume: RestoredCheckpoint | None = None
        self._managed_resume_load_seconds = 0.0
        self._dataset_loader = dataset_loader
        self._admin_registry: AdminRegistry | None = None
        self._admin_server: AdminServer | None = None
        self._runtime_uid = uuid4().hex
        self._runtime_created_at = utc_now_iso()
        self._runtime_phase = "Pending"
        self._last_runtime_error: str | None = None
        self._closed = False
        self._checkpoint_manager = CheckpointManager(self.checkpointing_config)

        self._metrics_sink = build_metrics_sink(
            self.metrics_config,
            model_name=self.actor_config.model_name,
        )

        self._initialize_runtime()

    def _apply_random_seed(self) -> None:
        """Apply the configured host-process RNG seed."""
        random.seed(self.controller_config.seed)
        try:
            import torch

            torch.manual_seed(self.controller_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.controller_config.seed)
        except Exception:
            return None

    def _bootstrap_console_enabled(self) -> bool:
        """Return whether live bootstrap console output should be emitted."""
        return bool(self.logging_config.console)

    def _emit_bootstrap_console(self, line: str) -> None:
        """Write one live bootstrap line to stdout immediately."""
        self._runtime_bootstrap_console_lines.append(line)
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
        reference_state = (
            self.reference_config.backend if self.reference_config is not None else "disabled"
        )
        self._emit_bootstrap_console("FlashRL startup")
        self._emit_bootstrap_console(
            "  startup  "
            f"model={self.actor_config.model_name}  "
            f"actor={self.actor_config.backend}  "
            f"dp_size={self.actor_config.dp_size}  "
            f"serving={self.serving_config.backend}"
            f"{self._serving_replica_summary()}  "
            f"reference={reference_state}"
        )

    def _emit_bootstrap_stage(self, label: str, component: str, message: str) -> None:
        """Print one concise bootstrap stage line."""
        self._emit_bootstrap_console(f"  {label:<8} {component:<17} {message}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlashRL":
        """Construct FlashRL from a YAML run config."""
        config_path = Path(path)
        run_config = RunConfig.from_yaml(config_path)
        if run_config.hooks is None:
            raise ValueError(
                "FlashRL.from_yaml(...) requires hooks.rollout_fn, hooks.reward_fn, and "
                "hooks.dataset_fn in the YAML config."
            )
        rollout_fn = runtime_support.instantiate_hook(run_config.hooks.rollout_fn)
        reward_fn = runtime_support.instantiate_hook(run_config.hooks.reward_fn)
        dataset_fn = runtime_support.instantiate_hook(run_config.hooks.dataset_fn)

        return cls(
            rollout_fn=rollout_fn,
            reward_fn=reward_fn,
            run_config=run_config,
            dataset_loader=dataset_fn,
        )

    def _initialize_runtime(self) -> None:
        """Initialize training and serving backends."""
        self._runtime_phase = "Starting"
        self._runtime_bootstrap_events = []
        self._runtime_bootstrap_console_lines = []
        self._runtime_bootstrap_totals = {}
        startup_total_seconds = 0.0
        self._apply_random_seed()
        self._emit_bootstrap_banner()
        self._startup_artifact_dir = Path(
            tempfile.mkdtemp(prefix="flashrl-runtime-")
        ).resolve()

        self._emit_bootstrap_stage(
            "startup",
            "actor_backend",
            f"starting backend={self.actor_config.backend} dp_size={self.actor_config.dp_size}",
        )
        self._actor_backend = create_training_backend(
            self.actor_config,
            learning_rate=self.controller_config.learning_rate,
            role="actor",
        )
        for event in self._actor_backend.startup_events:
            duration_seconds = float(event["metadata"]["duration_seconds"])
            startup_total_seconds += duration_seconds
            component = str(event["component"])
            if component == "actor_backend":
                self._emit_bootstrap_stage(
                    "ready",
                    "actor_backend",
                    f"backend={self.actor_config.backend} "
                    f"device={self._actor_backend.device} "
                    f"dp_size={self.actor_config.dp_size} "
                    f"cpu={self.actor_config.num_threads} "
                    f"{self._format_bootstrap_duration(duration_seconds)}",
                )
                self._runtime_bootstrap_totals["startup_actor_backend_seconds"] = duration_seconds
            self._runtime_bootstrap_events.append(
                self._make_model_load_event(
                    component=component,
                    duration_seconds=duration_seconds,
                    device=event["metadata"]["device"],
                    cpu_threads=event["metadata"]["cpu_threads"],
                )
            )

        if self.reference_config is not None:
            self._emit_bootstrap_stage(
                "startup",
                "reference_backend",
                f"starting backend={self.reference_config.backend} dp_size={self.reference_config.dp_size}",
            )
            self._reference_backend = create_training_backend(
                self.reference_config,
                role="reference",
            )
            for event in self._reference_backend.startup_events:
                duration_seconds = float(event["metadata"]["duration_seconds"])
                startup_total_seconds += duration_seconds
                component = str(event["component"])
                self._emit_bootstrap_stage(
                    "ready",
                    "reference_backend",
                    f"backend={self.reference_config.backend} "
                    f"device={self._reference_backend.device} "
                    f"dp_size={self.reference_config.dp_size} "
                    f"cpu={self.reference_config.num_threads} "
                    f"{self._format_bootstrap_duration(duration_seconds)}",
                )
                self._runtime_bootstrap_totals["startup_reference_backend_seconds"] = (
                    duration_seconds
                )
                self._runtime_bootstrap_events.append(
                    self._make_model_load_event(
                        component=component,
                        duration_seconds=duration_seconds,
                        device=event["metadata"]["device"],
                        cpu_threads=event["metadata"]["cpu_threads"],
                    )
                )
        else:
            self._reference_backend = None

        self._emit_bootstrap_stage(
            "startup",
            "serving_backend",
            f"starting backend={self.serving_config.backend}"
            f"{self._serving_replica_summary()}",
        )
        self._serving_backend, duration_seconds = timed_call(
            lambda: create_serving_backend(
                self.serving_config,
                startup_logger=(
                    self._emit_bootstrap_console if self._bootstrap_console_enabled() else None
                ),
                log_dir=self._startup_artifact_dir,
            )
        )
        startup_total_seconds += duration_seconds
        serving_ready_line = (
            f"backend={self.serving_config.backend} "
            f"device={self._serving_backend.device}"
        )
        if self.serving_config.backend == "vllm":
            serving_ready_line += f" replicas={self.serving_config.num_replicas}"
        serving_ready_line += f" {self._format_bootstrap_duration(duration_seconds)}"
        self._emit_bootstrap_stage("ready", "serving_backend", serving_ready_line)
        self._runtime_bootstrap_totals["startup_serving_backend_seconds"] = duration_seconds
        self._runtime_bootstrap_events.append(
            self._make_model_load_event(
                component="serving_backend",
                duration_seconds=duration_seconds,
                device=self._serving_backend.device,
                cpu_threads=self.serving_config.num_threads,
            )
        )

        self._rollout_generator = build_rollout_generator(
            rollout_fn=self.rollout_fn,
            serving_backend=self._serving_backend,
            config=self.rollout_config,
        )
        self._reward = UserDefinedReward(
            reward_fn=self.reward_fn,
            config=self.reward_config,
        )

        self._controller = GRPOController(
            config=self.controller_config,
            grpo_config=self.grpo_config,
            actor_backend=self._actor_backend,
            reference_backend=self._reference_backend,
            serving_backend=self._serving_backend,
            reward_fn=self._reward,
            rollout_generator=self._rollout_generator,
            run_logger=None,
            metrics_sink=self._metrics_sink,
            on_step_complete=self._on_controller_step_complete,
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
        return runtime_support.build_model_load_event(
            component=component,
            duration_seconds=duration_seconds,
            device=device,
            cpu_threads=cpu_threads,
        )

    def _initialize_admin(self) -> None:
        """Start the runtime-owned admin registry and HTTP server when enabled."""
        self._admin_registry = AdminRegistry()
        self._admin_registry.register(self._runtime_admin_objects)
        self._admin_registry.register(self._actor_backend_admin_objects)
        self._admin_registry.register(self._actor_child_admin_objects)
        self._admin_registry.register(self._reference_backend_admin_objects)
        self._admin_registry.register(self._reference_child_admin_objects)
        self._admin_registry.register(self._serving_backend_admin_objects)
        self._admin_registry.register(self._serving_child_admin_objects)

        if not self.admin_enabled:
            return

        self._admin_server = AdminServer(
            self._admin_registry,
            host=self.admin_host,
            port=self.admin_port,
        )
        self.admin_base_url = self._admin_server.start()
        self._emit_bootstrap_stage("ready", "admin", f"url={self.admin_base_url}")

    def _runtime_labels(self) -> dict[str, str]:
        """Build shared labels for all runtime-owned admin objects."""
        labels = {
            "flashrl.dev/runtime": "flashrl",
            "flashrl.dev/model-name": self.actor_config.model_name,
            "flashrl.dev/actor-backend": self.actor_config.backend,
            "flashrl.dev/serving-backend": self.serving_config.backend,
        }
        if self.reference_config is not None:
            labels["flashrl.dev/reference-backend"] = self.reference_config.backend
        return labels

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
                    "modelName": self.actor_config.model_name,
                    "referenceConfigured": self.reference_config is not None,
                    "adminBaseUrl": self.admin_base_url,
                },
                status={
                    "phase": self._runtime_phase,
                    "startedAt": self._runtime_created_at,
                    "currentEpoch": self._admin_current_epoch(),
                    "currentStep": self._admin_current_step(),
                    "lastError": self._last_runtime_error,
                    "memory": self._runtime_memory_snapshot(),
                },
            )
        ]

    def _actor_backend_admin_objects(self) -> list[dict[str, Any]]:
        """Return the actor backend admin object."""
        if self._actor_backend is None:
            return []
        return [
            build_admin_object(
                "ActorBackend",
                "actor-backend",
                uid=self._runtime_object_uid("actor-backend"),
                created_at=self._runtime_created_at,
                labels=self._runtime_labels(),
                spec={
                    "backend": self.actor_config.backend,
                    "modelName": self.actor_config.model_name,
                    "device": self.actor_config.device or "auto",
                    "dtype": self.actor_config.dtype,
                    "numThreads": self.actor_config.num_threads,
                    "dpSize": self.actor_config.dp_size,
                },
                status={
                    "phase": self._component_phase(),
                    "device": str(self._actor_backend.device),
                    "optimizer": self._actor_backend.optimizer_name,
                    "loaded": True,
                    "worldSize": self._actor_backend.world_size,
                    "memory": capture_memory_snapshot(self._actor_backend.device),
                },
            )
        ]

    def _actor_child_admin_objects(self) -> list[dict[str, Any]]:
        """Return backend-owned actor child objects."""
        if self._actor_backend is None:
            return []
        return self._actor_backend.list_admin_objects()

    def _reference_backend_admin_objects(self) -> list[dict[str, Any]]:
        """Return the optional reference backend admin object."""
        if self._reference_backend is None or self.reference_config is None:
            return []
        return [
            build_admin_object(
                "ReferenceBackend",
                "reference-backend",
                uid=self._runtime_object_uid("reference-backend"),
                created_at=self._runtime_created_at,
                labels=self._runtime_labels(),
                spec={
                    "backend": self.reference_config.backend,
                    "modelName": self.reference_config.model_name,
                    "device": self.reference_config.device or "auto",
                    "dtype": self.reference_config.dtype,
                    "numThreads": self.reference_config.num_threads,
                    "dpSize": self.reference_config.dp_size,
                },
                status={
                    "phase": self._component_phase(),
                    "device": str(self._reference_backend.device),
                    "loaded": True,
                    "worldSize": self._reference_backend.world_size,
                    "memory": capture_memory_snapshot(self._reference_backend.device),
                },
            )
        ]

    def _reference_child_admin_objects(self) -> list[dict[str, Any]]:
        """Return backend-owned reference child objects."""
        if self._reference_backend is None:
            return []
        return self._reference_backend.list_admin_objects()

    def _serving_backend_admin_objects(self) -> list[dict[str, Any]]:
        """Return the serving backend admin object."""
        if self._serving_backend is None:
            return []

        child_objects = self._serving_child_admin_objects()
        sync_status = self._serving_weight_sync_status()
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
                    "activeWeightVersion": sync_status.get("activeWeightVersion"),
                    "pendingWeightVersion": sync_status.get("pendingWeightVersion"),
                    "lastSuccessfulSyncAt": sync_status.get("lastSuccessfulSyncAt"),
                    "syncHealthy": sync_status.get("syncHealthy"),
                    "lastSyncError": sync_status.get("lastSyncError"),
                    "memory": capture_memory_snapshot(self._serving_backend.device),
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

    def _serving_weight_sync_status(self) -> dict[str, Any]:
        """Return serving sync status with a compatibility fallback for custom backends."""
        if self._serving_backend is None:
            return {
                "activeWeightVersion": None,
                "pendingWeightVersion": None,
                "lastSuccessfulSyncAt": None,
                "syncHealthy": None,
                "lastSyncError": None,
            }

        status_getter = getattr(self._serving_backend, "weight_sync_status", None)
        if callable(status_getter):
            try:
                status = status_getter()
            except Exception:
                status = None
            if isinstance(status, dict):
                return status

        active = self._serving_current_weight_version()
        return {
            "activeWeightVersion": active.model_dump() if isinstance(active, WeightVersionInfo) else None,
            "pendingWeightVersion": None,
            "lastSuccessfulSyncAt": (
                active.activated_at if isinstance(active, WeightVersionInfo) else None
            ),
            "syncHealthy": isinstance(active, WeightVersionInfo),
            "lastSyncError": None,
        }

    def _serving_current_weight_version(self) -> WeightVersionInfo | None:
        """Return the active serving version when the backend exposes it."""
        if self._serving_backend is None:
            return None
        getter = getattr(self._serving_backend, "current_weight_version", None)
        if not callable(getter):
            return None
        try:
            weight_version = getter()
        except Exception:
            return None
        if isinstance(weight_version, WeightVersionInfo):
            return weight_version
        return None

    def _component_phase(self) -> str:
        """Map the runtime phase to a generic component phase."""
        if self._runtime_phase == "Training":
            return "Ready"
        return self._runtime_phase

    def _runtime_memory_snapshot(self) -> dict[str, Any]:
        """Return the runtime-level memory snapshot for admin/status surfaces."""
        if self._actor_backend is not None:
            return capture_memory_snapshot(self._actor_backend.device)
        return capture_memory_snapshot(self.actor_config.device)

    def _admin_current_epoch(self) -> int:
        """Return the user-facing epoch number for the runtime object."""
        if self._controller is None:
            return 0
        if self._controller.total_steps == 0 and self._runtime_phase == "Ready":
            return 0
        return self._controller.current_epoch + 1

    def _admin_current_step(self) -> int:
        """Return the current training step for the runtime object."""
        if self._controller is None:
            return 0
        return self._controller.total_steps

    def _learner_backends(self) -> list[TrainingBackend]:
        """Return the initialized learner backends in stable actor/reference order."""
        backends: list[TrainingBackend] = []
        if self._actor_backend is not None:
            backends.append(self._actor_backend)
        if self._reference_backend is not None:
            backends.append(self._reference_backend)
        return backends

    def close(self) -> None:
        """Release runtime-owned resources."""
        if self._closed:
            return
        self._closed = True
        if self._metrics_sink is not None:
            self._metrics_sink.close()
        if self._run_logger is not None:
            self._run_logger.close()
            self._run_logger = None
        self._runtime_phase = "Closing"
        try:
            if self._serving_backend is not None:
                self._serving_backend.close()
            for backend in reversed(self._learner_backends()):
                backend.close()
        finally:
            self._runtime_phase = "Closed"
            if self._admin_server is not None:
                self._admin_server.close()
                self._admin_server = None
            if self._startup_artifact_dir is not None:
                shutil.rmtree(self._startup_artifact_dir, ignore_errors=True)
                self._startup_artifact_dir = None
        self._serving_backend = None
        self._actor_backend = None
        self._reference_backend = None

    def pause_serving(self) -> None:
        """Pause serving (if supported by backend)."""
        if self._serving_backend is None:
            return

        # Delegate to backend's pause method if it exists
        if hasattr(self._serving_backend, "pause_inference"):
            self._serving_backend.pause_inference()
        else:
            # Alternative approach for backends without pause support
            pass

    def resume_serving(self) -> None:
        """Resume serving (if supported by backend)."""
        if self._serving_backend is None:
            return

        # Delegate to backend's resume method if it exists
        if hasattr(self._serving_backend, "resume_inference"):
            self._serving_backend.resume_inference()
        else:
            # Alternative approach for backends without resume support
            pass

    def _resolve_train_dataset(
        self,
        dataset: list[Prompt] | list[str] | None,
    ) -> list[Prompt]:
        """Resolve the explicit or configured dataset into normalized prompts."""
        if dataset is None:
            if self._dataset_loader is None:
                raise ValueError(
                    "FlashRL.train() requires a dataset unless the controller was created with "
                    "a configured dataset_loader."
                )
            dataset = self._dataset_loader()
        return runtime_support.normalize_dataset(dataset)

    def _reset_run_lifecycle_totals(self) -> None:
        """Reset run totals while preserving checkpoint timings and managed carry-over."""
        checkpoint_totals = {
            key: value
            for key, value in self._run_lifecycle_totals.items()
            if key.startswith("checkpoint_")
        }
        self._run_lifecycle_totals = self._merge_float_mappings(
            self._restored_run_lifecycle_totals,
            self._runtime_bootstrap_totals,
            checkpoint_totals,
        )

    def _merge_float_mappings(self, *mappings: dict[str, Any]) -> dict[str, float]:
        """Accumulate float-like values from one or more mappings."""
        merged: dict[str, float] = {}
        for mapping in mappings:
            for key, value in mapping.items():
                try:
                    merged[key] = merged.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue
        return merged

    def _build_train_run_state(self, dataset: list[Prompt]) -> TrainRunState:
        """Compute the per-run dataset and step metadata once."""
        return build_train_run_state(
            dataset,
            controller_config=self.controller_config,
            grpo_config=self.grpo_config,
        )

    def _open_run_logger(self, run_state: TrainRunState) -> RunLogger:
        """Open and initialize the run-scoped logger for one training invocation."""
        assert self._actor_backend is not None
        assert self._serving_backend is not None
        set_log_dir = getattr(self._serving_backend, "set_log_dir", None)
        run_logger = open_shared_run_logger(
            logging_config=self.logging_config,
            model_name=self.actor_config.model_name,
            actor_config=self.actor_config,
            reference_config=self.reference_config,
            serving_config=self.serving_config,
            controller_config=self.controller_config,
            grpo_config=self.grpo_config,
            run_state=run_state,
            actor_device=str(self._actor_backend.device),
            reference_device=(
                str(self._reference_backend.device) if self._reference_backend is not None else None
            ),
            serving_device=str(self._serving_backend.device),
            admin_base_url=self.admin_base_url,
            bootstrap_console_lines=self._runtime_bootstrap_console_lines,
            bootstrap_events=self._runtime_bootstrap_events,
            managed_resume=self._managed_resume,
            managed_resume_load_seconds=self._managed_resume_load_seconds,
            resumed_epoch=self._controller.current_epoch + 1,
            resumed_step=self._controller.total_steps,
            serving_log_dir_setter=set_log_dir,
        )
        self._run_logger = run_logger
        run_state.run_logger = run_logger
        return run_logger

    def _prepare_controller_for_run(self, run_logger: RunLogger) -> None:
        """Apply resume/reset behavior and attach the run logger to the controller."""
        assert self._controller is not None
        if not self._resume_from_checkpoint:
            self._controller.reset_state()
        self._controller.attach_run_logger(run_logger)

    def _prepare_managed_resume(self) -> RestoredCheckpoint | None:
        """Load one configured managed checkpoint before opening the run logger."""
        assert self._controller is not None
        assert self._actor_backend is not None
        checkpoint_path = self._checkpoint_manager.resolve_resume_path()
        if checkpoint_path is None:
            self._managed_resume = None
            self._restored_run_lifecycle_totals = {}
            self._managed_resume_load_seconds = 0.0
            return None
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        def load_managed_checkpoint():
            for backend in self._learner_backends():
                backend.barrier()
            try:
                return self._controller.load_checkpoint_with_metadata(str(checkpoint_path))
            finally:
                for backend in reversed(self._learner_backends()):
                    backend.barrier()

        (_, checkpoint_metadata), duration_seconds = timed_call(load_managed_checkpoint)
        managed_resume = self._checkpoint_manager.build_restored_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
        )
        self._managed_resume = managed_resume
        self._managed_resume_load_seconds = duration_seconds
        self._resume_from_checkpoint = True
        self._restored_run_lifecycle_totals = dict(managed_resume.lifecycle_totals)
        self._run_lifecycle_totals["checkpoint_load_seconds"] = (
            self._run_lifecycle_totals.get("checkpoint_load_seconds", 0.0)
            + duration_seconds
        )
        self._checkpoint_manager.mark_resume_consumed()
        return managed_resume

    def _build_checkpoint_metadata(self, *, trigger: str) -> dict[str, Any]:
        """Capture the metadata needed for manual restores and managed append-resume."""
        run_metadata: dict[str, Any] = {}
        run_logger_state: dict[str, Any] = {}
        serving_metadata: dict[str, Any] = {}
        if self._run_logger is not None:
            run_metadata = {
                "run_id": self._run_logger.run_id,
                "run_index": self._run_logger.run_index,
                "run_dir": str(self._run_logger.run_dir),
            }
            run_logger_state = self._run_logger.export_state()
        if self._serving_backend is not None:
            export_state = getattr(self._serving_backend, "export_weight_version_state", None)
            if callable(export_state):
                serving_metadata = {
                    "weight_version_state": export_state(),
                }
            else:
                active = self._serving_current_weight_version()
                next_version_id = (
                    active.version_id + 1 if isinstance(active, WeightVersionInfo) else 0
                )
                serving_metadata = {
                    "weight_version_state": {
                        "schema_version": 1,
                        "next_version_id": next_version_id,
                        "active_weight_version": (
                            active.model_dump() if isinstance(active, WeightVersionInfo) else None
                        ),
                        "last_successful_sync_at": (
                            active.activated_at if isinstance(active, WeightVersionInfo) else None
                        ),
                    }
                }
        assert self._controller is not None
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "trigger": trigger,
            "saved_epoch": self._controller.current_epoch,
            "saved_step": self._controller.total_steps,
            "lifecycle_totals": dict(self._run_lifecycle_totals),
            "run": run_metadata,
            "run_logger_state": run_logger_state,
            "serving": serving_metadata,
        }

    def _save_checkpoint_to_path(
        self,
        path: str | Path,
        *,
        trigger: str,
        update_latest: bool,
    ) -> float:
        """Save one checkpoint, optionally updating the managed latest manifest."""
        assert self._controller is not None
        assert self._actor_backend is not None
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid4().hex}.tmp")

        def save_checkpoint_file() -> None:
            for backend in self._learner_backends():
                backend.barrier()
            try:
                if self._actor_backend.is_primary:
                    self._controller.save_checkpoint(
                        str(temp_path),
                        checkpoint_metadata=self._build_checkpoint_metadata(trigger=trigger),
                    )
                    os.replace(temp_path, checkpoint_path)
                for backend in reversed(self._learner_backends()):
                    backend.barrier()
            finally:
                temp_path.unlink(missing_ok=True)

        _, duration_seconds = timed_call(save_checkpoint_file)
        self._run_lifecycle_totals["checkpoint_save_seconds"] = (
            self._run_lifecycle_totals.get("checkpoint_save_seconds", 0.0)
            + duration_seconds
        )
        if self._run_logger is not None:
            self._run_logger.log_checkpoint(
                "save",
                str(checkpoint_path),
                epoch=self._controller.current_epoch + 1,
                step=self._controller.total_steps,
                duration_seconds=duration_seconds,
                trigger=trigger,
            )
            if update_latest:
                self._checkpoint_manager.write_latest_manifest(
                    run_dir=self._run_logger.run_dir,
                    checkpoint_path=checkpoint_path,
                    epoch=self._controller.current_epoch + 1,
                    step=self._controller.total_steps,
                    trigger=trigger,
                    run_id=self._run_logger.run_id,
                    run_index=self._run_logger.run_index,
                )
        return duration_seconds

    def _on_controller_step_complete(self, step_context: dict[str, Any]) -> None:
        """Persist managed interval checkpoints after completed training steps."""
        if self._run_logger is None:
            return
        step = int(step_context.get("step", 0))
        if not self._checkpoint_manager.should_save_interval(step):
            return
        checkpoint_path = self._checkpoint_manager.interval_checkpoint_path(
            run_dir=self._run_logger.run_dir,
            step=step,
        )
        self._save_checkpoint_to_path(
            checkpoint_path,
            trigger="interval",
            update_latest=True,
        )

    def _save_final_checkpoint_if_enabled(self) -> None:
        """Persist the managed final checkpoint for a completed run when configured."""
        if not self.checkpointing_config.save_on_run_end or self._run_logger is None:
            return
        checkpoint_path = self._checkpoint_manager.final_checkpoint_path(
            run_dir=self._run_logger.run_dir
        )
        self._save_checkpoint_to_path(
            checkpoint_path,
            trigger="final",
            update_latest=True,
        )

    def _start_run_metrics(self, run_logger: RunLogger) -> None:
        """Open the metrics sink for the current run when enabled."""
        start_run_metrics(self._metrics_sink, run_logger=run_logger)

    def _execute_training_loop(self, dataset: list[Prompt]) -> None:
        """Mark the runtime as training and delegate to the controller."""
        assert self._controller is not None
        self._last_runtime_error = None
        self._runtime_phase = "Training"
        self._controller.train(dataset)

    def _handle_train_failure(self, exc: BaseException) -> None:
        """Preserve the current runtime and logging failure behavior."""
        assert self._controller is not None
        self._runtime_phase = (
            "Interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "Failed"
        )
        self._last_runtime_error = f"{type(exc).__name__}: {exc}"
        active_context = self._controller.active_step_context
        if self._run_logger is not None:
            if not bool(getattr(exc, "_flashrl_logged", False)):
                context: dict[str, Any] = {
                    "stage": "train",
                    "step": (
                        active_context.step
                        if active_context is not None
                        else self._controller.total_steps
                    ),
                    "epoch": (
                        active_context.epoch
                        if active_context is not None
                        else self._controller.current_epoch + 1
                    ),
                }
                learner_stage = getattr(exc, "stage_name", None)
                if learner_stage is not None:
                    context["learner_stage"] = str(learner_stage)
                memory_snapshot = getattr(exc, "memory_snapshot", None)
                if isinstance(memory_snapshot, dict):
                    context["memory"] = memory_snapshot
                reason_tags = getattr(exc, "reason_tags", None)
                if isinstance(reason_tags, list) and reason_tags:
                    context["memory_reason_tags"] = [str(tag) for tag in reason_tags]
                self._run_logger.log_exception(exc, context=context)
        if self._serving_backend is not None:
            self._serving_backend.close()

    def _finalize_train_run(self, *, status: str, started_at: float) -> None:
        """Finalize metrics, logger state, and lifecycle totals for one run."""
        assert self._controller is not None
        final_status = status
        self._run_lifecycle_totals["training_loop_seconds"] = time.perf_counter() - started_at
        try:
            if status == "completed":
                self._save_final_checkpoint_if_enabled()
                self._runtime_phase = "Ready"
        except (KeyboardInterrupt, SystemExit) as exc:
            final_status = "interrupted"
            self._handle_train_failure(exc)
            raise
        except Exception as exc:
            final_status = "failed"
            self._handle_train_failure(exc)
            raise
        finally:
            self._controller.attach_run_logger(None)
            self._resume_from_checkpoint = False
            self._managed_resume = None
            self._managed_resume_load_seconds = 0.0
            self._restored_run_lifecycle_totals = {}
            finish_run_observers(
                run_logger=self._run_logger,
                metrics_sink=self._metrics_sink,
                status=final_status,
                total_steps=self._controller.total_steps,
                lifecycle_totals=self._run_lifecycle_totals,
            )

    def train(self, dataset: list[Prompt] | list[str] | None = None) -> None:
        """Train on dataset or use the configured dataset loader."""
        run_state = self._build_train_run_state(self._resolve_train_dataset(dataset))
        self._prepare_managed_resume()
        self._reset_run_lifecycle_totals()
        run_logger = self._open_run_logger(run_state)
        status = "completed"
        assert self._controller is not None
        self._prepare_controller_for_run(run_logger)
        self._start_run_metrics(run_logger)
        training_loop_started_at = time.perf_counter()
        try:
            self._execute_training_loop(run_state.dataset)
        except (KeyboardInterrupt, SystemExit) as exc:
            status = "interrupted"
            self._handle_train_failure(exc)
            raise
        except Exception as exc:
            status = "failed"
            self._handle_train_failure(exc)
            raise
        finally:
            self._finalize_train_run(status=status, started_at=training_loop_started_at)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        self._save_checkpoint_to_path(path, trigger="manual", update_latest=False)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        assert self._controller is not None
        assert self._actor_backend is not None

        def load_manual_checkpoint() -> None:
            for backend in self._learner_backends():
                backend.barrier()
            try:
                self._controller.load_checkpoint(path)
            finally:
                for backend in reversed(self._learner_backends()):
                    backend.barrier()

        _, duration_seconds = timed_call(load_manual_checkpoint)
        self._resume_from_checkpoint = True
        self._run_lifecycle_totals["checkpoint_load_seconds"] = (
            self._run_lifecycle_totals.get("checkpoint_load_seconds", 0.0)
            + duration_seconds
        )
        if self._run_logger is not None:
            self._run_logger.log_checkpoint(
                "load",
                path,
                epoch=self._controller.current_epoch + 1,
                step=self._controller.total_steps,
                duration_seconds=duration_seconds,
                trigger="manual",
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
