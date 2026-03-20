"""Internal helpers for runtime config and dataset normalization."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from flashrl.framework.config import (
    AdminConfig,
    BuilderSpec,
    GrpoConfig,
    RolloutConfig,
    RunConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)
from flashrl.framework.data_models import Prompt


def resolve_import(import_string: str) -> Any:
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


def resolve_hook_target(binding: str | BuilderSpec) -> Any:
    """Resolve one hook binding into its imported target."""
    if isinstance(binding, BuilderSpec):
        return resolve_import(binding.import_path)
    return resolve_import(str(binding))


def instantiate_hook(binding: str | BuilderSpec) -> Any:
    """Instantiate one hook binding for local or platform runtime use."""
    target = resolve_hook_target(binding)
    if isinstance(binding, BuilderSpec):
        if not callable(target):
            raise TypeError(
                "Structured hook bindings must resolve to a callable builder."
            )
        return target(**binding.kwargs)
    return target


def normalize_dataset(dataset: list[Prompt] | list[str]) -> list[Prompt]:
    """Normalize string datasets into Prompt objects."""
    normalized: list[Prompt] = []
    for item in dataset:
        if isinstance(item, Prompt):
            normalized.append(item)
        else:
            normalized.append(Prompt(text=str(item)))
    return normalized


def load_run_config(
    *,
    config_path: str | Path | None,
    config_profile: str | None = None,
    run_config: RunConfig | dict[str, Any] | None,
) -> RunConfig | None:
    """Normalize the optional profile input into one RunConfig object."""
    if config_path is not None and run_config is not None:
        raise ValueError("Pass only one of config_path or run_config when constructing FlashRL.")
    if config_path is not None:
        return RunConfig.from_yaml(config_path, profile=config_profile)
    if run_config is None:
        return None
    if isinstance(run_config, RunConfig):
        return run_config
    if isinstance(run_config, dict):
        return RunConfig.from_dict(run_config)
    raise TypeError("run_config must be a RunConfig, dict, or None.")


def build_rollout_config(grpo_config: GrpoConfig) -> RolloutConfig:
    """Build the internal rollout config from GRPO sampling knobs."""
    return RolloutConfig(
        max_new_tokens=grpo_config.max_new_tokens,
        temperature=grpo_config.temperature,
        top_p=grpo_config.top_p,
        top_k=grpo_config.top_k,
        do_sample=grpo_config.do_sample,
    )


def build_trainer_config(run_config: RunConfig) -> TrainerConfig:
    """Extract trainer loop settings from one top-level RunConfig."""
    return run_config.trainer.model_copy(deep=True)


def build_runtime_profile(
    run_config: RunConfig,
) -> tuple[TrainingConfig, TrainingConfig | None, ServingConfig, TrainerConfig, AdminConfig]:
    """Resolve the runtime profile directly from the explicit role config."""
    return (
        run_config.actor.model_copy(deep=True),
        run_config.reference.model_copy(deep=True) if run_config.reference is not None else None,
        run_config.serving.model_copy(deep=True),
        build_trainer_config(run_config),
        run_config.admin.model_copy(deep=True),
    )


def build_model_load_event(
    *,
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
