"""Internal helpers for runtime config and dataset normalization."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from flashrl.framework.config import (
    CommonConfig,
    GrpoConfig,
    ModelConfig,
    RolloutConfig,
    RunConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)
from flashrl.framework.data_models import Prompt


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

SERVING_MODEL_FIELDS = {
    "backend",
    "runtime_python",
    "num_replicas",
    "vllm_args",
    "debug_live_rollout",
}

TRAINING_MODEL_FIELDS = {
    "backend",
    "dp_size",
    "fsdp2",
}


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
    run_config: RunConfig | dict[str, Any] | None,
) -> RunConfig | None:
    """Normalize the optional profile input into one RunConfig object."""
    if config_path is not None and run_config is not None:
        raise ValueError("Pass only one of config_path or run_config when constructing FlashRL.")
    if config_path is not None:
        return RunConfig.from_yaml(config_path)
    if run_config is None:
        return None
    if isinstance(run_config, RunConfig):
        return run_config
    if isinstance(run_config, dict):
        return RunConfig.from_dict(run_config)
    raise TypeError("run_config must be a RunConfig, dict, or None.")


def build_model_config(
    *,
    common: CommonConfig | None,
    section: CommonConfig,
    config_cls: type[ModelConfig] | type[ServingConfig] | type[TrainingConfig],
    section_name: str,
) -> ModelConfig | ServingConfig | TrainingConfig:
    """Merge optional common defaults with one model-copy section."""
    merged = {}
    if common is not None:
        merged.update(common.model_dump(exclude_none=True))
    include_fields = set(COMMON_MODEL_SECTION_FIELDS)
    if config_cls is ServingConfig:
        include_fields.update(SERVING_MODEL_FIELDS)
    if config_cls is TrainingConfig:
        include_fields.update(TRAINING_MODEL_FIELDS)
    merged.update(section.model_dump(include=include_fields, exclude_none=True))

    if not merged.get("model_name"):
        raise ValueError(
            f"Run config requires '{section_name}.model_name' or 'common.model_name' after merge."
        )
    return config_cls(**merged)


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
    return TrainerConfig(
        learning_rate=run_config.training.learning_rate,
        batch_size=run_config.training.batch_size,
        max_epochs=run_config.training.max_epochs,
        seed=run_config.training.seed,
        shuffle_each_epoch=run_config.training.shuffle_each_epoch,
    )


def build_runtime_profile(
    run_config: RunConfig,
) -> tuple[TrainingConfig, ServingConfig, TrainerConfig]:
    """Resolve the merged training/serving configs from one RunConfig."""
    training_config = build_model_config(
        common=run_config.common,
        section=run_config.training,
        config_cls=TrainingConfig,
        section_name="training",
    )
    serving_config = build_model_config(
        common=run_config.common,
        section=run_config.serving,
        config_cls=ServingConfig,
        section_name="serving",
    )
    trainer_config = build_trainer_config(run_config)
    return training_config, serving_config, trainer_config


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
