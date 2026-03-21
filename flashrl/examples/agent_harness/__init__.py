"""Reference agent harness example built from generic agent primitives."""

from flashrl.examples.agent_harness.config import CodingHarnessConfig
from flashrl.examples.agent_harness.harness import (
    build_coding_agent,
    build_coding_eval_dataset,
    build_coding_reward_fn,
    build_coding_train_dataset,
    evaluate_rollouts,
)

__all__ = [
    "CodingHarnessConfig",
    "build_coding_agent",
    "build_coding_train_dataset",
    "build_coding_eval_dataset",
    "build_coding_reward_fn",
    "evaluate_rollouts",
]
