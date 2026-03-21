"""Reference agent harness example built from generic agent primitives."""

from flashrl.examples.agent_harness.config import AgentHarnessConfig
from flashrl.examples.agent_harness.dataset import (
    build_eval_dataset,
    build_train_dataset,
    reward_fn,
)
from flashrl.examples.agent_harness.evaluation import (
    evaluate_model,
    summarize_rollouts,
)
from flashrl.examples.agent_harness.harness import (
    build_agent_harness,
)

__all__ = [
    "AgentHarnessConfig",
    "build_agent_harness",
    "build_train_dataset",
    "build_eval_dataset",
    "reward_fn",
    "evaluate_model",
    "summarize_rollouts",
]
