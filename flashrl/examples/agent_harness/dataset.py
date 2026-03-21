"""Dataset and reward helpers for the reference agent harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from flashrl.framework.data_models import Prompt, RewardOutput, RolloutOutput

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EXAMPLE_DIR / "fixtures"


@dataclass(frozen=True)
class RepoTask:
    """One deterministic repo-inspection task used by the agent harness example."""

    task_id: str
    split: Literal["train", "eval"]
    repo_name: str
    question: str
    expected_answer: str
    preload_skills: tuple[str, ...] = ()


TASKS: tuple[RepoTask, ...] = (
    RepoTask(
        task_id="inventory-version",
        split="train",
        repo_name="inventory_repo",
        question=(
            "Inspect the repository and answer with the raw APP_VERSION value only. "
            "@skill:repo_triage"
        ),
        expected_answer="v3",
        preload_skills=("repo_triage",),
    ),
    RepoTask(
        task_id="inventory-port",
        split="train",
        repo_name="inventory_repo",
        question="Inspect the repository and answer with the default port integer only.",
        expected_answer="8080",
    ),
    RepoTask(
        task_id="service-header",
        split="train",
        repo_name="service_repo",
        question=(
            "Inspect the repository and answer with the function name that builds the auth header only. "
            "@skill:repo_triage"
        ),
        expected_answer="build_auth_header",
        preload_skills=("repo_triage",),
    ),
    RepoTask(
        task_id="service-retry",
        split="train",
        repo_name="service_repo",
        question="Inspect the repository and answer with the retry count integer only.",
        expected_answer="3",
    ),
    RepoTask(
        task_id="inventory-combo",
        split="eval",
        repo_name="inventory_repo",
        question="Inspect the repository and answer exactly `version=v3;port=8080`.",
        expected_answer="version=v3;port=8080",
        preload_skills=("repo_triage",),
    ),
    RepoTask(
        task_id="service-combo",
        split="eval",
        repo_name="service_repo",
        question=(
            "Inspect the repository and answer exactly `retry=3;header=build_auth_header`. "
            "If the work can be decomposed, you may delegate one sub-question with run_subagent."
        ),
        expected_answer="retry=3;header=build_auth_header",
        preload_skills=("repo_triage",),
    ),
)


def build_train_dataset(limit: int | None = None) -> list[Prompt]:
    return _build_dataset(split="train", limit=limit)


def build_eval_dataset(limit: int | None = None) -> list[Prompt]:
    return _build_dataset(split="eval", limit=limit)


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    prompt_metadata = dict(rollout.metadata.get("prompt_metadata", {}))
    expected_answer = _normalize_answer(str(prompt_metadata.get("expected_answer", "")))
    answer = _normalize_answer(str(rollout.text))
    accuracy_pass = bool(answer == expected_answer and expected_answer)
    return RewardOutput(
        reward=1.0 if accuracy_pass else 0.0,
        metadata={
            "expected_answer": expected_answer,
            "normalized_answer": answer,
            "accuracy_pass": accuracy_pass,
        },
    )


def _build_dataset(split: Literal["train", "eval"], limit: int | None) -> list[Prompt]:
    prompts: list[Prompt] = []
    for task in TASKS:
        if task.split != split:
            continue
        prompts.append(_build_prompt(task, split=split))
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def _build_prompt(task: RepoTask, *, split: Literal["train", "eval"]) -> Prompt:
    return Prompt(
        text=task.question,
        metadata={
            "task_id": task.task_id,
            "source": "flashrl/examples/agent_harness",
            "split": split,
            "repo_root": str((FIXTURES_DIR / "repos" / task.repo_name).resolve()),
            "expected_answer": task.expected_answer,
            "preload_skills": list(task.preload_skills),
            "verifier": "string_exact",
        },
    )


def _normalize_answer(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized.casefold()
