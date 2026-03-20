"""Unit tests for the strict reasoning example."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from flashrl.framework.data_models import Conversation, Message, Prompt, RolloutOutput

pytestmark = pytest.mark.unit


def load_script_module(
    module_name: str,
    relative_path: str,
    *,
    aliases: tuple[str, ...] = (),
):
    """Load one packaged example module from a file path for tests."""
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    for alias in aliases:
        sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


reasoning_example = load_script_module(
    "flashrl_reasoning_math_train",
    "flashrl/examples/math/train.py",
    aliases=("flashrl.examples.math.train",),
)
reasoning_eval = load_script_module(
    "flashrl_reasoning_math_eval",
    "flashrl/examples/math/eval.py",
    aliases=("flashrl.examples.math.eval",),
)
evaluate_model = reasoning_eval.evaluate_model


def make_prompt(
    *,
    problem: str = "Janet has 15 apples and buys 27 more. How many apples does she have now?",
    final_answer: str = "42",
    source: str = "openai/gsm8k",
) -> Prompt:
    """Build one strict reasoning prompt with GSM8K-style metadata."""
    return Prompt(
        text=reasoning_example.render_math_prompt(problem),
        metadata={
            "task_id": "gsm8k-train-000000",
            "source": source,
            "split": "train",
            "problem": problem,
            "final_answer": final_answer,
            "verifier": "numeric_exact",
        },
    )


def make_rollout(
    prompt: Prompt,
    response_text: str,
    *,
    finish_reason: str | None = None,
) -> RolloutOutput:
    """Build a minimal rollout with embedded prompt metadata."""
    metadata = {"prompt_metadata": dict(prompt.metadata)}
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    return RolloutOutput(
        text=response_text,
        log_prob=-0.1,
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[4, 5, 6],
        response_token_logprobs=[-0.1, -0.1, -0.1],
        metadata=metadata,
        conversation=Conversation(
            messages=[
                Message(role="user", content=prompt.text),
                Message(role="assistant", content=response_text),
            ]
        ),
    )


def test_render_math_prompt_has_no_system_prompt_and_enforces_contract() -> None:
    """The example prompt should be a user-only format contract."""
    prompt = reasoning_example.render_math_prompt("What is 15 + 27?")

    assert "system" not in prompt.lower()
    assert "<think>...</think>" in prompt
    assert "<answer>...</answer>" in prompt
    assert "Do not output any text before <think> or after </answer>." in prompt
    assert "Problem: What is 15 + 27?" in prompt


def test_build_math_train_dataset_loads_gsm8k_rows_into_prompt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset building should render prompts and preserve verifier metadata."""

    def fake_load(split: str, *, limit=None):
        assert split == reasoning_example.DEFAULT_GSM8K_TRAIN_SPLIT
        assert limit == 3
        return [
            {
                "task_id": "gsm8k-train-000001",
                "source": "openai/gsm8k",
                "split": split,
                "problem": "What is 15 + 27?",
                "final_answer": "42",
            }
        ]

    monkeypatch.setattr(reasoning_example, "_load_gsm8k_split", fake_load)

    dataset = reasoning_example.build_math_train_dataset(dataset="gsm8k", limit=3)

    assert len(dataset) == 1
    assert "Problem: What is 15 + 27?" in dataset[0].text
    assert dataset[0].metadata == {
        "task_id": "gsm8k-train-000001",
        "source": "openai/gsm8k",
        "split": "train",
        "problem": "What is 15 + 27?",
        "final_answer": "42",
        "verifier": "numeric_exact",
    }


def test_build_math_train_dataset_selects_aime25_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIME25 selection should use the explicit AIME loader and test split."""
    seen_calls: list[tuple[str, int | None]] = []

    def fake_load(split: str, *, limit=None):
        seen_calls.append((split, limit))
        return [
            {
                "task_id": "aime25-test-000001",
                "source": "math-ai/aime25",
                "split": split,
                "problem": "Compute 20 + 22.",
                "final_answer": "42",
            }
        ]

    monkeypatch.setattr(reasoning_example, "_load_aime25_split", fake_load)

    dataset = reasoning_example.build_math_train_dataset(dataset="aime25", limit=2)

    assert seen_calls == [(reasoning_example.DEFAULT_AIME25_TRAIN_SPLIT, 2)]
    assert dataset[0].metadata["source"] == "math-ai/aime25"
    assert dataset[0].metadata["final_answer"] == "42"


@pytest.mark.parametrize(
    ("dataset_name", "build_fn_name", "split", "expected_source", "problem_field", "target_format"),
    [
        ("gsm8k", "build_math_train_dataset", "train", "openai/gsm8k", "question", "parse #### + numeric normalize"),
        ("gsm8k", "build_math_eval_dataset", "test", "openai/gsm8k", "question", "parse #### + numeric normalize"),
        ("aime25", "build_math_train_dataset", "test", "math-ai/aime25", "problem", "direct numeric normalize"),
        ("aime25", "build_math_eval_dataset", "test", "math-ai/aime25", "problem", "direct numeric normalize"),
    ],
)
def test_build_math_dataset_prints_compact_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_name: str,
    build_fn_name: str,
    split: str,
    expected_source: str,
    problem_field: str,
    target_format: str,
) -> None:
    """Dataset loading should print the selected split summary to stdout."""
    expected_split = split

    def fake_load_dataset(dataset_id: str, *args, split: str):
        assert split == expected_split
        if dataset_name == "gsm8k":
            assert dataset_id == "openai/gsm8k"
            assert args == ("main",)
            return [
                {
                    "id": "gsm8k-row-1",
                    "question": "What is 15 + 27?",
                    "answer": "work\n#### 42",
                },
                {
                    "id": "gsm8k-row-2",
                    "question": "What is 12 + 15 + 8?",
                    "answer": "work\n#### 35",
                },
            ]
        assert dataset_id == reasoning_example.DEFAULT_AIME25_HF_DATASET
        assert args == ()
        return [
            {
                "id": "aime25-row-1",
                "problem": "Compute 20 + 22.",
                "answer": "42",
            },
            {
                "id": "aime25-row-2",
                "problem": "Compute 30 + 5.",
                "answer": "35",
            },
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    build_fn = getattr(reasoning_example, build_fn_name)
    dataset = build_fn(dataset=dataset_name, limit=1)
    output = capsys.readouterr().out

    assert len(dataset) == 1
    assert (
        f"dataset  name={dataset_name}  source={expected_source}  split={split}  available=2  selected=1"
        in output
    )
    assert (
        f"format   problem_field={problem_field}  answer_field=answer  target={target_format}"
        in output
    )


def test_build_math_train_dataset_uses_only_explicit_limit_not_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset size should come only from the function argument, not env vars."""
    monkeypatch.setenv("FLASHRL_REASONING_TRAIN_LIMIT", "11")
    monkeypatch.setenv("FLASHRL_REASONING_DATASET_LIMIT", "17")
    seen_limits: list[int | None] = []

    def fake_load(split: str, *, limit=None):
        del split
        seen_limits.append(limit)
        return []

    monkeypatch.setattr(reasoning_example, "_load_gsm8k_split", fake_load)

    reasoning_example.build_math_train_dataset(dataset="gsm8k")
    reasoning_example.build_math_train_dataset(dataset="gsm8k", limit=5)

    assert seen_limits == [None, 5]


def test_build_math_eval_dataset_uses_only_explicit_limit_not_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out dataset size should also ignore the old env-var path."""
    monkeypatch.setenv("FLASHRL_REASONING_TEST_LIMIT", "13")
    seen_limits: list[int | None] = []

    def fake_load(split: str, *, limit=None):
        assert split == reasoning_example.DEFAULT_GSM8K_EVAL_SPLIT
        seen_limits.append(limit)
        return []

    monkeypatch.setattr(reasoning_example, "_load_gsm8k_split", fake_load)

    reasoning_example.build_math_eval_dataset(dataset="gsm8k")
    reasoning_example.build_math_eval_dataset(dataset="gsm8k", limit=7)

    assert seen_limits == [None, 7]


def test_build_math_eval_dataset_selects_aime25_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out AIME25 evaluation should use the same explicit split."""
    seen_calls: list[tuple[str, int | None]] = []

    def fake_load(split: str, *, limit=None):
        seen_calls.append((split, limit))
        return [
            {
                "task_id": "aime25-test-000001",
                "source": "math-ai/aime25",
                "split": split,
                "problem": "Compute 20 + 22.",
                "final_answer": "42",
            }
        ]

    monkeypatch.setattr(reasoning_example, "_load_aime25_split", fake_load)

    dataset = reasoning_example.build_math_eval_dataset(dataset="aime25", limit=4)

    assert seen_calls == [(reasoning_example.DEFAULT_AIME25_EVAL_SPLIT, 4)]
    assert dataset[0].metadata["source"] == "math-ai/aime25"


def test_load_gsm8k_split_falls_back_and_records_loaded_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader should try both HF ids and preserve the one that succeeded."""

    def fake_load_dataset(dataset_name: str, config_name: str, *, split: str):
        assert config_name == "main"
        if dataset_name == "openai/gsm8k":
            raise RuntimeError("first alias unavailable")
        assert dataset_name == "gsm8k"
        assert split == reasoning_example.DEFAULT_GSM8K_TRAIN_SPLIT
        return [
            {
                "id": "gsm8k-train-000001",
                "question": "What is 15 + 27?",
                "answer": "work\n#### 42",
            }
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    rows = reasoning_example._load_gsm8k_split(reasoning_example.DEFAULT_GSM8K_TRAIN_SPLIT)

    assert rows == [
        {
            "task_id": "gsm8k-train-000001",
            "source": "gsm8k",
            "split": "train",
            "problem": "What is 15 + 27?",
            "final_answer": "42",
        }
    ]


def test_load_aime25_split_maps_problem_answer_and_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIME25 rows should use direct problem/answer fields and numeric normalization."""

    def fake_load_dataset(dataset_name: str, *, split: str):
        assert dataset_name == reasoning_example.DEFAULT_AIME25_HF_DATASET
        assert split == reasoning_example.DEFAULT_AIME25_TRAIN_SPLIT
        return [
            {
                "id": "aime25-000001",
                "problem": "Compute 40 + 2.",
                "answer": "042.0",
            }
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    rows = reasoning_example._load_aime25_split(reasoning_example.DEFAULT_AIME25_TRAIN_SPLIT)

    assert rows == [
        {
            "task_id": "aime25-000001",
            "source": "math-ai/aime25",
            "split": "test",
            "problem": "Compute 40 + 2.",
            "final_answer": "42",
        }
    ]


def test_extract_math_target_answer_normalizes_gsm8k_final_answers() -> None:
    """Ground-truth parsing should read the `####` target and normalize commas."""
    assert reasoning_example._extract_math_target_answer("work\n#### 1,234") == "1234"
    assert reasoning_example._extract_math_target_answer("work\n#### 42.0") == "42"


def test_math_reward_cases_cover_accuracy_and_format_matrix() -> None:
    """Reward cases should cover the documented 1.1 / 1.0 / 0.1 / 0.0 matrix."""
    prompt = make_prompt()

    perfect = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think><answer>42</answer>",
        )
    )
    normalized = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think><answer>$42.00.</answer>",
        )
    )
    strict_wrong = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 41.</think><answer>41</answer>",
        )
    )
    trailing = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think><answer>42</answer>\nextra",
        )
    )
    duplicate_answer = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think><answer>42</answer><answer>42</answer>",
        )
    )
    empty_answer = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think><answer> </answer>",
        )
    )
    missing_answer = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think>",
        )
    )

    assert perfect.reward == pytest.approx(1.1)
    assert perfect.metadata["accuracy_pass"] is True
    assert perfect.metadata["format_pass"] is True
    assert normalized.reward == pytest.approx(1.1)
    assert normalized.metadata["normalized_answer"] == "42"
    assert normalized.metadata["accuracy_pass"] is True
    assert normalized.metadata["format_pass"] is True
    assert strict_wrong.reward == pytest.approx(0.1)
    assert strict_wrong.metadata["answer_parse_pass"] is True
    assert strict_wrong.metadata["accuracy_pass"] is False
    assert strict_wrong.metadata["format_pass"] is True
    assert trailing.reward == pytest.approx(1.0)
    assert trailing.metadata["accuracy_pass"] is True
    assert trailing.metadata["format_pass"] is False
    assert duplicate_answer.reward == pytest.approx(0.0)
    assert duplicate_answer.metadata["answer_parse_pass"] is False
    assert duplicate_answer.metadata["format_pass"] is False
    assert empty_answer.reward == pytest.approx(0.0)
    assert empty_answer.metadata["answer_parse_pass"] is False
    assert empty_answer.metadata["format_pass"] is False
    assert missing_answer.reward == pytest.approx(0.0)
    assert missing_answer.metadata["answer_parse_pass"] is False
    assert missing_answer.metadata["format_pass"] is False


def test_math_reward_marks_truncated_outputs_invalid_for_format() -> None:
    """Length truncation should remove only the format bonus when parsing still works."""
    prompt = make_prompt()
    reward = reasoning_example.math_reward_fn(
        make_rollout(
            prompt,
            "<think>Add 15 and 27 to get 42.</think><answer>42</answer>",
            finish_reason="length",
        )
    )

    assert reward.reward == pytest.approx(1.0)
    assert reward.metadata["truncated"] is True
    assert reward.metadata["format_pass"] is False
    assert reward.metadata["accuracy_pass"] is True


def test_extract_answer_block_requires_exactly_one_non_empty_block() -> None:
    """Answer parsing should reject missing, empty, and duplicate blocks."""
    assert reasoning_example._extract_answer_block("<answer>42</answer>") == "42"
    assert reasoning_example._extract_answer_block("<answer> </answer>") is None
    assert reasoning_example._extract_answer_block("42") is None
    assert reasoning_example._extract_answer_block("<answer>1</answer><answer>2</answer>") is None


def test_evaluate_model_reports_exact_match_format_and_truncation() -> None:
    """Held-out evaluation should reuse the strict reward parser and aggregate metrics."""
    prompts = [
        Prompt(
            text=reasoning_example.render_math_prompt("What is 15 + 27?"),
            metadata={
                "task_id": "gsm8k-test-000001",
                "source": "openai/gsm8k",
                "split": "test",
                "problem": "What is 15 + 27?",
                "final_answer": "42",
                "verifier": "numeric_exact",
            },
        ),
        Prompt(
            text=reasoning_example.render_math_prompt("What is 12 + 15 + 8?"),
            metadata={
                "task_id": "gsm8k-test-000002",
                "source": "openai/gsm8k",
                "split": "test",
                "problem": "What is 12 + 15 + 8?",
                "final_answer": "35",
                "verifier": "numeric_exact",
            },
        ),
    ]

    class FakeServingBackend:
        def __init__(self) -> None:
            self.generation_defaults: dict[str, object] = {}

        def set_generation_defaults(self, **kwargs) -> None:
            self.generation_defaults = dict(kwargs)

        def generate_batch(self, prompt_texts: list[str], **kwargs):
            del kwargs
            outputs = []
            for prompt_text in prompt_texts:
                if "15 + 27" in prompt_text:
                    outputs.append(
                        SimpleNamespace(
                            text="<think>Add 15 and 27 to get 42.</think><answer>42</answer>",
                            prompt_token_ids=[1, 2, 3],
                            response_token_ids=[4, 5, 6],
                            response_token_logprobs=[-0.1, -0.1, -0.1],
                            log_prob=-0.3,
                            metadata={"finish_reason": "stop"},
                        )
                    )
                else:
                    outputs.append(
                        SimpleNamespace(
                            text="<think>Add 12 and 15 first.",
                            prompt_token_ids=[1, 2, 3],
                            response_token_ids=[4, 5, 6],
                            response_token_logprobs=[-0.1, -0.1, -0.1],
                            log_prob=-0.3,
                            metadata={"finish_reason": "length"},
                        )
                    )
            return outputs

    flashrl = SimpleNamespace(
        _serving_backend=FakeServingBackend(),
        rollout_config=SimpleNamespace(max_new_tokens=64),
    )

    metrics = evaluate_model(flashrl, dataset=prompts, batch_size=2)

    assert flashrl._serving_backend.generation_defaults == {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "do_sample": False,
    }
    assert metrics == {
        "sample_count": 2,
        "reward_mean": pytest.approx(0.55),
        "exact_match": pytest.approx(0.5),
        "format_pass_rate": pytest.approx(0.5),
        "truncation_rate": pytest.approx(0.5),
    }


def test_reasoning_cli_help_uses_reduced_flag_surface() -> None:
    """The example CLIs should expose only the intended explicit operator knobs."""
    train_help = reasoning_example.build_argument_parser().format_help()
    assert "--config" in train_help
    assert "--profile" in train_help
    assert "--dataset" in train_help
    assert "--train-limit" in train_help
    assert "--checkpoint" not in train_help
    assert "--checkpoint-out" not in train_help
    assert "--task-config" not in train_help

    eval_help = reasoning_eval.build_argument_parser().format_help()
    assert "--config" in eval_help
    assert "--profile" in eval_help
    assert "--dataset" in eval_help
    assert "--eval-limit" in eval_help
    assert "--checkpoint" in eval_help
    assert "--batch-size" in eval_help
    assert "--task-config" not in eval_help


def test_default_eval_batch_size_and_checkpoint_path_are_code_constants() -> None:
    """Default example behavior should now come from code, not a sidecar file."""
    assert reasoning_example.DEFAULT_REASONING_CHECKPOINT_PATH == "/tmp/flashrl_reasoning_checkpoint.pt"
    assert reasoning_example.DEFAULT_REASONING_EVAL_BATCH_SIZE == 8


def test_prepare_reasoning_environment_sets_default_vllm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct example entrypoint should auto-fill the default vLLM runtime."""
    monkeypatch.delenv("FLASHRL_VLLM_PYTHON", raising=False)
    monkeypatch.setattr(reasoning_example, "find_default_vllm_python", lambda: "/tmp/fake-vllm-python")

    reasoning_example.prepare_reasoning_environment(
        "flashrl/examples/math/config.yaml",
        "vllm",
    )

    assert os.environ["FLASHRL_VLLM_PYTHON"] == "/tmp/fake-vllm-python"
    os.environ.pop("FLASHRL_VLLM_PYTHON", None)


def test_prepare_reasoning_environment_leaves_non_vllm_configs_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the vLLM example config should trigger env auto-discovery."""
    monkeypatch.delenv("FLASHRL_VLLM_PYTHON", raising=False)
    monkeypatch.setattr(reasoning_example, "find_default_vllm_python", lambda: "/tmp/fake-vllm-python")

    reasoning_example.prepare_reasoning_environment(
        "flashrl/examples/math/config.yaml"
    )

    assert "FLASHRL_VLLM_PYTHON" not in os.environ


def test_eval_module_uses_packaged_example_imports() -> None:
    """The eval module should resolve everything from flashrl.examples."""
    source = Path("flashrl/examples/math/eval.py").read_text(encoding="utf-8")
    source_tree = ast.parse(source)
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "flashrl.examples.math"
        for node in ast.walk(source_tree)
    )
    assert "from flashrl.examples.math import train as reasoning_math_example" in source
    assert "flashrl.example_support" not in source


def test_reasoning_math_module_holds_the_actual_logic() -> None:
    """The canonical module should hold the real implementation directly."""
    source = Path("flashrl/examples/math/train.py").read_text(encoding="utf-8")
    assert "flashrl.examples.math.train" in source
    assert "WORKFLOW_PATH" not in source
    assert "exec(compile(" not in source
    assert not Path("flashrl/examples/math/workflow.py").exists()
