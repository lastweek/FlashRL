"""Unit tests for the code-single-turn example."""

from __future__ import annotations

import importlib.util
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
    """Load one hyphen-folder script as a normal Python module for tests."""
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


code_basic_executor = load_script_module(
    "flashrl_code_basic_executor",
    "flashrl.framework.examples.code-single-turn/executor.py",
    aliases=("executor",),
)
code_basic = load_script_module(
    "flashrl_code_basic_train",
    "flashrl.framework.examples.code-single-turn/train.py",
    aliases=("train",),
)
code_basic_eval = load_script_module(
    "flashrl_code_basic_eval",
    "flashrl.framework.examples.code-single-turn/eval.py",
)


def make_prompt(*, task_id: str = "codeforces-train-1", rating: int = 1200) -> Prompt:
    """Build one strict code prompt with minimal metadata."""
    return Prompt(
        text=code_basic.render_code_prompt("Write a program that prints 42."),
        metadata={
            "task_id": task_id,
            "source": code_basic.DEFAULT_CODEFORCES_HF_DATASET,
            "config": code_basic.DEFAULT_CODEFORCES_HF_CONFIG,
            "split": "train",
            "language": "python",
            "rating": rating,
            "verifier": "python_tests",
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


def test_render_code_prompt_enforces_strict_contract() -> None:
    """The code example should keep the same visible strict output contract."""
    prompt = code_basic.render_code_prompt("Solve X.")

    assert "<think>...</think>" in prompt
    assert "<answer>...</answer>" in prompt
    assert "fenced Python code block" in prompt
    assert "Do not output any text before <think> or after </answer>." in prompt


def test_build_code_train_dataset_filters_python_stdio_rating_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dataset loading should keep only the supported Python Codeforces rows."""

    def fake_load_dataset(dataset_name: str, config_name: str, *, split: str):
        assert dataset_name == code_basic.DEFAULT_CODEFORCES_HF_DATASET
        assert config_name == code_basic.DEFAULT_CODEFORCES_HF_CONFIG
        assert split == code_basic.DEFAULT_CODEFORCES_TRAIN_SPLIT
        return [
            {
                "problem_id": "1000A",
                "language": "python",
                "prompt": "Add two numbers.",
                "official_tests_complete": True,
                "official_tests": [
                    {"input": "20 22\n", "output": "42\n"},
                    {"input": "1 2\n", "output": "3\n"},
                ],
                "generated_checker": "",
                "input_mode": "stdio",
                "interactive": False,
                "rating": 1200,
                "time_limit": 2,
                "memory_limit": 256,
            },
            {
                "problem_id": "1000B",
                "language": "cplusplus",
                "prompt": "Skip because not Python.",
                "official_tests_complete": True,
                "official_tests": [{"input": "", "output": ""}],
                "input_mode": "stdio",
                "interactive": False,
                "rating": 1200,
            },
            {
                "problem_id": "1000C",
                "language": "python",
                "prompt": "Skip because file IO.",
                "official_tests_complete": True,
                "official_tests": [{"input": "", "output": ""}],
                "input_mode": "file",
                "interactive": False,
                "rating": 1200,
            },
            {
                "problem_id": "1000D",
                "language": "python",
                "prompt": "Skip because too hard.",
                "official_tests_complete": True,
                "official_tests": [{"input": "", "output": ""}],
                "input_mode": "stdio",
                "interactive": False,
                "rating": 2000,
            },
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    dataset = code_basic.build_code_train_dataset(limit=1, max_tests_per_problem=1)
    output = capsys.readouterr().out

    assert len(dataset) == 1
    assert dataset[0].metadata["task_id"] == "codeforces-train-1000A"
    assert dataset[0].metadata["rating"] == 1200
    assert "dataset  name=codeforces" in output
    assert "source=open-r1/codeforces" in output
    assert "config=verifiable-prompts" in output
    assert "available=4  selected=1" in output
    assert "language=python" in output
    assert "rating=<= 1600" in output
    assert code_basic.CODEFORCES_EXECUTION_PAYLOADS["codeforces-train-1000A"]["official_tests"] == [
        {"input": "20 22\n", "output": "42\n"}
    ]


def test_build_code_eval_dataset_uses_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out evaluation should read the Codeforces test split."""
    seen_splits: list[str] = []

    def fake_load_dataset(dataset_name: str, config_name: str, *, split: str):
        del dataset_name, config_name
        seen_splits.append(split)
        return [
            {
                "problem_id": "1001A",
                "language": "python",
                "prompt": "Print 42.",
                "official_tests_complete": True,
                "official_tests": [{"input": "", "output": "42\n"}],
                "generated_checker": "",
                "input_mode": "stdio",
                "interactive": False,
                "rating": 800,
            }
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    dataset = code_basic.build_code_eval_dataset(limit=1)

    assert seen_splits == [code_basic.DEFAULT_CODEFORCES_EVAL_SPLIT]
    assert dataset[0].metadata["split"] == code_basic.DEFAULT_CODEFORCES_EVAL_SPLIT


def test_score_code_rollout_uses_execution_reward_and_strict_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reward cases should mirror the code example's 1.1 / 1.0 / 0.1 / 0.0 behavior."""
    prompt = make_prompt()
    code_basic.CODEFORCES_EXECUTION_PAYLOADS[prompt.metadata["task_id"]] = {
        "official_tests": [{"input": "", "output": "42\n"}],
        "checker_code": None,
        "time_limit_seconds": 1.0,
        "memory_limit_mb": 256,
    }

    def fake_run_python_solution(code: str, **kwargs):
        del kwargs
        if "print(42)" in code:
            return code_basic_executor.ExecutionResult(
                passed_tests=1,
                total_tests=1,
                pass_rate=1.0,
                execution_seconds=0.2,
                failure_reason=None,
                checker_used=False,
            )
        return code_basic_executor.ExecutionResult(
            passed_tests=0,
            total_tests=1,
            pass_rate=0.0,
            execution_seconds=0.2,
            failure_reason="wrong_answer",
            checker_used=False,
        )

    monkeypatch.setattr(code_basic.executor, "run_python_solution", fake_run_python_solution)

    perfect = code_basic.score_code_rollout(
        make_rollout(
            prompt,
            "<think>Use print.</think><answer>```python\nprint(42)\n```</answer>",
        ),
        run_timeout_seconds=1.0,
        memory_limit_mb=256,
    )
    trailing = code_basic.score_code_rollout(
        make_rollout(
            prompt,
            "<think>Use print.</think><answer>```python\nprint(42)\n```</answer>\nextra",
        ),
        run_timeout_seconds=1.0,
        memory_limit_mb=256,
    )
    strict_wrong = code_basic.score_code_rollout(
        make_rollout(
            prompt,
            "<think>Try print.</think><answer>```python\nprint(0)\n```</answer>",
        ),
        run_timeout_seconds=1.0,
        memory_limit_mb=256,
    )
    missing_fence = code_basic.score_code_rollout(
        make_rollout(
            prompt,
            "<think>Try print.</think><answer>print(42)</answer>",
        ),
        run_timeout_seconds=1.0,
        memory_limit_mb=256,
    )

    assert perfect.reward == pytest.approx(1.1)
    assert perfect.metadata["task_id"] == "codeforces-train-1"
    assert perfect.metadata["accuracy_pass"] is True
    assert perfect.metadata["format_pass"] is True
    assert perfect.metadata["execution_status"] == "passed"
    assert perfect.metadata["code_preview"] == "print(42)"
    assert trailing.reward == pytest.approx(1.0)
    assert trailing.metadata["accuracy_pass"] is True
    assert trailing.metadata["format_pass"] is False
    assert trailing.metadata["execution_status"] == "passed"
    assert strict_wrong.reward == pytest.approx(0.1)
    assert strict_wrong.metadata["accuracy_pass"] is False
    assert strict_wrong.metadata["format_pass"] is True
    assert strict_wrong.metadata["execution_status"] == "wrong_answer"
    assert strict_wrong.metadata["code_preview"] == "print(0)"
    assert missing_fence.reward == pytest.approx(0.0)
    assert missing_fence.metadata["format_pass"] is False
    assert missing_fence.metadata["failure_reason"] == "invalid_code_fence"
    assert missing_fence.metadata["execution_status"] == "invalid_code_fence"
    assert missing_fence.metadata["code_preview"] == "print(42)"

    output = capsys.readouterr().out
    assert "code     task=codeforces-train-1" in output
    assert "status=passed" in output
    assert "status=wrong_answer" in output
    assert "status=invalid_code_fence" in output
    assert "preview  print(42)" in output
    assert "preview  print(0)" in output


def test_score_code_rollout_marks_truncation_invalid_for_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Length truncation should remove the format bonus even when tests pass."""
    prompt = make_prompt(task_id="codeforces-train-trunc")
    code_basic.CODEFORCES_EXECUTION_PAYLOADS[prompt.metadata["task_id"]] = {
        "official_tests": [{"input": "", "output": "42\n"}],
        "checker_code": None,
        "time_limit_seconds": 1.0,
        "memory_limit_mb": 256,
    }
    monkeypatch.setattr(
        code_basic.executor,
        "run_python_solution",
        lambda code, **kwargs: code_basic_executor.ExecutionResult(
            passed_tests=1,
            total_tests=1,
            pass_rate=1.0,
            execution_seconds=0.1,
            failure_reason=None,
            checker_used=False,
        ),
    )

    reward = code_basic.score_code_rollout(
        make_rollout(
            prompt,
            "<think>Use print.</think><answer>```python\nprint(42)\n```</answer>",
            finish_reason="length",
        ),
        run_timeout_seconds=1.0,
        memory_limit_mb=256,
    )

    assert reward.reward == pytest.approx(1.0)
    assert reward.metadata["truncated"] is True
    assert reward.metadata["format_pass"] is False
    assert reward.metadata["accuracy_pass"] is True
    assert reward.metadata["execution_status"] == "passed"

    output = capsys.readouterr().out
    assert "truncated=yes" in output


def test_run_python_solution_handles_success_and_checker_path() -> None:
    """The executor should support direct stdout matches and optional checkers."""
    direct = code_basic_executor.run_python_solution(
        "import sys\nprint(sum(map(int, sys.stdin.read().split())))\n",
        official_tests=[{"input": "20 22\n", "output": "42\n"}],
        checker_code=None,
        timeout_seconds=1.0,
        memory_limit_mb=256,
    )
    checker = code_basic_executor.run_python_solution(
        "print('3 2 1')\n",
        official_tests=[{"input": "", "output": "1 2 3\n"}],
        checker_code=(
            "import sys\n"
            "expected = open(sys.argv[2], encoding='utf-8').read().split()\n"
            "actual = open(sys.argv[3], encoding='utf-8').read().split()\n"
            "print(1 if sorted(expected) == sorted(actual) else 0)\n"
        ),
        timeout_seconds=1.0,
        memory_limit_mb=256,
    )

    assert direct.passed_tests == 1
    assert direct.pass_rate == pytest.approx(1.0)
    assert direct.failure_reason is None
    assert checker.passed_tests == 1
    assert checker.checker_used is True


@pytest.mark.parametrize(
    ("code", "expected_reason"),
    [
        ("print('0')\n", "wrong_answer"),
        ("def broken(:\n", "syntax_error"),
        ("raise RuntimeError('boom')\n", "runtime_error"),
        ("while True:\n    pass\n", "timeout"),
    ],
)
def test_run_python_solution_failure_modes(code: str, expected_reason: str) -> None:
    """The executor should surface the main local failure modes clearly."""
    result = code_basic_executor.run_python_solution(
        code,
        official_tests=[{"input": "", "output": "42\n"}],
        checker_code=None,
        timeout_seconds=0.2,
        memory_limit_mb=256,
    )

    assert result.passed_tests == 0
    assert result.failure_reason == expected_reason


def test_code_basic_cli_help_uses_explicit_flag_surface() -> None:
    """The example CLIs should expose the intended explicit operator knobs."""
    train_help = code_basic.build_argument_parser().format_help()
    assert "--config" in train_help
    assert "--train-limit" in train_help
    assert "--checkpoint" not in train_help
    assert "--checkpoint-out" not in train_help
    assert "--rating-min" in train_help
    assert "--rating-max" in train_help
    assert "--run-timeout-seconds" in train_help
    assert "--memory-limit-mb" in train_help
    assert "--max-tests-per-problem" in train_help

    eval_help = code_basic_eval.build_argument_parser().format_help()
    assert "--config" in eval_help
    assert "--eval-limit" in eval_help
    assert "--batch-size" in eval_help
    assert "--rating-min" in eval_help
    assert "--rating-max" in eval_help
    assert "--run-timeout-seconds" in eval_help
    assert "--memory-limit-mb" in eval_help
    assert "--max-tests-per-problem" in eval_help


def test_prepare_code_basic_environment_sets_default_vllm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script entrypoint should auto-fill the default vLLM runtime."""
    monkeypatch.delenv("FLASHRL_VLLM_PYTHON", raising=False)
    monkeypatch.setattr(
        code_basic,
        "find_default_vllm_python",
        lambda: "/tmp/fake-vllm-python",
    )

    code_basic.prepare_code_basic_environment(
        "flashrl.framework.examples.code-single-turn/config_vllm.yaml"
    )

    assert "FLASHRL_VLLM_PYTHON" in sys.modules["os"].environ
    sys.modules["os"].environ.pop("FLASHRL_VLLM_PYTHON", None)


def test_code_basic_main_uses_profile_aware_flashrl_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The code example should pass config_path into FlashRL instead of parsing YAML locally."""
    captured: dict[str, object] = {}

    class FakeFlashRL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def train(self, dataset) -> None:
            captured["dataset"] = dataset

        def close(self) -> None:
            return None

    monkeypatch.setattr(code_basic, "FlashRL", FakeFlashRL)
    monkeypatch.setattr(code_basic, "build_code_train_dataset", lambda **kwargs: [])

    exit_code = code_basic.main(
        [
            "--config",
            "flashrl.framework.examples.code-single-turn/config.yaml",
            "--run-timeout-seconds",
            "2.5",
            "--memory-limit-mb",
            "123",
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == "flashrl.framework.examples.code-single-turn/config.yaml"
    assert callable(captured["rollout_fn"])
    assert callable(captured["reward_fn"])
    assert "checkpoint_out" not in captured


def test_code_basic_train_is_the_real_example_module() -> None:
    """train.py should hold the actual code example logic without workflow indirection."""
    train_source = Path("flashrl.framework.examples.code-single-turn/train.py").read_text(
        encoding="utf-8"
    )
    eval_source = Path("flashrl.framework.examples.code-single-turn/eval.py").read_text(
        encoding="utf-8"
    )
    assert "WORKFLOW_PATH" not in train_source
    assert "exec(compile(" not in train_source
    assert "import train as code_example" in eval_source
    assert not Path("flashrl.framework.examples.code-single-turn/workflow.py").exists()


def test_evaluate_model_reports_pass_rate_solve_rate_and_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out evaluation should aggregate solve-rate style metrics."""
    prompts = [
        make_prompt(task_id="codeforces-test-1", rating=900),
        make_prompt(task_id="codeforces-test-2", rating=1000),
    ]
    code_basic.CODEFORCES_EXECUTION_PAYLOADS[prompts[0].metadata["task_id"]] = {
        "official_tests": [{"input": "", "output": "42\n"}],
        "checker_code": None,
        "time_limit_seconds": 1.0,
        "memory_limit_mb": 256,
    }
    code_basic.CODEFORCES_EXECUTION_PAYLOADS[prompts[1].metadata["task_id"]] = {
        "official_tests": [{"input": "", "output": "42\n"}],
        "checker_code": None,
        "time_limit_seconds": 1.0,
        "memory_limit_mb": 256,
    }

    def fake_run_python_solution(code: str, **kwargs):
        del kwargs
        if "print(42)" in code:
            return code_basic_executor.ExecutionResult(
                passed_tests=1,
                total_tests=1,
                pass_rate=1.0,
                execution_seconds=0.1,
                failure_reason=None,
                checker_used=False,
            )
        return code_basic_executor.ExecutionResult(
            passed_tests=0,
            total_tests=1,
            pass_rate=0.0,
            execution_seconds=0.1,
            failure_reason="wrong_answer",
            checker_used=False,
        )

    monkeypatch.setattr(code_basic.executor, "run_python_solution", fake_run_python_solution)

    class FakeServingBackend:
        def __init__(self) -> None:
            self.generation_defaults: dict[str, object] = {}

        def set_generation_defaults(self, **kwargs) -> None:
            self.generation_defaults = dict(kwargs)

        def generate_batch(self, prompt_texts: list[str], **kwargs):
            del kwargs
            outputs = []
            for prompt_text in prompt_texts:
                if "codeforces-test-1" in prompt_text:
                    raise AssertionError("prompt metadata should not appear in prompt text")
                if "prints 42" in prompt_text:
                    outputs.append(
                        SimpleNamespace(
                            text="<think>Use print.</think><answer>```python\nprint(42)\n```</answer>",
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
                            text="<think>Use print.</think><answer>```python\nprint(0)\n```</answer>",
                            prompt_token_ids=[1, 2, 3],
                            response_token_ids=[4, 5, 6],
                            response_token_logprobs=[-0.1, -0.1, -0.1],
                            log_prob=-0.3,
                            metadata={"finish_reason": "length"},
                        )
                    )
            return outputs

    prompts[0].text = code_basic.render_code_prompt("Write code that prints 42.")
    prompts[1].text = code_basic.render_code_prompt("Write code that prints 0.")
    flashrl = SimpleNamespace(
        _serving_backend=FakeServingBackend(),
        rollout_config=SimpleNamespace(max_new_tokens=96),
    )

    metrics = code_basic_eval.evaluate_model(
        flashrl,
        dataset=prompts,
        batch_size=2,
        run_timeout_seconds=1.0,
        memory_limit_mb=256,
    )

    assert flashrl._serving_backend.generation_defaults == {
        "max_new_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "do_sample": False,
    }
    assert metrics == {
        "sample_count": 2,
        "reward_mean": pytest.approx(0.55),
        "pass_rate_mean": pytest.approx(0.5),
        "solve_rate": pytest.approx(0.5),
        "format_pass_rate": pytest.approx(0.5),
        "truncation_rate": pytest.approx(0.5),
    }
