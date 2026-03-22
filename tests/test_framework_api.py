"""Regression tests for the public ``flashrl.framework`` import surface."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import flashrl.framework.flashrl as flashrl_module
from flashrl.framework import FlashRL, GrpoConfig, ServingConfig, TrainingConfig


pytestmark = pytest.mark.unit


def _run_script(tmp_path: Path, script_body: str) -> subprocess.CompletedProcess[str]:
    """Write one Python script to disk and execute it with the test interpreter."""
    script_path = tmp_path / "script.py"
    script_path.write_text(textwrap.dedent(script_body), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath_entries = [str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_framework_root_import_is_lazy(tmp_path: Path) -> None:
    """Importing the package root should not eagerly load heavyweight modules."""
    result = _run_script(
        tmp_path,
        """
        import sys

        import flashrl.framework as framework

        assert "flashrl.framework.flashrl" not in sys.modules
        assert "flashrl.framework.agent" not in sys.modules
        assert "flashrl.framework.agent.runtime" not in sys.modules

        assert framework.LoggingConfig.__name__ == "LoggingConfig"
        assert "flashrl.framework.flashrl" not in sys.modules

        assert framework.Agent.__name__ == "Agent"
        assert not hasattr(framework, "AgentRollout")
        assert not hasattr(framework, "AgentRuntime")
        assert "flashrl.framework.agent.runtime" in sys.modules
        assert "flashrl.framework.flashrl" not in sys.modules

        assert framework.FlashRL.__name__ == "FlashRL"
        assert "flashrl.framework.flashrl" in sys.modules
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_framework_root_import_is_spawn_safe(tmp_path: Path) -> None:
    """Spawned child processes should be able to import the package root safely."""
    result = _run_script(
        tmp_path,
        """
        import multiprocessing as mp


        def child(queue):
            import flashrl.framework as framework

            queue.put(
                {
                    "logging": framework.LoggingConfig.__name__,
                    "agent": framework.Agent.__name__,
                    "has_agent_rollout": hasattr(framework, "AgentRollout"),
                    "has_agent_runtime": hasattr(framework, "AgentRuntime"),
                    "flashrl": framework.FlashRL.__name__,
                }
            )


        if __name__ == "__main__":
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=child, args=(queue,))
            process.start()
            process.join(timeout=30)

            assert process.exitcode == 0, process.exitcode
            payload = queue.get(timeout=5)
            assert payload == {
                "logging": "LoggingConfig",
                "agent": "Agent",
                "has_agent_rollout": False,
                "has_agent_runtime": False,
                "flashrl": "FlashRL",
            }
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_framework_vllm_wrapper_import_is_spawn_safe_when_vllm_is_available(
    tmp_path: Path,
) -> None:
    """Importing the wrapper before spawning should not break child imports."""
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vllm is not installed in the current test interpreter.")
    try:
        import flashrl.framework.serving.vllm.server  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        error_text = str(exc)
        error_name = getattr(exc, "name", "") or ""
        if error_name.startswith("vllm") or "from 'vllm'" in error_text or "No module named 'vllm" in error_text:
            pytest.skip("vllm wrapper dependencies are not importable in the current interpreter.")
        raise

    result = _run_script(
        tmp_path,
        """
        import multiprocessing as mp

        import flashrl.framework.serving.vllm.server


        def child(queue):
            import flashrl.framework as framework

            queue.put(
                {
                    "agent": framework.Agent.__name__,
                    "has_agent_rollout": hasattr(framework, "AgentRollout"),
                    "has_agent_runtime": hasattr(framework, "AgentRuntime"),
                    "flashrl": framework.FlashRL.__name__,
                }
            )


        if __name__ == "__main__":
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=child, args=(queue,))
            process.start()
            process.join(timeout=30)

            assert process.exitcode == 0, process.exitcode
            payload = queue.get(timeout=5)
            assert payload == {
                "agent": "Agent",
                "has_agent_rollout": False,
                "has_agent_runtime": False,
                "flashrl": "FlashRL",
            }
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_framework_agent_toolbox_exports_expected_primitives(tmp_path: Path) -> None:
    """The agent package should expose runtime, context, and tool building blocks."""
    result = _run_script(
        tmp_path,
        """
        import importlib.util

        import flashrl.framework.agent as agent
        import flashrl.framework.tools as tools_shim

        assert agent.Agent.__name__ == "Agent"
        assert agent.AgentSample.__name__ == "AgentSample"
        assert agent.AgentState.__name__ == "AgentState"
        assert agent.SessionContext.__name__ == "SessionContext"
        assert agent.BaseContextManager.__name__ == "BaseContextManager"
        assert agent.CompactionManager.__name__ == "CompactionManager"
        assert agent.CompactionPolicy.__name__ == "CompactionPolicy"
        assert agent.WindowedContextManager.__name__ == "WindowedContextManager"
        assert agent.SkillManager.__name__ == "SkillManager"
        assert agent.SubagentManager.__name__ == "SubagentManager"
        assert agent.AgentToolExecutor.__name__ == "AgentToolExecutor"
        assert agent.Tool.__name__ == "Tool"
        assert agent.ToolProfile.__name__ == "ToolProfile"
        assert agent.ToolRegistry.__name__ == "ToolRegistry"
        assert agent.SubprocessToolRuntime.__name__ == "SubprocessToolRuntime"
        assert tools_shim.Tool is agent.Tool
        assert tools_shim.SubprocessToolRuntime is agent.SubprocessToolRuntime
        assert not hasattr(agent, "AgentRollout")
        assert not hasattr(agent, "AgentRuntime")
        assert importlib.util.find_spec("flashrl.framework.agent.presets") is None
        assert importlib.util.find_spec("flashrl.framework.rollout.user_defined") is None
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_run_logger_uses_direct_internal_imports() -> None:
    """Framework internals should not route submodule imports through the public root."""
    source = Path(__file__).resolve().parents[1] / "flashrl" / "framework" / "run_logger.py"
    text = source.read_text(encoding="utf-8")
    assert "from flashrl.framework import log_paths, rollout_logging" not in text


def test_flashrl_emits_shared_mps_guardrail_warning_for_explicit_shared_mps() -> None:
    """Explicit shared-MPS Hugging Face runs should emit a reliability warning."""
    flashrl = FlashRL.__new__(FlashRL)
    flashrl.reference_config = None
    flashrl.actor_config = TrainingConfig(model_name="fake/model", device="mps")
    flashrl.serving_config = ServingConfig(model_name="fake/model", device="mps", backend="huggingface")
    flashrl.grpo_config = GrpoConfig(group_size=2, max_new_tokens=384)
    flashrl._actor_backend = SimpleNamespace(device=SimpleNamespace(type="mps"))
    flashrl._serving_backend = SimpleNamespace(device=SimpleNamespace(type="mps"))

    recorded: list[tuple[str, str, str]] = []
    flashrl._emit_bootstrap_stage = lambda label, component, message: recorded.append(
        (label, component, message)
    )

    flashrl._emit_shared_mps_guardrail_warning()

    assert len(recorded) == 1
    label, component, message = recorded[0]
    assert label == "warn"
    assert component == "mps_guardrail"
    assert "device=cpu" in message
    assert "lower grpo.max_new_tokens" in message


def test_flashrl_emits_local_mps_policy_warning_on_macos_for_explicit_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS local explicit Hugging Face MPS usage should emit the broader policy warning."""
    monkeypatch.setattr(flashrl_module.sys, "platform", "darwin")
    flashrl = FlashRL.__new__(FlashRL)
    flashrl.actor_config = TrainingConfig(model_name="fake/model", device="mps")
    flashrl.reference_config = None
    flashrl.serving_config = ServingConfig(model_name="fake/model", device="cpu", backend="huggingface")

    recorded: list[tuple[str, str, str]] = []
    flashrl._emit_bootstrap_stage = lambda label, component, message: recorded.append(
        (label, component, message)
    )

    flashrl._emit_local_mps_policy_warning()

    assert len(recorded) == 1
    label, component, message = recorded[0]
    assert label == "warn"
    assert component == "mps_local_policy"
    assert "macOS" in message
    assert "device=mps" in message
    assert "device=cpu" in message


def test_flashrl_skips_local_mps_policy_warning_outside_macos_or_without_explicit_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local MPS policy warning should stay limited to explicit macOS MPS opt-in."""
    flashrl = FlashRL.__new__(FlashRL)
    flashrl.actor_config = TrainingConfig(model_name="fake/model", device="mps")
    flashrl.reference_config = None
    flashrl.serving_config = ServingConfig(model_name="fake/model", device="cpu", backend="huggingface")

    recorded: list[tuple[str, str, str]] = []
    flashrl._emit_bootstrap_stage = lambda label, component, message: recorded.append(
        (label, component, message)
    )

    monkeypatch.setattr(flashrl_module.sys, "platform", "linux")
    flashrl._emit_local_mps_policy_warning()
    assert recorded == []

    monkeypatch.setattr(flashrl_module.sys, "platform", "darwin")
    flashrl.actor_config = TrainingConfig(model_name="fake/model", device="cpu")
    flashrl._emit_local_mps_policy_warning()
    assert recorded == []
