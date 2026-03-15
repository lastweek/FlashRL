"""Tests for the managed HTTP vLLM serving backend."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
import yaml

import flashrl.framework.flashrl as flashrl_module
import flashrl.framework.serving as serving_module
import flashrl.framework.serving.huggingface as huggingface_module
import flashrl.framework.serving.vllm as vllm_module
from flashrl.framework.config import (
    GrpoConfig,
    LoggingConfig,
    MetricsConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.flashrl import FlashRL
from flashrl.framework.serving import (
    HuggingFaceServingBackend,
    ServingBackend,
    VLLMServingBackend,
    create_serving_backend,
)
from flashrl.framework.training import ActorTrainingBackend
from tests.conftest import TinyActor

pytestmark = pytest.mark.unit


class LocalActor:
    """Small actor stub used to isolate serving backend factory tests."""

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(load_state_dict=lambda state_dict: None)

    def eval(self) -> None:
        return None

    def generate(self, prompts: list[str], **kwargs):
        del prompts, kwargs
        return []

    def generate_batch(self, prompts: list[str], **kwargs):
        del prompts, kwargs
        return []

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        del prompts, group_size, kwargs
        return []

    def set_generation_defaults(self, **kwargs) -> None:
        del kwargs


class SnapshotModel:
    """Model stub that records save_pretrained calls."""

    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save_pretrained(self, path: str | Path, safe_serialization: bool = False) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_text("weights", encoding="utf-8")
        (target / "config.json").write_text("{}", encoding="utf-8")
        self.saved_paths.append(target)
        assert safe_serialization is True


class SnapshotTokenizer:
    """Tokenizer stub that records save_pretrained calls."""

    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.saved_paths.append(target)


class SnapshotTrainingActor:
    """Training actor stub used to verify snapshot sync behavior."""

    def __init__(self) -> None:
        self.model = SnapshotModel()
        self.tokenizer = SnapshotTokenizer()


class FakeProcess:
    """Minimal `Popen` stand-in for backend tests."""

    def __init__(self, pid: int | None = None) -> None:
        self.stderr = StringIO("")
        self._returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.pid = pid

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = 0


class StubTrainingBackend(ActorTrainingBackend):
    """Actor backend stub for FlashRL lifecycle tests."""

    def __init__(self, config, learning_rate: float = 1e-5) -> None:
        resolved_config = config or TrainingConfig(model_name="fake/model", device="cpu")
        super().__init__(resolved_config, learning_rate=learning_rate)
        self.model_copy = TinyActor(bias_shift=0.25)
        self.model_copy.train()
        self.device = self.model_copy.device
        self.optimizer = torch.optim.SGD(self.model_copy.model.parameters(), lr=learning_rate)
        self.startup_events = [
            {
                "component": "actor_backend",
                "status": "completed",
                "metadata": {
                    "device": str(self.device),
                    "cpu_threads": resolved_config.num_threads,
                    "duration_seconds": 0.0,
                },
            }
        ]


class ClosableServingBackend:
    """Serving backend stub that records close calls."""

    def __init__(self) -> None:
        self._actor = TinyActor(bias_shift=0.1)
        self.device = self._actor.device
        self.generation_defaults: dict[str, object] = {}
        self.close_calls = 0

    def generate(self, prompts: list[str], **kwargs):
        return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)
        self._actor.set_generation_defaults(**kwargs)

    def sync_from_training_actor(self, training_actor) -> None:
        del training_actor

    def set_live_rollout_debug(self, callback, context) -> None:
        del callback, context

    def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
        del candidate_index

    def clear_live_rollout_debug(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1


class MissingLogprobServingBackend(ClosableServingBackend):
    """Serving backend stub that omits token logprobs from rollout samples."""

    def generate_batch(self, prompts: list[str], **kwargs):
        samples = super().generate_batch(prompts, **kwargs)
        stripped = []
        for sample in samples:
            stripped.append(
                SimpleNamespace(
                    text=sample.text,
                    prompt_token_ids=sample.prompt_token_ids,
                    response_token_ids=sample.response_token_ids,
                    response_token_logprobs=[],
                    log_prob=0.0,
                    metadata=dict(sample.metadata),
                )
            )
        return stripped


def build_rollout_fn(prompts: list[Prompt], serving_backend: ServingBackend) -> list[RolloutOutput]:
    """Construct one rollout per prompt using the serving batch API."""
    samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
    return [
        RolloutOutput(
            text=sample.text,
            log_prob=sample.log_prob,
            prompt_token_ids=sample.prompt_token_ids,
            response_token_ids=sample.response_token_ids,
            response_token_logprobs=sample.response_token_logprobs,
            metadata=dict(sample.metadata),
            conversation=Conversation(
                messages=[
                    Message(role="user", content=prompt.text),
                    Message(role="assistant", content=sample.text),
                ]
            ),
        )
        for prompt, sample in zip(prompts, samples, strict=True)
    ]


def failing_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Raise during reward to exercise train-failure cleanup."""
    raise ValueError(f"reward failure for {rollout.text}")


def dataset_fn() -> list[Prompt]:
    """Small dataset for YAML config loading tests."""
    return [Prompt(text="prompt 0"), Prompt(text="prompt 1")]


def reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Deterministic non-failing reward for training smoke tests."""
    return RewardOutput(reward=float(len(rollout.text)))


def patch_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    serving_backend=None,
) -> None:
    """Patch FlashRL to use local actor and serving stubs."""
    monkeypatch.setattr(
        flashrl_module,
        "create_training_backend",
        lambda config, role, learning_rate=None: StubTrainingBackend(
            config,
            learning_rate=float(learning_rate or 1e-5),
        ),
    )
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None: (
            serving_backend if serving_backend is not None else FakeServingBackend(config)
        ),
    )


def build_flashrl(
    tmp_path: Path,
    *,
    reward_callback=reward_fn,
    serving_config: ServingConfig | None = None,
) -> FlashRL:
    """Create one explicit-config FlashRL instance for serving tests."""
    return FlashRL(
        actor_config=TrainingConfig(model_name="fake/model", device="cpu"),
        serving_config=serving_config or ServingConfig(model_name="fake/model", device="cpu"),
        trainer_config=TrainerConfig(batch_size=2, max_epochs=1),
        grpo_config=GrpoConfig(group_size=2),
        rollout_fn=build_rollout_fn,
        reward_fn=reward_callback,
        logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
        metrics_config=MetricsConfig(enabled=False),
    )


def _make_fake_runtime(
    tmp_path: Path,
    *,
    runtime_name: str = "runtime",
    python_name: str = "python",
    import_exit_code: int = 0,
) -> Path:
    bin_dir = tmp_path / runtime_name / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    python_path = bin_dir / python_name
    python_path.write_text(f"#!/bin/sh\nexit {import_exit_code}\n", encoding="utf-8")
    python_path.chmod(0o755)

    vllm_path = bin_dir / "vllm"
    vllm_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vllm_path.chmod(0o755)
    return python_path


def _patch_current_vllm_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    python_path = _make_fake_runtime(tmp_path)
    monkeypatch.setattr(vllm_module.sys, "executable", str(python_path))
    return python_path


def test_create_serving_backend_returns_huggingface_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving factory should keep the in-process backend as the default."""
    monkeypatch.setattr(huggingface_module, "ActorModel", LocalActor)

    backend = create_serving_backend(ServingConfig(model_name="fake/model"))

    assert isinstance(backend, HuggingFaceServingBackend)


def test_vllm_backend_rejects_debug_live_rollout(tmp_path: Path) -> None:
    """`vllm` should fail fast when serving debug live rollout is enabled."""
    python_path = _make_fake_runtime(tmp_path)

    with pytest.raises(ValueError, match="debug_live_rollout"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(python_path),
                debug_live_rollout=True,
            )
        )


def test_vllm_backend_rejects_missing_vllm_package(
    tmp_path: Path,
) -> None:
    """`vllm` should fail fast when the selected runtime cannot import the package."""
    python_path = _make_fake_runtime(tmp_path, import_exit_code=1)

    with pytest.raises(ValueError, match="requires the selected Python runtime to import `vllm`"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(python_path),
            )
        )


def test_vllm_backend_rejects_missing_runtime_python(tmp_path: Path) -> None:
    """`vllm` should fail fast when the selected runtime path does not exist."""
    missing_python = tmp_path / "missing" / "python"

    with pytest.raises(ValueError, match="runtime_python does not exist"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(missing_python),
            )
        )


def test_vllm_backend_rejects_missing_vllm_console_script(
    tmp_path: Path,
) -> None:
    """`vllm` should fail fast when the selected runtime lacks the console script."""
    bin_dir = tmp_path / "runtime" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python_path = bin_dir / "python"
    python_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_path.chmod(0o755)

    with pytest.raises(ValueError, match="console script"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(python_path),
            )
        )


def test_vllm_backend_rejects_reserved_vllm_args(
    tmp_path: Path,
) -> None:
    """FlashRL-owned launch flags should not be overridable via `vllm_args`."""
    python_path = _make_fake_runtime(tmp_path)

    with pytest.raises(ValueError, match="reserved flag"):
        VLLMServingBackend(
            ServingConfig(
                model_name="fake/model",
                backend="vllm",
                runtime_python=str(python_path),
                vllm_args=["--host=0.0.0.0"],
            )
        )


def test_create_serving_backend_returns_vllm_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The serving factory should dispatch to the managed `vllm` backend."""
    python_path = _make_fake_runtime(tmp_path)
    spawned_commands: list[list[str]] = []

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100 + len(spawned_commands))
    monkeypatch.setattr(
        VLLMServingBackend,
        "_spawn_process",
        lambda self, command: spawned_commands.append(command) or FakeProcess(),
    )
    monkeypatch.setattr(
        VLLMServingBackend,
        "_request_json",
        lambda self, url, *, method, payload=None, timeout: {"data": [{"id": "fake/model"}]},
    )

    backend = create_serving_backend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
            num_replicas=2,
        )
    )

    assert isinstance(backend, VLLMServingBackend)
    assert backend._snapshot_dir is None
    assert len(spawned_commands) == 2
    assert spawned_commands[0][:3] == [str(python_path.with_name("vllm")), "serve", "fake/model"]
    assert "--served-model-name" in spawned_commands[0]
    backend.close()


def test_vllm_backend_defaults_to_current_python_when_runtime_python_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The backend should fall back to the current interpreter when runtime_python is unset."""
    python_path = _patch_current_vllm_environment(monkeypatch, tmp_path)
    spawned_commands: list[list[str]] = []

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100)
    monkeypatch.setattr(
        VLLMServingBackend,
        "_spawn_process",
        lambda self, command: spawned_commands.append(command) or FakeProcess(),
    )
    monkeypatch.setattr(
        VLLMServingBackend,
        "_request_json",
        lambda self, url, *, method, payload=None, timeout: {"data": [{"id": "fake/model"}]},
    )

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
        )
    )

    try:
        assert backend.list_admin_objects()[0]["spec"]["pythonExecutable"] == str(python_path)
        assert spawned_commands[0][0] == str(python_path.with_name("vllm"))
    finally:
        backend.close()


def test_vllm_backend_retries_startup_with_local_snapshot_after_remote_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed vLLM startup should retry with a local snapshot when remote startup fails."""
    python_path = _make_fake_runtime(tmp_path)
    launched_sources: list[str] = []
    process = FakeProcess()

    monkeypatch.setattr(
        VLLMServingBackend,
        "_download_model_snapshot",
        lambda self, model_name: str(tmp_path / "hf-cache" / model_name.replace("/", "--")),
    )
    monkeypatch.setattr(VLLMServingBackend, "_stop_replicas", lambda self: None)
    monkeypatch.setattr(VLLMServingBackend, "_stop_replica_group", lambda self, replicas: None)

    def fake_launch(self, model_source: str):
        launched_sources.append(model_source)
        if model_source == "fake/model":
            raise RuntimeError("remote startup failed")
        return [
            vllm_module._Replica(
                index=0,
                port=8100,
                process=process,
                model_source=model_source,
                command=[str(python_path.with_name("vllm")), "serve", model_source],
                phase="Ready",
                ready_at="2026-03-13T00:00:01Z",
            )
        ]

    monkeypatch.setattr(VLLMServingBackend, "_launch_replicas", fake_launch)

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
        )
    )

    try:
        assert launched_sources == [
            "fake/model",
            str(tmp_path / "hf-cache" / "fake--model"),
        ]
        assert backend.list_admin_objects()[0]["spec"]["modelSource"] == str(
            tmp_path / "hf-cache" / "fake--model"
        )
    finally:
        backend.close()


def test_vllm_backend_sync_generate_and_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The backend should route completions over HTTP, restart on sync, and clean up."""
    python_path = _make_fake_runtime(tmp_path)
    spawned_commands: list[list[str]] = []
    processes: list[FakeProcess] = []
    requests: list[tuple[str, str, dict[str, object] | None]] = []
    ports = iter([8100, 8101, 8200, 8201])

    def fake_spawn(self, command):
        spawned_commands.append(command)
        process = FakeProcess()
        processes.append(process)
        return process

    def fake_request(self, url, *, method, payload=None, timeout):
        del timeout
        requests.append((method, url, payload))
        if method == "GET":
            return {"data": [{"id": self.config.model_name}]}
        assert payload is not None
        prompt = payload["prompt"]
        assert isinstance(prompt, str)
        grouped_choices = []
        prompt_token_ids = [ord(char) for char in prompt] or [0]
        for candidate_index in range(int(payload["n"])):
            grouped_choices.append(
                {
                    "text": f"vllm::{prompt}::{candidate_index}",
                    "prompt_token_ids": prompt_token_ids,
                    "token_ids": [10 + candidate_index, 20 + candidate_index],
                    "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    "finish_reason": "stop",
                }
            )
        return {"choices": grouped_choices}

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: next(ports))
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", fake_spawn)
    monkeypatch.setattr(VLLMServingBackend, "_request_json", fake_request)
    monkeypatch.setattr(
        vllm_module.tempfile,
        "mkdtemp",
        lambda prefix: str(tmp_path / "snapshot"),
    )

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
            num_replicas=2,
            vllm_args=["--max-num-seqs=32"],
        )
    )
    grouped = backend.generate_grouped(["ab", "cd", "ef"], group_size=2)
    startup_admin_objects = backend.list_admin_objects()
    training_actor = SnapshotTrainingActor()
    backend.sync_from_training_actor(training_actor)
    synced_admin_objects = backend.list_admin_objects()

    completion_requests = [
        payload
        for method, _, payload in requests
        if method == "POST" and payload is not None
    ]
    assert len(grouped) == 3
    assert [sample.text for sample in grouped[0]] == ["vllm::ab::0", "vllm::ab::1"]
    assert [sample.text for sample in grouped[1]] == ["vllm::cd::0", "vllm::cd::1"]
    assert [sample.text for sample in grouped[2]] == ["vllm::ef::0", "vllm::ef::1"]
    assert any(payload["return_token_ids"] is True for payload in completion_requests)
    assert [payload["prompt"] for payload in completion_requests] == ["ab", "cd", "ef"]
    assert startup_admin_objects[0]["spec"]["modelSource"] == "fake/model"
    assert startup_admin_objects[0]["spec"]["pythonExecutable"] == str(python_path)
    assert startup_admin_objects[0]["status"]["phase"] == "Ready"
    assert backend._snapshot_dir is not None
    assert (backend._snapshot_dir / "model.safetensors").exists()
    assert len(training_actor.model.saved_paths) == 1
    assert len(spawned_commands) == 4
    assert spawned_commands[0][2] == "fake/model"
    assert spawned_commands[2][2] == str(backend._snapshot_dir)
    assert synced_admin_objects[0]["spec"]["modelSource"] == str(backend._snapshot_dir)
    assert synced_admin_objects[0]["status"]["phase"] == "Ready"
    assert processes[0].terminate_calls == 1
    assert processes[1].terminate_calls == 1

    snapshot_dir = backend._snapshot_dir
    backend.close()
    closed_admin_objects = backend.list_admin_objects()

    assert snapshot_dir is not None
    assert not snapshot_dir.exists()
    assert processes[2].terminate_calls == 1
    assert processes[3].terminate_calls == 1
    assert closed_admin_objects[0]["status"]["phase"] == "Closed"


def test_vllm_backend_surfaces_startup_stderr_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replica stderr during startup should be mirrored into the parent startup console."""
    python_path = _make_fake_runtime(tmp_path)
    startup_lines: list[str] = []

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100)
    monkeypatch.setattr(
        VLLMServingBackend,
        "_spawn_process",
        lambda self, command: FakeProcess(),
    )
    monkeypatch.setattr(
        VLLMServingBackend,
        "_request_json",
        lambda self, url, *, method, payload=None, timeout: {"data": [{"id": self.config.model_name}]},
    )

    process = FakeProcess()
    process.stderr = StringIO("INFO engine: loading weights\nINFO engine: warming up\n")
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", lambda self, command: process)

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
        ),
        startup_logger=startup_lines.append,
    )

    try:
        for _ in range(50):
            if len(startup_lines) >= 2:
                break
            time.sleep(0.01)
        assert startup_lines == [
            "vllm[0] INFO engine: loading weights",
            "vllm[0] INFO engine: warming up",
        ]
    finally:
        backend.close()


def test_vllm_backend_close_uses_process_group_teardown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed vLLM replicas should be torn down via their process group."""
    python_path = _make_fake_runtime(tmp_path)
    killpg_calls: list[tuple[int, int]] = []
    process = FakeProcess(pid=4321)

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100)
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", lambda self, command: process)
    monkeypatch.setattr(
        VLLMServingBackend,
        "_request_json",
        lambda self, url, *, method, payload=None, timeout: {"data": [{"id": self.config.model_name}]},
    )
    monkeypatch.setattr(vllm_module.os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(
        vllm_module.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
        )
    )
    backend.close()

    assert killpg_calls == [(4322, vllm_module.signal.SIGTERM)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_flashrl_from_yaml_parses_vllm_serving_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """YAML-driven runs should preserve serving backend selection and expand runtime env vars."""
    captured_config: list[ServingConfig] = []
    runtime_python = str(tmp_path / "runtime" / "bin" / "python")
    monkeypatch.setenv("FLASHRL_VLLM_PYTHON", runtime_python)

    class FakeServingBackend:
        def __init__(self, config) -> None:
            captured_config.append(config)
            self.config = config
            self._actor = TinyActor(bias_shift=0.1)
            self.device = self._actor.device
            self.generation_defaults: dict[str, object] = {}

        def generate(self, prompts: list[str], **kwargs):
            return self._actor.generate(prompts, **kwargs)

        def generate_batch(self, prompts: list[str], **kwargs):
            return self._actor.generate_batch(prompts, **kwargs)

        def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
            return self._actor.generate_grouped(prompts, group_size, **kwargs)

        def set_generation_defaults(self, **kwargs) -> None:
            self.generation_defaults = dict(kwargs)
            self._actor.set_generation_defaults(**kwargs)

        def sync_from_training_actor(self, training_actor) -> None:
            del training_actor

        def set_live_rollout_debug(self, callback, context) -> None:
            del callback, context

        def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
            del candidate_index

        def clear_live_rollout_debug(self) -> None:
            return None

        def close(self) -> None:
            return None

    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "actor": {
                    "model_name": "fake/model",
                    "device": "cpu",
                    "num_threads": 1,
                },
                "serving": {
                    "model_name": "fake/model",
                    "backend": "vllm",
                    "runtime_python": "${FLASHRL_VLLM_PYTHON}",
                    "num_replicas": 2,
                    "vllm_args": ["--max-num-seqs=32", "--enable-prefix-caching"],
                },
                "trainer": {"batch_size": 2, "max_epochs": 1},
                "grpo": {"group_size": 2, "clip_ratio": 0.2, "kl_coefficient": 0.0},
                "logging": {"log_dir": str(tmp_path / "logs"), "console": False, "file": True},
                "metrics": {"enabled": False},
                "admin": {"admin_enabled": False},
                "hooks": {
                    "rollout_fn": "tests.test_vllm_backend:build_rollout_fn",
                    "reward_fn": "tests.test_vllm_backend:failing_reward_fn",
                    "dataset_fn": "tests.test_vllm_backend:dataset_fn",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        flashrl_module,
        "create_training_backend",
        lambda config, role, learning_rate=None: StubTrainingBackend(
            config,
            learning_rate=float(learning_rate or 1e-5),
        ),
    )
    monkeypatch.setattr(
        flashrl_module,
        "create_serving_backend",
        lambda config, startup_logger=None: FakeServingBackend(config),
    )

    trainer = FlashRL.from_yaml(config_path)

    assert trainer.serving_config.backend == "vllm"
    assert trainer.serving_config.runtime_python == runtime_python
    assert trainer.serving_config.num_replicas == 2
    assert trainer.serving_config.vllm_args == ["--max-num-seqs=32", "--enable-prefix-caching"]
    assert captured_config[0].backend == "vllm"
    assert captured_config[0].runtime_python == runtime_python


def test_flashrl_from_yaml_requires_present_runtime_python_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing config env vars should fail fast with a clear error."""
    monkeypatch.delenv("FLASHRL_VLLM_PYTHON", raising=False)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "actor": {
                    "model_name": "fake/model",
                    "device": "cpu",
                    "num_threads": 1,
                },
                "serving": {
                    "model_name": "fake/model",
                    "backend": "vllm",
                    "runtime_python": "${FLASHRL_VLLM_PYTHON}",
                },
                "trainer": {"batch_size": 2, "max_epochs": 1},
                "grpo": {"group_size": 2, "clip_ratio": 0.2, "kl_coefficient": 0.0},
                "logging": {"log_dir": str(tmp_path / "logs"), "console": False, "file": True},
                "metrics": {"enabled": False},
                "admin": {"admin_enabled": False},
                "hooks": {
                    "rollout_fn": "tests.test_vllm_backend:build_rollout_fn",
                    "reward_fn": "tests.test_vllm_backend:failing_reward_fn",
                    "dataset_fn": "tests.test_vllm_backend:dataset_fn",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required environment variable 'FLASHRL_VLLM_PYTHON'"):
        FlashRL.from_yaml(config_path)


def test_flashrl_train_failure_closes_serving_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Training failures should close the serving backend worker."""
    serving_backend = ClosableServingBackend()
    patch_backends(monkeypatch, serving_backend=serving_backend)

    trainer = build_flashrl(
        tmp_path,
        reward_callback=failing_reward_fn,
    )

    with pytest.raises(ValueError, match="reward failure"):
        trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    assert serving_backend.close_calls == 1


def test_flashrl_train_computes_missing_rollout_logprobs_from_training_actor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRPO training should fall back to the synced training actor when serving omits logprobs."""
    serving_backend = MissingLogprobServingBackend()
    patch_backends(monkeypatch, serving_backend=serving_backend)

    trainer = build_flashrl(
        tmp_path,
    )

    try:
        trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])
    finally:
        trainer.close()

    assert serving_backend.close_calls >= 1
