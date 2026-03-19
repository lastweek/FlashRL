"""Tests for the managed HTTP vLLM serving backend."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import os
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch
import yaml

import flashrl.framework.flashrl as flashrl_module
import flashrl.framework.serving.huggingface as huggingface_module
from flashrl.framework.config import (
    GrpoConfig,
    LoggingConfig,
    MetricsConfig,
    ServingConfig,
    TrainerConfig,
    TrainingConfig,
)
from flashrl.framework.data_models import (
    Conversation,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    WeightVersionInfo,
)
from flashrl.framework.flashrl import FlashRL
from flashrl.framework.serving import (
    HuggingFaceServingBackend,
    ServingBackend,
    VLLMServingBackend,
    create_serving_backend,
)
from flashrl.framework.serving.vllm.backend import _Replica
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
        self.config = SimpleNamespace(debug_live_rollout=False)
        self._actor = TinyActor(bias_shift=0.1)
        self.device = self._actor.device
        self.generation_defaults: dict[str, object] = {}
        self.close_calls = 0
        self._active_weight_version = WeightVersionInfo(
            version_id=0,
            source_training_step=None,
            source_epoch=None,
            activated_at="2026-03-19T00:00:00Z",
            model_source="test-serving://startup",
            origin="startup",
        )
        self._next_weight_version_id = 1

    def generate(self, prompts: list[str], **kwargs):
        return self._actor.generate(prompts, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs):
        return self._actor.generate_batch(prompts, **kwargs)

    def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
        return self._actor.generate_grouped(prompts, group_size, **kwargs)

    def set_generation_defaults(self, **kwargs) -> None:
        self.generation_defaults = dict(kwargs)
        self._actor.set_generation_defaults(**kwargs)

    def sync_from_training_actor(
        self,
        training_actor,
        *,
        source_training_step: int | None = None,
        source_epoch: int | None = None,
        origin: str = "sync",
    ) -> WeightVersionInfo:
        del training_actor
        self._active_weight_version = WeightVersionInfo(
            version_id=self._next_weight_version_id,
            source_training_step=source_training_step,
            source_epoch=source_epoch,
            activated_at=f"2026-03-19T00:00:{self._next_weight_version_id:02d}Z",
            model_source=f"test-serving://version-{self._next_weight_version_id}",
            origin=origin,
        )
        self._next_weight_version_id += 1
        return self.current_weight_version()

    def current_weight_version(self) -> WeightVersionInfo:
        return self._active_weight_version.model_copy(deep=True)

    def export_weight_version_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "next_version_id": self._next_weight_version_id,
        }

    def restore_weight_version_state(self, state: dict[str, object] | None) -> None:
        if not isinstance(state, dict):
            return
        next_version_id = state.get("next_version_id")
        if isinstance(next_version_id, int):
            self._next_weight_version_id = max(next_version_id, self._next_weight_version_id)

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
        lambda config, startup_logger=None, log_dir=None: (
            serving_backend if serving_backend is not None else ClosableServingBackend()
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
    monkeypatch.setattr(sys, "executable", str(python_path))
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
            vllm_args=["--max-num-seqs=32"],
        )
    )

    assert isinstance(backend, VLLMServingBackend)
    assert backend._snapshot_dir is None
    assert len(spawned_commands) == 2
    assert spawned_commands[0][:3] == [
        str(python_path),
        "-m",
        "flashrl.framework.serving.vllm.server",
    ]
    assert spawned_commands[0][4] == "fake/model"
    assert "--served-model-name" in spawned_commands[0]
    assert "--max-num-seqs=32" in spawned_commands[0]
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
        assert spawned_commands[0][0] == str(python_path)
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
            _Replica(
                index=0,
                port=8100,
                process=process,
                model_source=model_source,
                command=[
                    str(python_path),
                    "-m",
                    "flashrl.framework.serving.vllm.server",
                    "--model",
                    model_source,
                ],
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


def test_vllm_backend_download_model_snapshot_prefers_local_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Snapshot fallback should prefer the local HF cache before touching the network."""
    calls: list[tuple[str, bool]] = []
    cached_path = tmp_path / "hf-cache" / "fake--model"

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool = False):
        calls.append((repo_id, local_files_only))
        assert local_files_only is True
        return str(cached_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    backend = object.__new__(VLLMServingBackend)
    snapshot = VLLMServingBackend._download_model_snapshot(backend, "fake/model")

    assert snapshot == str(cached_path)
    assert calls == [("fake/model", True)]


def test_vllm_backend_download_model_snapshot_falls_back_to_network_when_cache_misses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Snapshot fallback should retry without `local_files_only` when cache lookup fails."""
    calls: list[tuple[str, bool]] = []
    downloaded_path = tmp_path / "hf-cache" / "fake--model"

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool = False):
        calls.append((repo_id, local_files_only))
        if local_files_only:
            raise RuntimeError("cache miss")
        return str(downloaded_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    backend = object.__new__(VLLMServingBackend)
    snapshot = VLLMServingBackend._download_model_snapshot(backend, "fake/model")

    assert snapshot == str(downloaded_path)
    assert calls == [
        ("fake/model", True),
        ("fake/model", False),
    ]


def test_vllm_backend_sync_generate_and_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run-backed vLLM sync should write into the run directory and preserve the final version."""
    python_path = _make_fake_runtime(tmp_path)
    spawned_commands: list[list[str]] = []
    processes: list[FakeProcess] = []
    requests: list[tuple[str, str, dict[str, object] | None]] = []
    ports = iter([8100, 8101])
    run_dir = tmp_path / "logs" / "run-1"

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
        if url.endswith("/v1/load_weights_from_disk"):
            return {"status": "success", "model_source": payload["model_source"]}
        prompt = payload["prompt"]
        assert isinstance(prompt, str)
        grouped_choices = []
        prompt_token_ids = [ord(char) for char in prompt] or [0]
        for candidate_index in range(int(payload["n"])):
            grouped_choices.append(
                {
                    "text": f"vllm::{prompt}::{candidate_index}",
                    "index": candidate_index,
                    "prompt_token_ids": prompt_token_ids,
                    "token_ids": [10 + candidate_index, 20 + candidate_index],
                    "logprobs": {
                        "text_offset": [0, 1],
                        "token_logprobs": [-0.1, -0.2],
                        "tokens": [f"tok::{candidate_index}::0", f"tok::{candidate_index}::1"],
                        "top_logprobs": [
                            {f"tok::{candidate_index}::0": -0.1},
                            {f"tok::{candidate_index}::1": -0.2},
                        ],
                    },
                    "finish_reason": "stop",
                }
            )
        return {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 123,
            "model": self.config.model_name,
            "choices": grouped_choices,
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": 2 * int(payload["n"]),
                "total_tokens": len(prompt_token_ids) + (2 * int(payload["n"])),
            },
        }

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: next(ports))
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", fake_spawn)
    monkeypatch.setattr(VLLMServingBackend, "_request_json", fake_request)

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
            num_replicas=2,
            vllm_args=["--max-num-seqs=32"],
        ),
        log_dir=run_dir,
    )
    grouped = backend.generate_grouped(["ab", "cd", "ef"], group_size=2)
    startup_admin_objects = backend.list_admin_objects()
    training_actor = SnapshotTrainingActor()
    sync_info = backend.sync_from_training_actor(
        training_actor,
        source_training_step=4,
        source_epoch=2,
    )
    synced_admin_objects = backend.list_admin_objects()
    snapshot_root = backend._snapshot_dir
    assert snapshot_root is not None
    assert snapshot_root == (run_dir / "vllm" / "weights").resolve()
    synced_snapshot_dir = Path(sync_info.model_source)

    completion_requests = [
        payload
        for method, _, payload in requests
        if method == "POST" and payload is not None and payload.get("prompt") is not None
    ]
    assert len(grouped) == 3
    assert [sample.text for sample in grouped[0]] == ["vllm::ab::0", "vllm::ab::1"]
    assert [sample.text for sample in grouped[1]] == ["vllm::cd::0", "vllm::cd::1"]
    assert [sample.text for sample in grouped[2]] == ["vllm::ef::0", "vllm::ef::1"]
    assert grouped[0][0].response_token_logprobs == [-0.1, -0.2]
    assert grouped[0][0].log_prob == pytest.approx(-0.3)
    assert any(payload["return_token_ids"] is True for payload in completion_requests)
    assert all(payload["logprobs"] == 1 for payload in completion_requests)
    assert [payload["prompt"] for payload in completion_requests] == ["ab", "cd", "ef"]
    assert startup_admin_objects[0]["spec"]["modelSource"] == "fake/model"
    assert startup_admin_objects[0]["spec"]["pythonExecutable"] == str(python_path)
    assert startup_admin_objects[0]["status"]["phase"] == "Ready"
    assert startup_admin_objects[0]["status"]["activeWeightVersion"]["version_id"] == 0
    assert startup_admin_objects[0]["status"]["pendingWeightVersion"] is None
    assert sync_info.version_id == 1
    assert sync_info.source_training_step == 4
    assert sync_info.source_epoch == 2
    assert sync_info.origin == "sync"
    assert synced_snapshot_dir.parent == snapshot_root
    assert synced_snapshot_dir.name == "version-00000001"
    assert (synced_snapshot_dir / "model.safetensors").exists()
    assert len(training_actor.model.saved_paths) == 1
    assert training_actor.model.saved_paths[0].parent == snapshot_root
    assert training_actor.model.saved_paths[0].name.startswith(".version-00000001.")
    assert len(spawned_commands) == 2
    assert spawned_commands[0][:3] == [
        str(python_path),
        "-m",
        "flashrl.framework.serving.vllm.server",
    ]
    assert "--served-model-name" in spawned_commands[0]
    assert "--max-num-seqs=32" in spawned_commands[0]
    sync_requests = [
        (method, url, payload)
        for method, url, payload in requests
        if method == "POST" and "/v1/load_weights_from_disk" in url
    ]
    assert len(sync_requests) == 2
    assert {payload["model_source"] for _, _, payload in sync_requests} == {str(synced_snapshot_dir)}
    assert synced_admin_objects[0]["spec"]["modelSource"] == str(synced_snapshot_dir)
    assert synced_admin_objects[0]["status"]["phase"] == "Ready"
    assert synced_admin_objects[0]["status"]["activeWeightVersion"]["version_id"] == 1
    assert synced_admin_objects[0]["status"]["pendingWeightVersion"] is None
    assert processes[0].terminate_calls == 0
    assert processes[1].terminate_calls == 0
    stray_dir = snapshot_root / ".version-00000002.stray.tmp"
    stray_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_root / ".orphan.tmp").write_text("cleanup", encoding="utf-8")

    backend.close()
    closed_admin_objects = backend.list_admin_objects()

    assert snapshot_root.exists()
    assert [child.name for child in snapshot_root.iterdir()] == ["version-00000001"]
    assert processes[0].terminate_calls == 1
    assert processes[1].terminate_calls == 1
    assert closed_admin_objects[0]["status"]["phase"] == "Closed"


def test_vllm_backend_sync_without_log_dir_uses_temp_snapshot_root_and_deletes_it_on_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backends created without a run dir should keep the temp-root fallback behavior."""
    python_path = _make_fake_runtime(tmp_path)
    ports = iter([8100])
    snapshot_root = (tmp_path / "snapshot-root").resolve()

    def fake_request(self, url, *, method, payload=None, timeout):
        del timeout
        if method == "GET":
            return {"data": [{"id": self.config.model_name}]}
        assert payload is not None
        if url.endswith("/v1/load_weights_from_disk"):
            return {"status": "success", "model_source": payload["model_source"]}
        raise RuntimeError(f"Unexpected request to {url}")

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: next(ports))
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", lambda self, command: FakeProcess())
    monkeypatch.setattr(VLLMServingBackend, "_request_json", fake_request)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(snapshot_root))

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
        )
    )

    try:
        sync_info = backend.sync_from_training_actor(
            SnapshotTrainingActor(),
            source_training_step=1,
            source_epoch=1,
        )
        assert backend._snapshot_dir == snapshot_root
        assert Path(sync_info.model_source).parent == snapshot_root
        assert snapshot_root.exists()
    finally:
        backend.close()

    assert not snapshot_root.exists()


def test_vllm_backend_rolls_back_partial_replica_reload_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed replica reload should roll earlier replicas back to the previous version."""
    python_path = _make_fake_runtime(tmp_path)
    ports = iter([8100, 8101])
    reload_requests: list[tuple[str, str]] = []

    def fake_request(self, url, *, method, payload=None, timeout):
        del timeout
        if method == "GET":
            return {"data": [{"id": self.config.model_name}]}
        assert payload is not None
        if url.endswith("/v1/load_weights_from_disk"):
            model_source = str(payload["model_source"])
            reload_requests.append((url, model_source))
            if url.startswith("http://127.0.0.1:8101") and model_source.endswith("version-00000001"):
                raise RuntimeError("replica reload failed")
            return {"status": "success", "model_source": model_source}
        raise RuntimeError(f"Unexpected request to {url}")

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: next(ports))
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", lambda self, command: FakeProcess())
    monkeypatch.setattr(VLLMServingBackend, "_request_json", fake_request)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path / "snapshot-root"))

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
            num_replicas=2,
        )
    )

    try:
        training_actor = SnapshotTrainingActor()
        with pytest.raises(RuntimeError, match="Failed to load weights into vllm replicas"):
            backend.sync_from_training_actor(
                training_actor,
                source_training_step=1,
                source_epoch=1,
            )

        assert [replica.model_source for replica in backend._replicas] == ["fake/model", "fake/model"]
        sync_status = backend.weight_sync_status()
        assert sync_status["activeWeightVersion"]["version_id"] == 0
        assert sync_status["pendingWeightVersion"] is None
        assert sync_status["syncHealthy"] is True
        assert "Failed to load weights into vllm replicas" in sync_status["lastSyncError"]
        snapshot_root = backend._snapshot_dir
        assert snapshot_root is not None
        assert [child.name for child in snapshot_root.iterdir() if not child.name.startswith(".")] == []
        assert len(reload_requests) == 3
        assert reload_requests[0][1].endswith("version-00000001")
        assert reload_requests[1][1].endswith("version-00000001")
        assert reload_requests[2][1] == "fake/model"
        admin_objects = backend.list_admin_objects()
        assert admin_objects[0]["status"]["activeWeightVersion"]["version_id"] == 0
    finally:
        backend.close()


def test_vllm_backend_treats_missing_completion_logprobs_as_unavailable() -> None:
    """Malformed completion logprobs should fall back to learner-side recomputation."""
    backend = object.__new__(VLLMServingBackend)

    assert (
        VLLMServingBackend._extract_output_logprobs(
            backend,
            {"token_logprobs": [None, None]},
            [10, 20],
        )
        == []
    )
    assert (
        VLLMServingBackend._extract_output_logprobs(
            backend,
            {"token_logprobs": [-0.1]},
            [10, 20],
        )
        == []
    )


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
    monkeypatch.setattr(os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(
        os,
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

    assert killpg_calls == [(4322, signal.SIGTERM)]
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
            self._active_weight_version = WeightVersionInfo(
                version_id=0,
                source_training_step=None,
                source_epoch=None,
                activated_at="2026-03-19T00:00:00Z",
                model_source="yaml-serving://startup",
                origin="startup",
            )
            self._next_weight_version_id = 1

        def generate(self, prompts: list[str], **kwargs):
            return self._actor.generate(prompts, **kwargs)

        def generate_batch(self, prompts: list[str], **kwargs):
            return self._actor.generate_batch(prompts, **kwargs)

        def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
            return self._actor.generate_grouped(prompts, group_size, **kwargs)

        def set_generation_defaults(self, **kwargs) -> None:
            self.generation_defaults = dict(kwargs)
            self._actor.set_generation_defaults(**kwargs)

        def sync_from_training_actor(
            self,
            training_actor,
            *,
            source_training_step: int | None = None,
            source_epoch: int | None = None,
            origin: str = "sync",
        ) -> WeightVersionInfo:
            del training_actor
            self._active_weight_version = WeightVersionInfo(
                version_id=self._next_weight_version_id,
                source_training_step=source_training_step,
                source_epoch=source_epoch,
                activated_at=f"2026-03-19T00:00:{self._next_weight_version_id:02d}Z",
                model_source=f"yaml-serving://version-{self._next_weight_version_id}",
                origin=origin,
            )
            self._next_weight_version_id += 1
            return self.current_weight_version()

        def current_weight_version(self) -> WeightVersionInfo:
            return self._active_weight_version.model_copy(deep=True)

        def export_weight_version_state(self) -> dict[str, object]:
            return {
                "schema_version": 1,
                "next_version_id": self._next_weight_version_id,
            }

        def restore_weight_version_state(self, state: dict[str, object] | None) -> None:
            if not isinstance(state, dict):
                return
            next_version_id = state.get("next_version_id")
            if isinstance(next_version_id, int):
                self._next_weight_version_id = max(next_version_id, self._next_weight_version_id)

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
        lambda config, startup_logger=None, log_dir=None: FakeServingBackend(config),
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


def test_vllm_backend_pause_and_resume_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pause and resume inference should work correctly."""
    python_path = _make_fake_runtime(tmp_path)
    admin_requests: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_request(self, url, *, method, payload=None, timeout):
        del timeout
        admin_requests.append((method, url, payload))
        # Return appropriate responses
        if "/admin/pause" in url:
            return {"status": "inference_paused"}
        elif "/admin/resume" in url:
            return {"status": "inference_resumed"}
        elif "/v1/models" in url:
            return {"data": [{"id": "fake/model"}]}
        raise RuntimeError(f"Unexpected request to {url}")

    monkeypatch.setattr(VLLMServingBackend, "_reserve_port", lambda self: 8100)
    monkeypatch.setattr(VLLMServingBackend, "_spawn_process", lambda self, command: FakeProcess())
    monkeypatch.setattr(VLLMServingBackend, "_request_json", fake_request)

    backend = VLLMServingBackend(
        ServingConfig(
            model_name="fake/model",
            backend="vllm",
            runtime_python=str(python_path),
        )
    )

    try:
        admin_requests.clear()
        # Pause inference
        backend.pause_inference()
        assert len(admin_requests) == 1
        method, url, payload = admin_requests[0]
        assert method == "POST"
        assert "/admin/pause" in url
        assert payload is None

        # Resume inference
        backend.resume_inference()
        assert len(admin_requests) == 2
        method, url, payload = admin_requests[1]
        assert method == "POST"
        assert "/admin/resume" in url
        assert payload is None
    finally:
        backend.close()
