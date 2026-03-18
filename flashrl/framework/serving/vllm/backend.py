"""Managed HTTP `vllm serve` backend."""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import time
from typing import Callable, TextIO
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from flashrl.framework.admin import build_admin_object, utc_now_iso
from flashrl.framework.config import ServingConfig
from flashrl.framework.models.actor import ActorModel, GeneratedSample
from flashrl.framework.serving.base import ServingBackend


_READY_TIMEOUT_SECONDS = 300.0
_RESERVED_VLLM_FLAGS = {
    "--host",
    "--port",
    "--served-model-name",
    "--generation-config",
}


@dataclass
class _Replica:
    """One managed `vllm serve` subprocess."""

    index: int
    port: int
    process: subprocess.Popen[str]
    model_source: str
    command: list[str]
    started_at: str = field(default_factory=utc_now_iso)
    ready_at: str | None = None
    phase: str = "Starting"
    last_error: str | None = None
    exit_code: int | None = None
    stderr_lines: list[str] = field(default_factory=list)
    stderr_lock: Lock = field(default_factory=Lock)
    stderr_thread: Thread | None = None
    stderr_file_path: Path | None = None
    stderr_file: TextIO | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stderr_tail(self) -> str:
        with self.stderr_lock:
            if not self.stderr_lines:
                return ""
            return "\n".join(self.stderr_lines[-20:])


class VLLMServingBackend(ServingBackend):
    """Serving backend backed by managed `vllm serve` replicas."""

    def __init__(
        self,
        config: ServingConfig,
        startup_logger: Callable[[str], None] | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.device: Any = config.device or "auto"
        self.generation_defaults: dict[str, Any] = {}
        self._closed = False
        self._startup_logger = startup_logger
        self._python_executable = self._resolve_python_executable()
        self._vllm_executable = self._resolve_vllm_executable(self._python_executable)
        self._snapshot_dir: Path | None = None
        self._replicas: list[_Replica] = []
        self._admin_replicas: list[_Replica] = []
        self._log_dir = Path(log_dir) if log_dir else None

        self._validate_environment()

        try:
            self._replicas = self._launch_initial_replicas()
            self._admin_replicas = list(self._replicas)
            atexit.register(self.close)
        except Exception:
            self.close()
            raise

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return [sample.text for sample in self.generate_batch(prompts, **kwargs)]

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[GeneratedSample]:
        grouped = self.generate_grouped(prompts, group_size=1, **kwargs)
        return [candidates[0] for candidates in grouped]

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        **kwargs: Any,
    ) -> list[list[GeneratedSample]]:
        if self._closed:
            raise RuntimeError("vllm backend is closed.")
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if not prompts:
            return []
        if not self._replicas:
            raise RuntimeError("vllm replicas are not running.")

        generation_kwargs = {
            **self.generation_defaults,
            **kwargs,
        }
        ranges = self._prompt_ranges(len(prompts), min(len(self._replicas), len(prompts)))
        grouped_outputs: list[list[GeneratedSample] | None] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = []
            for replica, (start, end) in zip(self._replicas, ranges, strict=True):
                futures.append(
                    executor.submit(
                        self._generate_chunk,
                        replica,
                        prompts[start:end],
                        group_size,
                        generation_kwargs,
                        start,
                    )
                )

            for future in futures:
                start_index, chunk_outputs = future.result()
                for offset, prompt_outputs in enumerate(chunk_outputs):
                    grouped_outputs[start_index + offset] = prompt_outputs

        return [
            prompt_outputs
            for prompt_outputs in grouped_outputs
            if prompt_outputs is not None
        ]

    def set_generation_defaults(self, **kwargs: Any) -> None:
        self.generation_defaults = dict(kwargs)

    def list_admin_objects(self) -> list[dict[str, Any]]:
        """Return one admin object per managed vLLM replica."""
        items: list[dict[str, Any]] = []
        for replica in self._admin_replicas:
            exit_code = replica.process.poll()
            if exit_code is not None:
                replica.exit_code = int(exit_code)
            phase = replica.phase
            if exit_code is not None and phase not in {"Closed", "Failed"}:
                phase = "Exited" if int(exit_code) == 0 else "Failed"
            items.append(
                build_admin_object(
                    "VLLMInstance",
                    f"vllm-instance-{replica.index}",
                    uid=f"vllm:{self.config.model_name}:{replica.index}",
                    created_at=replica.started_at,
                    labels={
                        "flashrl.dev/serving-backend": "vllm",
                        "flashrl.dev/model-name": self.config.model_name,
                    },
                    spec={
                        "replicaIndex": replica.index,
                        "host": "127.0.0.1",
                        "port": replica.port,
                        "modelSource": replica.model_source,
                        "servedModelName": self.config.model_name,
                        "pythonExecutable": str(self._python_executable),
                        "command": list(replica.command),
                    },
                    status={
                        "phase": phase,
                        "pid": getattr(replica.process, "pid", None),
                        "healthy": phase == "Ready" and exit_code is None,
                        "startedAt": replica.started_at,
                        "readyAt": replica.ready_at,
                        "exitCode": replica.exit_code,
                        "stderrTail": replica.stderr_tail(),
                        "lastError": replica.last_error,
                    },
                )
            )
        return items

    def sync_from_training_actor(self, training_actor: ActorModel) -> None:
        snapshot_dir = self._write_snapshot(training_actor)
        self._restart_replicas(str(snapshot_dir))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(ValueError):
            atexit.unregister(self.close)
        self._stop_replicas()
        if self._snapshot_dir is not None:
            shutil.rmtree(self._snapshot_dir, ignore_errors=True)
            self._snapshot_dir = None

    def _validate_environment(self) -> None:
        if self.config.debug_live_rollout:
            raise ValueError("vllm does not support serving.debug_live_rollout in v1.")
        if not self._python_executable.exists():
            if self.config.runtime_python is not None:
                raise ValueError(f"vllm runtime_python does not exist: {self._python_executable}")
            raise ValueError(f"Current Python executable does not exist: {self._python_executable}")
        if not os.access(self._python_executable, os.X_OK):
            if self.config.runtime_python is not None:
                raise ValueError(
                    f"vllm runtime_python is not executable: {self._python_executable}"
                )
            raise ValueError(f"Current Python executable is not executable: {self._python_executable}")
        self._ensure_vllm_importable()
        if not self._vllm_executable.exists():
            raise ValueError(
                "Managed vllm serving requires the `vllm` console script next to the selected "
                f"Python runtime. Expected at {self._vllm_executable}. "
                "Install vllm into that runtime or point serving.runtime_python at a prepared "
                "vllm environment."
            )
        if not os.access(self._vllm_executable, os.X_OK):
            raise ValueError(
                "Managed vllm serving requires an executable `vllm` console script next to the "
                "selected Python runtime. "
                f"Found non-executable path at {self._vllm_executable}."
            )
        for arg in self.config.vllm_args:
            if not isinstance(arg, str) or not arg.strip():
                raise ValueError("serving.vllm_args entries must be non-empty strings.")
            if not arg.startswith("-"):
                raise ValueError(
                    "serving.vllm_args must contain full CLI flag entries such as "
                    "'--max-model-len=4096'."
                )
            if any(arg == flag or arg.startswith(f"{flag}=") for flag in _RESERVED_VLLM_FLAGS):
                raise ValueError(f"serving.vllm_args may not override reserved flag: {arg}")

    def _resolve_python_executable(self) -> Path:
        python_path = self.config.runtime_python or sys.executable
        return Path(os.path.abspath(os.path.expanduser(str(python_path))))

    def _ensure_vllm_importable(self) -> None:
        probe = subprocess.run(
            [str(self._python_executable), "-c", "import vllm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return
        detail = (probe.stderr or probe.stdout or "").strip()
        message = (
            "Managed vllm serving requires the selected Python runtime to import `vllm`. "
            f"Checked {self._python_executable}. Install `vllm` into that runtime or point "
            "serving.runtime_python at a prepared vllm environment."
        )
        if detail:
            message = f"{message}\n{detail}"
        raise ValueError(message)

    def _resolve_vllm_executable(self, python_executable: Path) -> Path:
        candidates = [
            python_executable.with_name("vllm"),
            python_executable.with_name("vllm.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _launch_replicas(self, model_source: str) -> list[_Replica]:
        replicas: list[_Replica] = []
        vllm_log_dir: Path | None = None
        if self._log_dir is not None:
            vllm_log_dir = self._log_dir / "vllm"
            vllm_log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index in range(self.config.num_replicas):
                port = self._reserve_port()
                command = self._build_command(model_source, port)
                process = self._spawn_process(command)
                stderr_file_path: Path | None = None
                stderr_file: TextIO | None = None
                if vllm_log_dir is not None:
                    stderr_file_path = vllm_log_dir / f"replica-{index}.log"
                    stderr_file = open(stderr_file_path, "a", encoding="utf-8")
                    stderr_file.write(f"\n{'=' * 72}\n")
                    stderr_file.write(f"VLLM Replica {index} started at {utc_now_iso()}\n")
                    stderr_file.write(f"Command: {' '.join(command)}\n")
                    stderr_file.write(f"{'=' * 72}\n\n")
                    stderr_file.flush()
                replica = _Replica(
                    index=index,
                    port=port,
                    process=process,
                    model_source=model_source,
                    command=command,
                    stderr_file_path=stderr_file_path,
                    stderr_file=stderr_file,
                )
                replica.stderr_thread = Thread(
                    target=self._drain_stderr,
                    args=(replica,),
                    daemon=True,
                )
                replica.stderr_thread.start()
                replicas.append(replica)

            for replica in replicas:
                self._wait_for_ready(replica)
            self._admin_replicas = list(replicas)
            return replicas
        except Exception:
            self._stop_replica_group(replicas)
            self._admin_replicas = list(replicas)
            raise

    def _launch_initial_replicas(self) -> list[_Replica]:
        model_source = self.config.model_name
        try:
            return self._launch_replicas(model_source)
        except Exception:
            if self._is_local_model_source(model_source):
                raise
            local_snapshot = self._download_model_snapshot(model_source)
            return self._launch_replicas(local_snapshot)

    def _is_local_model_source(self, model_source: str) -> bool:
        return Path(os.path.expanduser(model_source)).exists()

    def _download_model_snapshot(self, model_name: str) -> str:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(repo_id=model_name)
        return str(Path(snapshot_path))

    def _restart_replicas(self, model_source: str) -> None:
        self._stop_replicas()
        self._replicas = self._launch_replicas(model_source)

    def _build_command(self, model_source: str, port: int) -> list[str]:
        command = [
            str(self._vllm_executable),
            "serve",
            model_source,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            self.config.model_name,
        ]
        command.extend(self.config.vllm_args)
        return command

    def _spawn_process(self, command: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=(os.name != "nt"),
        )

    def _drain_stderr(self, replica: _Replica) -> None:
        if replica.process.stderr is None:
            return
        for line in replica.process.stderr:
            normalized = line.rstrip()
            if not normalized:
                continue
            with replica.stderr_lock:
                if replica.stderr_file is not None:
                    replica.stderr_file.write(normalized + "\n")
                    replica.stderr_file.flush()
                replica.stderr_lines.append(normalized)
                if len(replica.stderr_lines) > 200:
                    replica.stderr_lines[:] = replica.stderr_lines[-200:]
            if replica.stderr_file is None:
                self._maybe_emit_startup_log(replica, normalized)

    def _maybe_emit_startup_log(self, replica: _Replica, line: str) -> None:
        """Surface replica initialization stderr lines in the parent console."""
        if self._startup_logger is None:
            return
        if replica.stderr_file is not None:
            return
        if replica.ready_at is not None or replica.phase != "Starting":
            return
        self._startup_logger(f"vllm[{replica.index}] {line}")

    def _wait_for_ready(self, replica: _Replica) -> None:
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if replica.process.poll() is not None:
                replica.exit_code = int(replica.process.poll() or 0)
                replica.phase = "Failed"
                stderr_tail = replica.stderr_tail()
                replica.last_error = "vllm replica exited before becoming ready."
                raise RuntimeError(
                    "vllm replica exited before becoming ready."
                    + (f"\n{stderr_tail}" if stderr_tail else "")
                )
            try:
                self._request_json(
                    f"{replica.base_url}/v1/models",
                    method="GET",
                    timeout=5.0,
                )
                replica.phase = "Ready"
                replica.ready_at = utc_now_iso()
                return
            except Exception as exc:  # pragma: no cover - exercised in retry loop via tests
                last_error = exc
                replica.last_error = str(exc)
                time.sleep(0.2)

        stderr_tail = replica.stderr_tail()
        detail = f"\n{last_error}" if last_error is not None else ""
        replica.phase = "Failed"
        replica.last_error = (
            f"Timed out waiting for vllm replica on port {replica.port}."
            + (detail if detail else "")
        )
        raise RuntimeError(
            f"Timed out waiting for vllm replica on port {replica.port}.{detail}"
            + (f"\n{stderr_tail}" if stderr_tail else "")
        )

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib_request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:  # pragma: no cover - depends on live server behavior
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vllm HTTP {exc.code} from {url}: {body}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"vllm request failed for {url}: {exc}") from exc

        if not body:
            return {}
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"vllm returned a non-dict JSON payload from {url}.")
        if isinstance(parsed.get("error"), dict):
            message = parsed["error"].get("message") or parsed["error"]
            raise RuntimeError(f"vllm returned an error payload from {url}: {message}")
        return parsed

    def _generate_chunk(
        self,
        replica: _Replica,
        prompts: list[str],
        group_size: int,
        generation_kwargs: dict[str, Any],
        start_index: int,
    ) -> tuple[int, list[list[GeneratedSample]]]:
        try:
            outputs: list[list[GeneratedSample]] = []
            for prompt in prompts:
                response = self._request_json(
                    f"{replica.base_url}/v1/completions",
                    method="POST",
                    payload=self._build_completion_payload(prompt, group_size, generation_kwargs),
                    timeout=120.0,
                )
                outputs.extend(
                    self._parse_grouped_response(
                        response,
                        [prompt],
                        group_size,
                        replica=replica,
                    )
                )
            return start_index, outputs
        except Exception as exc:
            replica.last_error = f"{type(exc).__name__}: {exc}"
            stderr_tail = replica.stderr_tail()
            if stderr_tail:
                raise RuntimeError(f"{exc}\n{stderr_tail}") from exc
            raise

    def _build_completion_payload(
        self,
        prompt: str,
        group_size: int,
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "n": int(group_size),
            "max_tokens": int(generation_kwargs.get("max_new_tokens", 512)),
            "temperature": (
                float(generation_kwargs.get("temperature", 1.0))
                if bool(generation_kwargs.get("do_sample", True))
                else 0.0
            ),
            "top_p": float(generation_kwargs.get("top_p", 0.9)),
            "top_k": int(generation_kwargs.get("top_k", 0)),
            "return_token_ids": True,
            "stream": False,
        }
        return payload

    def _parse_grouped_response(
        self,
        response: dict[str, Any],
        prompts: list[str],
        group_size: int,
        replica: _Replica | None = None,
    ) -> list[list[GeneratedSample]]:
        raw_choices = response.get("choices")
        if not isinstance(raw_choices, list):
            raise RuntimeError("vllm completion response did not include choices.")

        grouped: list[list[GeneratedSample]] = [[] for _ in prompts]
        prompt_token_ids_by_prompt = self._coerce_prompt_token_ids(
            response.get("prompt_token_ids"),
            len(prompts),
        )
        for choice_position, raw_choice in enumerate(raw_choices):
            if not isinstance(raw_choice, dict):
                raise RuntimeError("vllm returned a malformed completion choice.")
            prompt_index = self._choice_prompt_index(
                raw_choice,
                choice_position,
                len(prompts),
                group_size,
            )
            prompt_token_ids = self._choice_prompt_token_ids(
                raw_choice,
                prompt_token_ids_by_prompt,
                prompt_index,
                prompts,
                replica,
            )
            grouped[prompt_index].append(self._parse_choice(raw_choice, prompt_token_ids))

        for prompt_index, candidates in enumerate(grouped):
            if len(candidates) != group_size:
                raise RuntimeError(
                    "vllm returned an unexpected number of candidates for "
                    f"prompt_index={prompt_index}: expected {group_size}, got {len(candidates)}."
                )
        return grouped

    def _choice_prompt_index(
        self,
        raw_choice: dict[str, Any],
        choice_position: int,
        prompt_count: int,
        group_size: int,
    ) -> int:
        for key in ("prompt_index", "prompt_idx"):
            value = raw_choice.get(key)
            if isinstance(value, int):
                if 0 <= value < prompt_count:
                    return value
                raise RuntimeError(f"vllm returned out-of-range {key}={value}.")
        if prompt_count == 1:
            return 0
        prompt_index = choice_position // group_size
        if prompt_index >= prompt_count:
            raise RuntimeError("vllm returned more choices than expected for the prompt batch.")
        return prompt_index

    def _choice_prompt_token_ids(
        self,
        raw_choice: dict[str, Any],
        prompt_token_ids_by_prompt: list[list[int]] | None,
        prompt_index: int,
        prompts: list[str],
        replica: _Replica | None,
    ) -> list[int]:
        choice_prompt_ids = raw_choice.get("prompt_token_ids")
        if choice_prompt_ids is not None:
            return self._coerce_int_list(choice_prompt_ids)
        if prompt_token_ids_by_prompt is not None:
            return list(prompt_token_ids_by_prompt[prompt_index])
        if replica is not None:
            return self._tokenize_prompt(replica, prompts[prompt_index])
        raise RuntimeError("vllm completion response did not include prompt_token_ids.")

    def _tokenize_prompt(self, replica: _Replica, prompt: str) -> list[int]:
        response = self._request_json(
            f"{replica.base_url}/tokenize",
            method="POST",
            payload={
                "model": self.config.model_name,
                "prompt": prompt,
            },
            timeout=30.0,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("vllm tokenize response did not include tokens.")
        return self._coerce_int_list(tokens)

    def _parse_choice(
        self,
        raw_choice: dict[str, Any],
        prompt_token_ids: list[int],
    ) -> GeneratedSample:
        response_token_ids = self._coerce_int_list(raw_choice.get("token_ids"))
        if not response_token_ids:
            raise RuntimeError("vllm completion choice did not include token_ids.")
        response_token_logprobs = self._extract_output_logprobs(
            raw_choice.get("logprobs"),
            response_token_ids,
        )
        cumulative_logprob = raw_choice.get("cumulative_logprob")
        metadata = {
            key: value
            for key, value in {
                "finish_reason": raw_choice.get("finish_reason"),
                "stop_reason": raw_choice.get("stop_reason"),
            }.items()
            if value is not None
        }
        return GeneratedSample(
            text=str(raw_choice.get("text", "")),
            prompt_token_ids=list(prompt_token_ids),
            response_token_ids=response_token_ids,
            response_token_logprobs=response_token_logprobs,
            log_prob=(
                float(cumulative_logprob)
                if cumulative_logprob is not None
                else float(sum(response_token_logprobs)) if response_token_logprobs else 0.0
            ),
            metadata=metadata,
        )

    def _coerce_prompt_token_ids(
        self,
        raw_prompt_token_ids: Any,
        prompt_count: int,
    ) -> list[list[int]] | None:
        if raw_prompt_token_ids is None:
            return None
        if prompt_count == 1 and isinstance(raw_prompt_token_ids, list):
            if all(not isinstance(item, (list, tuple)) for item in raw_prompt_token_ids):
                return [self._coerce_int_list(raw_prompt_token_ids)]
        if not isinstance(raw_prompt_token_ids, list) or len(raw_prompt_token_ids) != prompt_count:
            raise RuntimeError("vllm returned malformed prompt_token_ids.")
        return [self._coerce_int_list(item) for item in raw_prompt_token_ids]

    def _coerce_int_list(self, values: Any) -> list[int]:
        if values is None:
            return []
        return [int(value) for value in list(values)]

    def _extract_output_logprobs(self, logprobs: Any, token_ids: list[int]) -> list[float]:
        if not token_ids:
            return []
        if logprobs is None:
            return []

        token_logprobs: Any = None
        if isinstance(logprobs, dict):
            if isinstance(logprobs.get("token_logprobs"), list):
                token_logprobs = logprobs["token_logprobs"]
            elif isinstance(logprobs.get("content"), list):
                token_logprobs = logprobs["content"]
        elif isinstance(logprobs, list):
            token_logprobs = logprobs

        if not isinstance(token_logprobs, list) or len(token_logprobs) != len(token_ids):
            raise RuntimeError(
                "vllm returned logprobs with a length that does not match token_ids."
            )
        return [
            self._extract_one_logprob(entry, token_id)
            for entry, token_id in zip(token_logprobs, token_ids, strict=True)
        ]

    def _extract_one_logprob(self, step_entry: Any, token_id: int) -> float:
        if isinstance(step_entry, (int, float)):
            return float(step_entry)
        if isinstance(step_entry, dict):
            if "logprob" in step_entry:
                return float(step_entry["logprob"])
            if token_id in step_entry:
                return self._extract_one_logprob(step_entry[token_id], token_id)
            if str(token_id) in step_entry:
                return self._extract_one_logprob(step_entry[str(token_id)], token_id)
            for value in step_entry.values():
                try:
                    return self._extract_one_logprob(value, token_id)
                except (TypeError, ValueError, RuntimeError):
                    continue
        if isinstance(step_entry, (list, tuple)):
            for value in step_entry:
                try:
                    return self._extract_one_logprob(value, token_id)
                except (TypeError, ValueError, RuntimeError):
                    continue
        raise RuntimeError(f"Unable to extract sampled token logprob for token_id={token_id}.")

    def _write_snapshot(self, training_actor: ActorModel) -> Path:
        snapshot_dir = self._ensure_snapshot_dir()
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        training_actor.model.save_pretrained(snapshot_dir, safe_serialization=True)
        training_actor.tokenizer.save_pretrained(snapshot_dir)
        return snapshot_dir

    def _ensure_snapshot_dir(self) -> Path:
        if self._snapshot_dir is None:
            self._snapshot_dir = Path(tempfile.mkdtemp(prefix="flashrl-vllm-")).resolve()
        return self._snapshot_dir

    def _stop_replicas(self) -> None:
        replicas = self._replicas
        if replicas:
            self._admin_replicas = list(replicas)
        self._stop_replica_group(replicas)
        self._replicas = []

    def _stop_replica_group(self, replicas: list[_Replica]) -> None:
        for replica in replicas:
            self._stop_replica(replica)

    def _stop_replica(self, replica: _Replica) -> None:
        replica.phase = "Closing"
        if replica.process.poll() is None:
            if os.name != "nt" and getattr(replica.process, "pid", None) is not None:
                process_group_id = os.getpgid(replica.process.pid)
                os.killpg(process_group_id, signal.SIGTERM)
                try:
                    replica.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process_group_id, signal.SIGKILL)
                    replica.process.wait(timeout=5)
            else:
                replica.process.terminate()
                try:
                    replica.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    replica.process.kill()
                    replica.process.wait(timeout=5)
        replica.exit_code = replica.process.poll()
        replica.phase = "Closed"
        if replica.process.stderr is not None:
            replica.process.stderr.close()
        if replica.stderr_file is not None:
            replica.stderr_file.write(f"\n{'=' * 72}\n")
            replica.stderr_file.write(f"VLLM Replica {replica.index} stopped at {utc_now_iso()}\n")
            replica.stderr_file.write(f"Exit code: {replica.exit_code}\n")
            replica.stderr_file.write(f"{'=' * 72}\n\n")
            replica.stderr_file.close()
            replica.stderr_file = None

    def _reserve_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    def _prompt_ranges(self, prompt_count: int, shard_count: int) -> list[tuple[int, int]]:
        base, remainder = divmod(prompt_count, shard_count)
        ranges: list[tuple[int, int]] = []
        start = 0
        for shard_index in range(shard_count):
            size = base + (1 if shard_index < remainder else 0)
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges


__all__ = ["VLLMServingBackend"]
