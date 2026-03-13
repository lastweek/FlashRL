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
import tempfile
from threading import Lock, Thread
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

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
    stderr_lines: list[str] = field(default_factory=list)
    stderr_lock: Lock = field(default_factory=Lock)
    stderr_thread: Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stderr_tail(self) -> str:
        with self.stderr_lock:
            if not self.stderr_lines:
                return ""
            return "\n".join(self.stderr_lines[-20:])


class VLLMServingBackend(ServingBackend):
    """Serving backend backed by managed local `vllm serve` replicas."""

    def __init__(self, config: ServingConfig) -> None:
        self.config = config
        self.device: Any = config.device or "auto"
        self.generation_defaults: dict[str, Any] = {}
        self._closed = False
        self._runtime_python = self._resolve_runtime_python(config.runtime_python)
        self._vllm_executable = self._resolve_vllm_executable(self._runtime_python)
        self._snapshot_dir: Path | None = None
        self._replicas: list[_Replica] = []

        self._validate_environment()

        try:
            self._replicas = self._launch_replicas(self.config.model_name)
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
        if not self._runtime_python.exists():
            raise ValueError(f"vllm runtime_python does not exist: {self._runtime_python}")
        if not os.access(self._runtime_python, os.X_OK):
            raise ValueError(f"vllm runtime_python is not executable: {self._runtime_python}")
        if not self._vllm_executable.exists():
            raise ValueError(f"vllm executable does not exist: {self._vllm_executable}")
        if not os.access(self._vllm_executable, os.X_OK):
            raise ValueError(f"vllm executable is not executable: {self._vllm_executable}")
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

    def _resolve_runtime_python(self, runtime_python: str | None) -> Path:
        candidate = runtime_python or "~/.venv-vllm/bin/python"
        return Path(candidate).expanduser().resolve()

    def _resolve_vllm_executable(self, runtime_python: Path) -> Path:
        candidates = [
            runtime_python.with_name("vllm"),
            runtime_python.with_name("vllm.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _launch_replicas(self, model_source: str) -> list[_Replica]:
        replicas: list[_Replica] = []
        try:
            for index in range(self.config.num_replicas):
                port = self._reserve_port()
                command = self._build_command(model_source, port)
                process = self._spawn_process(command)
                replica = _Replica(index=index, port=port, process=process)
                replica.stderr_thread = Thread(
                    target=self._drain_stderr,
                    args=(replica,),
                    daemon=True,
                )
                replica.stderr_thread.start()
                replicas.append(replica)

            for replica in replicas:
                self._wait_for_ready(replica)
            return replicas
        except Exception:
            self._stop_replica_group(replicas)
            raise

    def _restart_replicas(self, model_source: str) -> None:
        self._stop_replicas()
        self._replicas = self._launch_replicas(model_source)

    def _build_command(self, model_source: str, port: int) -> list[str]:
        return [
            str(self._vllm_executable),
            "serve",
            model_source,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            self.config.model_name,
            "--generation-config",
            "vllm",
            *self.config.vllm_args,
        ]

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
                replica.stderr_lines.append(normalized)
                if len(replica.stderr_lines) > 200:
                    replica.stderr_lines[:] = replica.stderr_lines[-200:]

    def _wait_for_ready(self, replica: _Replica) -> None:
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if replica.process.poll() is not None:
                stderr_tail = replica.stderr_tail()
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
                return
            except Exception as exc:  # pragma: no cover - exercised in retry loop via tests
                last_error = exc
                time.sleep(0.2)

        stderr_tail = replica.stderr_tail()
        detail = f"\n{last_error}" if last_error is not None else ""
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
        return parsed

    def _generate_chunk(
        self,
        replica: _Replica,
        prompts: list[str],
        group_size: int,
        generation_kwargs: dict[str, Any],
        start_index: int,
    ) -> tuple[int, list[list[GeneratedSample]]]:
        payload = {
            "model": self.config.model_name,
            "prompt": prompts,
            "n": int(group_size),
            "max_tokens": int(generation_kwargs.get("max_new_tokens", 512)),
            "temperature": (
                float(generation_kwargs.get("temperature", 1.0))
                if bool(generation_kwargs.get("do_sample", True))
                else 0.0
            ),
            "top_p": float(generation_kwargs.get("top_p", 0.9)),
            "top_k": int(generation_kwargs.get("top_k", 0)),
            "logprobs": 1,
            "return_token_ids": True,
            "stream": False,
        }
        response = self._request_json(
            f"{replica.base_url}/v1/completions",
            method="POST",
            payload=payload,
            timeout=120.0,
        )
        try:
            return start_index, self._parse_grouped_response(response, prompts, group_size)
        except Exception as exc:
            stderr_tail = replica.stderr_tail()
            if stderr_tail:
                raise RuntimeError(f"{exc}\n{stderr_tail}") from exc
            raise

    def _parse_grouped_response(
        self,
        response: dict[str, Any],
        prompts: list[str],
        group_size: int,
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
    ) -> list[int]:
        choice_prompt_ids = raw_choice.get("prompt_token_ids")
        if choice_prompt_ids is not None:
            return self._coerce_int_list(choice_prompt_ids)
        if prompt_token_ids_by_prompt is not None:
            return list(prompt_token_ids_by_prompt[prompt_index])
        raise RuntimeError("vllm completion response did not include prompt_token_ids.")

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
                else float(sum(response_token_logprobs))
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
            raise RuntimeError("vllm completion choice did not include logprobs.")

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
        self._replicas = []
        self._stop_replica_group(replicas)

    def _stop_replica_group(self, replicas: list[_Replica]) -> None:
        for replica in replicas:
            self._stop_replica(replica)

    def _stop_replica(self, replica: _Replica) -> None:
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
        if replica.process.stderr is not None:
            replica.process.stderr.close()

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
