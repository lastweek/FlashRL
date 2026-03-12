"""Separate-runtime `vllm_metal` serving backend and worker entrypoint."""

from __future__ import annotations

import argparse
import atexit
from contextlib import suppress
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import traceback
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from flashrl.framework.config import ServingConfig
from flashrl.framework.models.actor import ActorModel, GeneratedSample
from flashrl.framework.serving.base import ServingBackend


class _WorkerRpcClient:
    """Small JSON-RPC client backed by one worker subprocess."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self._request_id = 0
        self._lock = Lock()
        self._stderr_lines: list[str] = []
        self._stderr_thread = Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self.process.stderr is None:
            return
        for line in self.process.stderr:
            normalized = line.rstrip()
            if not normalized:
                continue
            self._stderr_lines.append(normalized)
            if len(self._stderr_lines) > 200:
                self._stderr_lines = self._stderr_lines[-200:]

    def _stderr_tail(self) -> str:
        if not self._stderr_lines:
            return ""
        return "\n".join(self._stderr_lines[-20:])

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.process.poll() is not None:
                stderr_tail = self._stderr_tail()
                raise RuntimeError(
                    "vllm_metal worker exited unexpectedly."
                    + (f"\n{stderr_tail}" if stderr_tail else "")
                )

            self._request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params or {},
            }
            if self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("vllm_metal worker pipes are unavailable.")

            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()

            while True:
                line = self.process.stdout.readline()
                if line == "":
                    stderr_tail = self._stderr_tail()
                    raise RuntimeError(
                        "vllm_metal worker closed stdout unexpectedly."
                        + (f"\n{stderr_tail}" if stderr_tail else "")
                    )
                normalized = line.strip()
                if not normalized:
                    continue
                try:
                    response = json.loads(normalized)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON-RPC response from vllm_metal worker: {normalized}"
                    ) from exc
                if response.get("id") != self._request_id:
                    raise RuntimeError(
                        "Out-of-order response from vllm_metal worker: "
                        f"expected id={self._request_id}, got id={response.get('id')}."
                    )
                error = response.get("error")
                if error is not None:
                    message = error.get("message", "unknown worker error")
                    traceback_text = error.get("data", {}).get("traceback")
                    stderr_tail = self._stderr_tail()
                    details = [message]
                    if traceback_text:
                        details.append(traceback_text)
                    if stderr_tail:
                        details.append(stderr_tail)
                    raise RuntimeError("\n".join(details))
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("vllm_metal worker returned a non-dict JSON-RPC result.")
                return result

    def close(self) -> None:
        with self._lock:
            if self.process.poll() is None:
                try:
                    if self.process.stdin is not None:
                        self.process.stdin.write(
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": 0,
                                    "method": "shutdown",
                                    "params": {},
                                }
                            )
                            + "\n"
                        )
                        self.process.stdin.flush()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
            if self.process.stdin is not None:
                self.process.stdin.close()
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()


class WorkerError(RuntimeError):
    """Raised when the worker cannot fulfill one RPC call."""


class _VLLMRuntime:
    """Thin wrapper around the vLLM Python API."""

    def __init__(
        self,
        model_path: str,
        dtype: str,
        max_model_len: int,
        trust_remote_code: bool,
    ) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self._llm: Any = None
        self._sampling_params_cls: Any = None
        self.reload_weights()

    def reload_weights(self) -> dict[str, Any]:
        if self._llm is not None:
            del self._llm
            gc.collect()

        from vllm import LLM, SamplingParams

        self._sampling_params_cls = SamplingParams
        self._llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
        )
        return {"reloaded": True}

    def health(self) -> dict[str, Any]:
        return {"ready": True}

    def generate_grouped(
        self,
        prompts: list[str],
        group_size: int,
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        sampling_params = self._sampling_params_cls(
            **self._sampling_params_kwargs(group_size, generation_kwargs)
        )
        outputs = self._llm.generate(prompts, sampling_params)
        grouped_samples: list[list[dict[str, Any]]] = []
        for request_output in outputs:
            prompt_token_ids = self._to_int_list(
                getattr(request_output, "prompt_token_ids", None)
            )
            prompt_samples: list[dict[str, Any]] = []
            for completion in getattr(request_output, "outputs", []):
                response_token_ids = self._to_int_list(getattr(completion, "token_ids", None))
                response_token_logprobs = self._extract_output_logprobs(
                    getattr(completion, "logprobs", None),
                    response_token_ids,
                )
                cumulative_logprob = getattr(completion, "cumulative_logprob", None)
                prompt_samples.append(
                    {
                        "text": str(getattr(completion, "text", "")),
                        "prompt_token_ids": prompt_token_ids,
                        "response_token_ids": response_token_ids,
                        "response_token_logprobs": response_token_logprobs,
                        "log_prob": (
                            float(cumulative_logprob)
                            if cumulative_logprob is not None
                            else float(sum(response_token_logprobs))
                        ),
                        "metadata": {
                            key: value
                            for key, value in {
                                "finish_reason": getattr(completion, "finish_reason", None),
                                "stop_reason": getattr(completion, "stop_reason", None),
                            }.items()
                            if value is not None
                        },
                    }
                )
            grouped_samples.append(prompt_samples)
        return {"grouped_samples": grouped_samples}

    def _sampling_params_kwargs(
        self,
        group_size: int,
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 512))
        do_sample = bool(generation_kwargs.get("do_sample", True))
        temperature = float(generation_kwargs.get("temperature", 1.0))
        params: dict[str, Any] = {
            "n": int(group_size),
            "max_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": float(generation_kwargs.get("top_p", 0.9)),
            "logprobs": 1,
        }
        top_k = int(generation_kwargs.get("top_k", 0))
        if top_k > 0:
            params["top_k"] = top_k
        return params

    def _to_int_list(self, values: Any) -> list[int]:
        if values is None:
            return []
        return [int(value) for value in list(values)]

    def _extract_output_logprobs(self, logprobs: Any, token_ids: list[int]) -> list[float]:
        if not token_ids:
            return []
        if logprobs is None:
            raise WorkerError("vLLM did not return output logprobs for generated tokens.")
        values = list(logprobs)
        if len(values) != len(token_ids):
            raise WorkerError(
                "vLLM returned output logprobs with a length that does not match token_ids."
            )
        return [
            self._extract_one_logprob(step_entry, token_id)
            for step_entry, token_id in zip(values, token_ids, strict=True)
        ]

    def _extract_one_logprob(self, step_entry: Any, token_id: int) -> float:
        if isinstance(step_entry, (int, float)):
            return float(step_entry)
        if hasattr(step_entry, "logprob"):
            return float(getattr(step_entry, "logprob"))
        if isinstance(step_entry, dict):
            if token_id in step_entry:
                return self._extract_one_logprob(step_entry[token_id], token_id)
            if str(token_id) in step_entry:
                return self._extract_one_logprob(step_entry[str(token_id)], token_id)
            if "logprob" in step_entry:
                return float(step_entry["logprob"])
            for value in step_entry.values():
                try:
                    return self._extract_one_logprob(value, token_id)
                except (TypeError, ValueError, WorkerError):
                    continue
        if isinstance(step_entry, (list, tuple)):
            for value in step_entry:
                try:
                    return self._extract_one_logprob(value, token_id)
                except (TypeError, ValueError, WorkerError):
                    continue
        raise WorkerError(f"Unable to extract sampled token logprob for token_id={token_id}.")


class VLLMMetalServingBackend(ServingBackend):
    """Serving backend backed by a local `vllm-metal` worker runtime."""

    def __init__(self, config: ServingConfig, training_actor: ActorModel) -> None:
        self.config = config
        self.device = (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.generation_defaults: dict[str, Any] = {}
        self._closed = False
        self._runtime_python = self._resolve_runtime_python(config.runtime_python)
        self._snapshot_dir = Path(tempfile.mkdtemp(prefix="flashrl-vllm-metal-")).resolve()
        self._rpc_client: _WorkerRpcClient | None = None

        self._validate_environment()

        try:
            self._write_snapshot(training_actor)
            self._rpc_client = self._launch_worker()
            self._rpc_client.call("health")
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
        if self._rpc_client is None:
            raise RuntimeError("vllm_metal worker is not running.")
        payload = self._rpc_client.call(
            "generate_grouped",
            {
                "prompts": prompts,
                "group_size": group_size,
                "generation_kwargs": {
                    **self.generation_defaults,
                    **kwargs,
                },
            },
        )
        raw_grouped = payload.get("grouped_samples")
        if not isinstance(raw_grouped, list):
            raise RuntimeError("vllm_metal worker returned invalid grouped samples.")
        grouped: list[list[GeneratedSample]] = []
        for raw_prompt_group in raw_grouped:
            if not isinstance(raw_prompt_group, list):
                raise RuntimeError("vllm_metal worker returned a malformed prompt group.")
            prompt_group: list[GeneratedSample] = []
            for raw_sample in raw_prompt_group:
                if not isinstance(raw_sample, dict):
                    raise RuntimeError("vllm_metal worker returned a malformed sample.")
                prompt_group.append(
                    GeneratedSample(
                        text=str(raw_sample["text"]),
                        prompt_token_ids=[int(token_id) for token_id in raw_sample["prompt_token_ids"]],
                        response_token_ids=[
                            int(token_id) for token_id in raw_sample["response_token_ids"]
                        ],
                        response_token_logprobs=[
                            float(value) for value in raw_sample["response_token_logprobs"]
                        ],
                        log_prob=float(raw_sample["log_prob"]),
                        metadata=dict(raw_sample.get("metadata", {})),
                    )
                )
            grouped.append(prompt_group)
        return grouped

    def set_generation_defaults(self, **kwargs: Any) -> None:
        self.generation_defaults = dict(kwargs)

    def sync_from_training_actor(self, training_actor: ActorModel) -> None:
        if self._rpc_client is None:
            raise RuntimeError("vllm_metal worker is not running.")
        self._write_snapshot(training_actor)
        self._rpc_client.call("reload_weights")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(ValueError):
            atexit.unregister(self.close)
        if self._rpc_client is not None:
            self._rpc_client.close()
            self._rpc_client = None
        shutil.rmtree(self._snapshot_dir, ignore_errors=True)

    def _validate_environment(self) -> None:
        if self.config.debug_live_rollout:
            raise ValueError("vllm_metal does not support serving.debug_live_rollout in v1.")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("vllm_metal serving is only supported on macOS arm64.")
        if not self._runtime_python.exists():
            raise ValueError(
                f"vllm_metal runtime_python does not exist: {self._runtime_python}"
            )
        if not os.access(self._runtime_python, os.X_OK):
            raise ValueError(
                f"vllm_metal runtime_python is not executable: {self._runtime_python}"
            )

    def _resolve_runtime_python(self, runtime_python: str | None) -> Path:
        candidate = runtime_python or "~/.venv-vllm-metal/bin/python"
        return Path(candidate).expanduser().resolve()

    def _write_snapshot(self, training_actor: ActorModel) -> None:
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        training_actor.model.save_pretrained(self._snapshot_dir, safe_serialization=True)
        training_actor.tokenizer.save_pretrained(self._snapshot_dir)

    def _launch_worker(self) -> _WorkerRpcClient:
        process = subprocess.Popen(
            [
                str(self._runtime_python),
                str(Path(__file__).resolve()),
                "--worker",
                "--model-path",
                str(self._snapshot_dir),
                "--dtype",
                self.config.dtype,
                "--max-model-len",
                str(self.config.max_length),
                *(["--trust-remote-code"] if self.config.trust_remote_code else []),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return _WorkerRpcClient(process)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashRL vllm_metal worker.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--dtype")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _success_response(request_id: int | None, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: int | None, exc: Exception) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32000,
            "message": str(exc),
            "data": {
                "traceback": traceback.format_exc(),
            },
        },
    }


def _run_worker(args: argparse.Namespace) -> int:
    runtime = _VLLMRuntime(
        model_path=args.model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id: int | None = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request["method"]
            params = request.get("params", {})
            if method == "health":
                response = _success_response(request_id, runtime.health())
            elif method == "reload_weights":
                response = _success_response(request_id, runtime.reload_weights())
            elif method == "generate_grouped":
                response = _success_response(
                    request_id,
                    runtime.generate_grouped(
                        prompts=[str(prompt) for prompt in params.get("prompts", [])],
                        group_size=int(params.get("group_size", 1)),
                        generation_kwargs=dict(params.get("generation_kwargs", {})),
                    ),
                )
            elif method == "shutdown":
                response = _success_response(request_id, {"shutdown": True})
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                return 0
            else:
                raise WorkerError(f"Unsupported JSON-RPC method: {method}")
        except Exception as exc:
            response = _error_response(request_id, exc)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the embedded worker entrypoint."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("This module is only meant to be launched with --worker.")
    if args.model_path is None or args.dtype is None or args.max_model_len is None:
        parser.error("--worker requires --model-path, --dtype, and --max-model-len.")
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VLLMMetalServingBackend"]
