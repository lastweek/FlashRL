"""Unit tests for the FastAPI-based vLLM wrapper server."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("vllm")
vllm_logprobs = pytest.importorskip("vllm.logprobs")
vllm_outputs = pytest.importorskip("vllm.outputs")
Logprob = vllm_logprobs.Logprob
CompletionOutput = vllm_outputs.CompletionOutput
RequestOutput = vllm_outputs.RequestOutput

from flashrl.framework.serving.vllm import server


pytestmark = pytest.mark.unit


class FakeTokenizer:
    """Tokenizer stub with deterministic ids and string conversions."""

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [ord(char) for char in prompt]
        if add_special_tokens:
            return [1, *tokens]
        return tokens

    def decode(self, token_id: int) -> str:
        if 32 <= token_id < 127:
            return chr(token_id)
        return f"<{token_id}>"

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.decode(token_id) for token_id in token_ids]


class DummyModelConfig:
    """Minimal model config surface consumed by the wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model = model_name
        self.max_model_len = 128
        self.logits_processor_pattern = None

    def get_diff_sampling_param(self) -> dict[str, object]:
        return {"temperature": 0.7}


class FakeLLM:
    """In-process fake that mimics the `vllm.LLM` surface used by the wrapper."""

    init_calls: list[dict[str, object]] = []
    shutdown_calls: list[str] = []
    generate_calls: list[dict[str, object]] = []
    logprobs_mode = "full"

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        FakeLLM.init_calls.append(self.kwargs)
        self.model_config = DummyModelConfig(str(kwargs["model"]))
        self.llm_engine = SimpleNamespace(
            shutdown=lambda: FakeLLM.shutdown_calls.append(str(kwargs["model"]))
        )

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def generate(self, prompts, sampling_params, *, use_tqdm: bool = True):
        FakeLLM.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        prompt_items = prompts if isinstance(prompts, list) else [prompts]
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompt_items)
        )
        tokenizer = self.get_tokenizer()
        outputs: list[RequestOutput] = []

        for prompt_index, (prompt_item, params_item) in enumerate(zip(prompt_items, params, strict=True)):
            if isinstance(prompt_item, dict):
                prompt_token_ids = list(prompt_item["prompt_token_ids"])
                prompt_text = "".join(
                    chr(token_id) for token_id in prompt_token_ids if 32 <= token_id < 127
                )
            else:
                prompt_text = str(prompt_item)
                prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

            prompt_logprobs = [None]
            for token_id in prompt_token_ids[1:]:
                prompt_logprobs.append(
                    {
                        token_id: Logprob(
                            logprob=-0.01 * token_id,
                            decoded_token=tokenizer.decode(token_id),
                        )
                    }
                )

            candidate_count = getattr(params_item, "n", 1)
            request_outputs = []
            for candidate_index in range(candidate_count):
                token_ids = [200 + candidate_index, 210 + candidate_index]
                if FakeLLM.logprobs_mode == "empty":
                    output_logprobs = []
                elif FakeLLM.logprobs_mode == "short":
                    output_logprobs = [
                        {
                            token_ids[0]: Logprob(
                                logprob=-0.3,
                                decoded_token=f"tok-{token_ids[0]}",
                            )
                        }
                    ]
                else:
                    output_logprobs = [
                        {
                            token_ids[0]: Logprob(
                                logprob=-0.3,
                                decoded_token=f"tok-{token_ids[0]}",
                            )
                        },
                        {
                            token_ids[1]: Logprob(
                                logprob=-0.4,
                                decoded_token=f"tok-{token_ids[1]}",
                            )
                        },
                    ]
                request_outputs.append(
                    CompletionOutput(
                        index=candidate_index,
                        text=f"::choice-{candidate_index}",
                        token_ids=token_ids,
                        cumulative_logprob=-0.7 - (candidate_index * 0.1),
                        logprobs=output_logprobs,
                        finish_reason="stop",
                    )
                )

            outputs.append(
                RequestOutput(
                    request_id=f"req-{prompt_index}",
                    prompt=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=prompt_logprobs,
                    outputs=request_outputs,
                    finished=True,
                )
            )

        return outputs


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch: pytest.MonkeyPatch):
    """Reset module globals and replace the live LLM with a fast fake."""
    FakeLLM.init_calls.clear()
    FakeLLM.shutdown_calls.clear()
    FakeLLM.generate_calls.clear()
    FakeLLM.logprobs_mode = "full"
    monkeypatch.setattr(server, "LLM", FakeLLM)
    monkeypatch.setattr(server, "app", None)
    monkeypatch.setattr(server, "llm", None)
    monkeypatch.setattr(server, "_engine_args", None)
    monkeypatch.setattr(server, "_model_source", "")
    monkeypatch.setattr(server, "_served_model_names", [])
    monkeypatch.setattr(server, "_default_sampling_params", {})
    monkeypatch.setattr(server, "_inference_paused", False)


def _build_client(*, vllm_args: list[str] | None = None) -> TestClient:
    application = server.create_app(
        "base/model",
        served_model_name="served-model",
        vllm_args=vllm_args,
    )
    return TestClient(application)


def test_create_app_passes_vllm_args_into_llm() -> None:
    """Wrapper startup should honor vLLM engine CLI arguments."""
    with _build_client(vllm_args=["--max-num-seqs=32", "--enable-prefix-caching"]) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert len(FakeLLM.init_calls) == 1
    assert FakeLLM.init_calls[0]["model"] == "base/model"
    assert FakeLLM.init_calls[0]["served_model_name"] == ["served-model"]
    assert FakeLLM.init_calls[0]["max_num_seqs"] == 32
    assert FakeLLM.init_calls[0]["enable_prefix_caching"] is True


def test_models_endpoint_matches_vllm_schema() -> None:
    """`/v1/models` should return a vLLM/OpenAI-compatible model list."""
    with _build_client() as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "served-model"
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "vllm"
    assert data["data"][0]["root"] == "base/model"
    assert data["data"][0]["max_model_len"] == 128
    assert isinstance(data["data"][0]["permission"], list)


def test_completions_endpoint_matches_vllm_response_shape() -> None:
    """`/v1/completions` should return the same non-streaming shape as vLLM."""
    request_payload = {
        "model": "served-model",
        "prompt": "ab",
        "n": 2,
        "max_tokens": 2,
        "logprobs": 1,
        "return_token_ids": True,
    }

    with _build_client() as client:
        response = client.post("/v1/completions", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["model"] == "served-model"
    assert data["usage"] == {
        "prompt_tokens": 3,
        "total_tokens": 7,
        "completion_tokens": 4,
        "prompt_tokens_details": None,
    }
    assert len(data["choices"]) == 2
    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["text"] == "::choice-0"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["prompt_token_ids"] == [1, 97, 98]
    assert data["choices"][0]["token_ids"] == [200, 210]
    assert data["choices"][0]["logprobs"]["token_logprobs"] == [-0.3, -0.4]
    assert data["choices"][0]["logprobs"]["tokens"] == ["tok-200", "tok-210"]
    assert data["choices"][1]["index"] == 1
    assert FakeLLM.generate_calls[0]["use_tqdm"] is False


def test_completions_endpoint_tolerates_missing_logprob_steps() -> None:
    """Missing sampled-token logprobs should not turn completions into HTTP 500s."""
    FakeLLM.logprobs_mode = "empty"

    with _build_client() as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "served-model",
                "prompt": "ab",
                "n": 1,
                "max_tokens": 2,
                "logprobs": 1,
                "return_token_ids": True,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["token_ids"] == [200, 210]
    assert data["choices"][0]["logprobs"]["token_logprobs"] == [None, None]
    assert data["choices"][0]["logprobs"]["top_logprobs"] == [None, None]
    assert data["choices"][0]["logprobs"]["tokens"] == ["<200>", "<210>"]


def test_completions_endpoint_pads_short_logprob_arrays() -> None:
    """Short logprob arrays should be padded instead of raising indexing errors."""
    FakeLLM.logprobs_mode = "short"

    with _build_client() as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "served-model",
                "prompt": "ab",
                "n": 1,
                "max_tokens": 2,
                "logprobs": 1,
                "return_token_ids": True,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["logprobs"]["token_logprobs"] == [-0.3, None]
    assert data["choices"][0]["logprobs"]["tokens"] == ["tok-200", "<210>"]
    assert data["choices"][0]["logprobs"]["top_logprobs"] == [
        {"tok-200": -0.3},
        None,
    ]


def test_tokenize_endpoint_matches_vllm_schema() -> None:
    """`/tokenize` should report tokens, count, and max_model_len like vLLM."""
    with _build_client() as client:
        response = client.post(
            "/tokenize",
            json={
                "model": "served-model",
                "prompt": "ab",
                "return_token_strs": True,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "count": 3,
        "max_model_len": 128,
        "tokens": [1, 97, 98],
        "token_strs": ["<1>", "a", "b"],
    }


def test_pause_and_resume_gate_completion_requests() -> None:
    """Admin pause should block completions until resumed."""
    with _build_client() as client:
        pause_response = client.post("/admin/pause")
        blocked_response = client.post(
            "/v1/completions",
            json={
                "model": "served-model",
                "prompt": "ab",
                "max_tokens": 2,
            },
        )
        resume_response = client.post("/admin/resume")
        ok_response = client.post(
            "/v1/completions",
            json={
                "model": "served-model",
                "prompt": "ab",
                "max_tokens": 2,
            },
        )

    assert pause_response.status_code == 200
    assert pause_response.json()["status"] == "inference_paused"
    assert blocked_response.status_code == 503
    assert blocked_response.json()["error"]["message"] == "Inference is paused."
    assert resume_response.status_code == 200
    assert resume_response.json()["status"] == "inference_resumed"
    assert ok_response.status_code == 200


def test_load_weights_from_disk_reloads_model_and_preserves_served_name() -> None:
    """Reloading weights should swap the model source without changing the served model id."""
    with _build_client() as client:
        reload_response = client.post(
            "/v1/load_weights_from_disk",
            json={"model_source": "/tmp/flashrl-snapshot"},
        )
        models_response = client.get("/v1/models")

    assert reload_response.status_code == 200
    assert reload_response.json() == {
        "status": "success",
        "model_source": "/tmp/flashrl-snapshot",
    }
    assert [call["model"] for call in FakeLLM.init_calls] == [
        "base/model",
        "/tmp/flashrl-snapshot",
    ]
    assert "base/model" in FakeLLM.shutdown_calls
    assert models_response.status_code == 200
    assert models_response.json()["data"][0]["id"] == "served-model"
    assert models_response.json()["data"][0]["root"] == "/tmp/flashrl-snapshot"
