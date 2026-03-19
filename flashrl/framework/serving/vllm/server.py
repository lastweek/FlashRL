"""FastAPI vLLM wrapper used by FlashRL.

This wrapper keeps model execution in-process so FlashRL can swap weights
without restarting the serving subprocess, while returning the same non-
streaming OpenAI-compatible payload shapes that vLLM serves natively.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
from http import HTTPStatus
import logging
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from vllm import EngineArgs, LLM
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ErrorInfo,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    TokenizeCompletionRequest,
    TokenizeResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_engine import clamp_prompt_logprobs
from vllm.entrypoints.utils import get_max_tokens
from vllm.outputs import RequestOutput
from vllm.tokenizers import TokenizerLike
from vllm.utils.argparse_utils import FlexibleArgumentParser


class _LoadWeightsRequest(BaseModel):
    model_source: str


app: FastAPI | None = None
llm: LLM | None = None
_model_lock = threading.RLock()
_engine_args: EngineArgs | None = None
_model_source: str = ""
_served_model_names: list[str] = []
_default_sampling_params: dict[str, Any] = {}
_inference_paused = False
logger = logging.getLogger(__name__)


def _build_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(prog="python -m flashrl.framework.serving.vllm.server")
    return make_arg_parser(parser)


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "model", None) is None and getattr(args, "model_tag", None):
        args.model = args.model_tag
    if getattr(args, "model", None) is None:
        parser.error("one of --model or model_tag is required")
    validate_parsed_serve_args(args)
    return args


def _normalize_served_model_names(
    served_model_name: str | list[str] | None,
    fallback: str,
) -> list[str]:
    if isinstance(served_model_name, str):
        return [served_model_name]
    if isinstance(served_model_name, list):
        names = [str(name) for name in served_model_name if str(name).strip()]
        if names:
            return names
    return [fallback]


def _engine_args_to_kwargs(engine_args: EngineArgs) -> dict[str, Any]:
    return {
        field.name: getattr(engine_args, field.name)
        for field in dataclasses.fields(engine_args)
    }


def _dispose_llm(model: LLM | None) -> None:
    if model is None:
        return
    llm_engine = getattr(model, "llm_engine", None)
    shutdown = getattr(llm_engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del model
    gc.collect()


def _error_response(
    message: str,
    status_code: int,
    *,
    param: str | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorInfo(
            message=str(message),
            type=HTTPStatus(status_code).phrase,
            param=param,
            code=status_code,
        )
    )
    return JSONResponse(content=payload.model_dump(), status_code=status_code)


def _validate_requested_model(model_name: str | None) -> JSONResponse | None:
    if model_name is None or model_name in _served_model_names:
        return None
    return _error_response(
        f"The model `{model_name}` does not match the served model names: {_served_model_names}.",
        HTTPStatus.NOT_FOUND,
        param="model",
    )


def _response_model_name(request_model_name: str | None = None) -> str:
    if request_model_name is not None and request_model_name in _served_model_names:
        return request_model_name
    if _served_model_names:
        return _served_model_names[0]
    return _model_source


def _current_llm() -> LLM:
    if llm is None:
        raise RuntimeError("Engine not initialized")
    return llm


def _encode_prompt(
    tokenizer: TokenizerLike,
    prompt: str,
    *,
    add_special_tokens: bool,
) -> list[int]:
    try:
        return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(prompt))


def _prepare_completion_inputs(
    request: CompletionRequest,
    tokenizer: TokenizerLike,
) -> tuple[list[Any], list[int]]:
    prompt = request.prompt
    if isinstance(prompt, str):
        return [prompt], [
            len(
                _encode_prompt(
                    tokenizer,
                    prompt,
                    add_special_tokens=request.add_special_tokens,
                )
            )
        ]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("`prompt` must be a non-empty string or token sequence.")
    if all(isinstance(item, str) for item in prompt):
        prompts = list(prompt)
        lengths = [
            len(
                _encode_prompt(
                    tokenizer,
                    item,
                    add_special_tokens=request.add_special_tokens,
                )
            )
            for item in prompts
        ]
        return prompts, lengths
    if all(isinstance(item, int) for item in prompt):
        token_ids = [int(item) for item in prompt]
        return [{"prompt_token_ids": token_ids}], [len(token_ids)]
    if all(isinstance(item, list) for item in prompt):
        prompts = []
        lengths = []
        for raw_item in prompt:
            token_ids = [int(item) for item in raw_item]
            prompts.append({"prompt_token_ids": token_ids})
            lengths.append(len(token_ids))
        return prompts, lengths
    raise ValueError("Unsupported `prompt` shape for completions.")


def _decode_token(
    tokenizer: TokenizerLike | None,
    token_id: int,
    *,
    return_as_token_id: bool,
    decoded_token: str | None = None,
) -> str:
    if return_as_token_id:
        return f"token_id:{token_id}"
    if decoded_token is not None:
        return decoded_token
    if tokenizer is None:
        raise ValueError("Tokenizer is required to decode completion logprobs.")
    return str(tokenizer.decode(token_id))


def _create_completion_logprobs(
    token_ids: list[int],
    top_logprobs: Any,
    *,
    num_output_top_logprobs: int,
    tokenizer: TokenizerLike | None,
    initial_text_offset: int = 0,
    return_as_token_id: bool | None = None,
) -> CompletionLogProbs:
    out_text_offset: list[int] = []
    out_token_logprobs: list[float | None] = []
    out_tokens: list[str] = []
    out_top_logprobs: list[dict[str, float] | None] = []
    last_token_len = 0
    should_return_as_token_id = bool(return_as_token_id)
    logprob_steps = list(top_logprobs or [])

    for index, token_id in enumerate(token_ids):
        step_top_logprobs = logprob_steps[index] if index < len(logprob_steps) else None
        decoded_top_logprobs = None
        if isinstance(step_top_logprobs, dict):
            decoded_top_logprobs = {
                _decode_token(
                    tokenizer,
                    candidate_token_id,
                    return_as_token_id=should_return_as_token_id,
                    decoded_token=candidate_logprob.decoded_token,
                ): max(float(candidate_logprob.logprob), -9999.0)
                for rank, (candidate_token_id, candidate_logprob) in enumerate(
                    step_top_logprobs.items()
                )
                if num_output_top_logprobs >= rank
            }
        sampled_token = None
        if isinstance(step_top_logprobs, dict):
            sampled_token = step_top_logprobs.get(token_id)

        if sampled_token is None:
            token = _decode_token(
                tokenizer,
                token_id,
                return_as_token_id=should_return_as_token_id,
            )
            out_tokens.append(token)
            out_token_logprobs.append(None)
            out_top_logprobs.append(decoded_top_logprobs or None)
        else:
            token = _decode_token(
                tokenizer,
                token_id,
                return_as_token_id=should_return_as_token_id,
                decoded_token=sampled_token.decoded_token,
            )
            out_tokens.append(token)
            out_token_logprobs.append(max(float(sampled_token.logprob), -9999.0))
            out_top_logprobs.append(decoded_top_logprobs or None)

        if not out_text_offset:
            out_text_offset.append(initial_text_offset)
        else:
            out_text_offset.append(out_text_offset[-1] + last_token_len)
        last_token_len = len(token)

    return CompletionLogProbs(
        text_offset=out_text_offset,
        token_logprobs=out_token_logprobs,
        tokens=out_tokens,
        top_logprobs=out_top_logprobs,
    )


def _request_output_to_completion_response(
    final_res_batch: list[RequestOutput],
    request: CompletionRequest,
    *,
    model_name: str,
    tokenizer: TokenizerLike | None,
) -> CompletionResponse:
    choices: list[CompletionResponseChoice] = []
    prompt_tokens = 0
    completion_tokens = 0

    for final_res in final_res_batch:
        prompt_token_ids = list(final_res.prompt_token_ids or [])
        prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
        prompt_text = final_res.prompt or ""

        for output in final_res.outputs:
            token_ids = list(output.token_ids)
            out_logprobs = output.logprobs
            output_text = output.text

            if request.echo:
                if request.return_token_ids:
                    prompt_text = ""
                if request.max_tokens == 0:
                    token_ids = list(prompt_token_ids)
                    out_logprobs = prompt_logprobs
                    output_text = prompt_text
                else:
                    token_ids = [*prompt_token_ids, *token_ids]
                    if request.logprobs is not None:
                        out_logprobs = [*(prompt_logprobs or []), *(output.logprobs or [])]
                    output_text = prompt_text + output.text

            logprobs = None
            if request.logprobs is not None:
                logprobs = _create_completion_logprobs(
                    token_ids,
                    out_logprobs or [],
                    num_output_top_logprobs=request.logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )

            choices.append(
                CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=prompt_logprobs,
                    prompt_token_ids=prompt_token_ids if request.return_token_ids else None,
                    token_ids=list(output.token_ids) if request.return_token_ids else None,
                )
            )
            completion_tokens += len(output.token_ids)

        prompt_tokens += len(prompt_token_ids)

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return CompletionResponse(
        id=f"cmpl-{request.request_id}",
        created=int(time.time()),
        model=model_name,
        choices=choices,
        usage=usage,
    )


def load_model(model_source: str, engine_args: EngineArgs | None = None) -> None:
    """Load or reload the current model in-process."""
    global llm, _model_source, _engine_args, _default_sampling_params

    resolved_engine_args = engine_args or _engine_args
    if resolved_engine_args is None:
        raise RuntimeError("Engine arguments have not been initialized.")

    with _model_lock:
        kwargs = _engine_args_to_kwargs(resolved_engine_args)
        kwargs["model"] = model_source

        previous = llm
        llm = None
        try:
            llm = LLM(**kwargs)
            _model_source = model_source
            _engine_args = resolved_engine_args
            _default_sampling_params = llm.model_config.get_diff_sampling_param() or {}
        finally:
            _dispose_llm(previous)


def _build_app() -> FastAPI:
    application = FastAPI(title="FlashRL vLLM Wrapper")

    @application.on_event("shutdown")
    def _shutdown() -> None:
        global llm
        with _model_lock:
            _dispose_llm(llm)
            llm = None

    @application.get("/health")
    def health_check() -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy",
                "inference_paused": _inference_paused,
            }
        )

    @application.get("/v1/models")
    def list_models() -> JSONResponse:
        with _model_lock:
            current = _current_llm()
            payload = ModelList(
                data=[
                    ModelCard(
                        id=model_name,
                        root=_model_source,
                        max_model_len=current.model_config.max_model_len,
                        permission=[ModelPermission()],
                    )
                    for model_name in _served_model_names
                ]
            )
        return JSONResponse(content=payload.model_dump())

    @application.post("/v1/completions")
    def create_completion(request: CompletionRequest) -> JSONResponse:
        if request.stream:
            return _error_response(
                "stream=True is not supported by the FlashRL vLLM wrapper.",
                HTTPStatus.NOT_IMPLEMENTED,
                param="stream",
            )
        if request.use_beam_search:
            return _error_response(
                "use_beam_search is not supported by the FlashRL vLLM wrapper.",
                HTTPStatus.NOT_IMPLEMENTED,
                param="use_beam_search",
            )
        if request.prompt_embeds is not None:
            return _error_response(
                "prompt_embeds is not supported by the FlashRL vLLM wrapper.",
                HTTPStatus.NOT_IMPLEMENTED,
                param="prompt_embeds",
            )
        if _inference_paused:
            return _error_response("Inference is paused.", HTTPStatus.SERVICE_UNAVAILABLE)
        model_error = _validate_requested_model(request.model)
        if model_error is not None:
            return model_error

        try:
            with _model_lock:
                current = _current_llm()
                tokenizer = current.get_tokenizer()
                prompts, input_lengths = _prepare_completion_inputs(request, tokenizer)
                sampling_params = [
                    request.to_sampling_params(
                        get_max_tokens(
                            current.model_config.max_model_len,
                            request,
                            input_length,
                            _default_sampling_params,
                        ),
                        current.model_config.logits_processor_pattern,
                        _default_sampling_params,
                    )
                    for input_length in input_lengths
                ]
                outputs = current.generate(
                    prompts[0] if len(prompts) == 1 else prompts,
                    sampling_params[0] if len(sampling_params) == 1 else sampling_params,
                    use_tqdm=False,
                )
                payload = _request_output_to_completion_response(
                    outputs,
                    request,
                    model_name=_response_model_name(request.model),
                    tokenizer=tokenizer,
                )
        except ValueError as exc:
            return _error_response(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception("vLLM completion request failed.")
            return _error_response(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

        return JSONResponse(content=payload.model_dump())

    @application.post("/tokenize")
    def tokenize_endpoint(request: TokenizeCompletionRequest) -> JSONResponse:
        model_error = _validate_requested_model(request.model)
        if model_error is not None:
            return model_error

        try:
            with _model_lock:
                current = _current_llm()
                tokenizer = current.get_tokenizer()
                tokens = _encode_prompt(
                    tokenizer,
                    request.prompt,
                    add_special_tokens=request.add_special_tokens,
                )
                token_strs = (
                    list(tokenizer.convert_ids_to_tokens(tokens))
                    if request.return_token_strs
                    else None
                )
                payload = TokenizeResponse(
                    tokens=tokens,
                    token_strs=token_strs,
                    count=len(tokens),
                    max_model_len=current.model_config.max_model_len,
                )
        except Exception as exc:
            logger.exception("vLLM tokenize request failed.")
            return _error_response(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

        return JSONResponse(content=payload.model_dump())

    @application.post("/v1/load_weights_from_disk")
    def load_weights_from_disk(request: _LoadWeightsRequest) -> JSONResponse:
        try:
            load_model(request.model_source)
        except Exception as exc:
            logger.exception("vLLM weight reload failed.")
            return _error_response(
                f"Failed to load weights: {exc}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return JSONResponse(
            {
                "status": "success",
                "model_source": request.model_source,
            }
        )

    @application.post("/admin/pause")
    def pause_inference() -> JSONResponse:
        global _inference_paused
        _inference_paused = True
        return JSONResponse({"status": "inference_paused"})

    @application.post("/admin/resume")
    def resume_inference() -> JSONResponse:
        global _inference_paused
        _inference_paused = False
        return JSONResponse({"status": "inference_resumed"})

    return application


def create_app(
    model_source: str,
    *,
    served_model_name: str | list[str] | None = None,
    vllm_args: list[str] | None = None,
) -> FastAPI:
    """Create a configured app for tests or embedding."""
    cli_args = ["--model", model_source]
    if served_model_name is not None:
        if isinstance(served_model_name, list):
            cli_args.extend(["--served-model-name", *served_model_name])
        else:
            cli_args.extend(["--served-model-name", served_model_name])
    cli_args.extend(vllm_args or [])
    args = _parse_cli_args(cli_args)
    return create_app_from_args(args)


def create_app_from_args(args: argparse.Namespace) -> FastAPI:
    """Initialize the wrapper state and build the FastAPI app."""
    global app, _engine_args, _served_model_names, _inference_paused

    engine_args = EngineArgs.from_cli_args(args)
    _engine_args = engine_args
    _served_model_names = _normalize_served_model_names(
        engine_args.served_model_name,
        str(engine_args.model),
    )
    _inference_paused = False
    load_model(str(engine_args.model), engine_args)
    app = _build_app()
    return app


def create_server(
    model_source: str,
    host: str,
    port: int,
    *,
    served_model_name: str | list[str] | None = None,
    vllm_args: list[str] | None = None,
) -> None:
    """Create and run the FastAPI wrapper server."""
    cli_args = [
        "--model",
        model_source,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if served_model_name is not None:
        if isinstance(served_model_name, list):
            cli_args.extend(["--served-model-name", *served_model_name])
        else:
            cli_args.extend(["--served-model-name", served_model_name])
    cli_args.extend(vllm_args or [])
    args = _parse_cli_args(cli_args)
    application = create_app_from_args(args)
    uvicorn.run(
        application,
        host=args.host or host,
        port=int(args.port or port),
        log_level=str(args.uvicorn_log_level),
        access_log=not bool(args.disable_uvicorn_access_log),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_cli_args(argv)
    application = create_app_from_args(args)
    uvicorn.run(
        application,
        host=args.host or "127.0.0.1",
        port=int(args.port),
        log_level=str(args.uvicorn_log_level),
        access_log=not bool(args.disable_uvicorn_access_log),
    )


if __name__ == "__main__":
    main()
