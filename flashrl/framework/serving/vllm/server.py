"""Custom vLLM HTTP server for FlashRL.

This server wraps vLLM's LLMEngine directly (in-process), enabling
weight swaps without process restart.
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any

try:
    from flask import Flask, request as flask_request, jsonify

    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

try:
    from vllm import LLMEngine, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


app: Flask | None = None
engine: LLMEngine | None = None
_model_lock: threading.Lock = threading.Lock()
_inference_paused: bool = False
_loading_thread: threading.Thread | None = None
_loading_complete: threading.Event = threading.Event()
_loading_success: bool = True
_loading_error: str | None = None


def create_server(model_source: str, host: str, port: int) -> None:
    """Create and run vLLM wrapper server.

    Args:
        model_source: Model path or HuggingFace model name
        host: Host to bind to
        port: Port to bind to

    Raises:
        ImportError: If Flask or vLLM are not installed
    """
    global app, engine

    if not _FLASK_AVAILABLE:
        raise ImportError(
            "Flask is required for custom vLLM server. "
            "Install it with: pip install flask"
        )

    if not _VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is required for custom vLLM server. "
            "Install it with: pip install vllm"
        )

    # Initialize Flask app
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    # Initialize LLM engine
    load_model(model_source)

    # Register routes
    register_routes()

    # Run server
    app.run(host=host, port=port, threaded=True)


def load_model(model_source: str) -> None:
    """Load or reload model into vLLM engine."""
    global engine

    # Clean up existing engine if present
    if engine is not None:
        del engine
        engine = None

    # Create new engine instance
    from vllm import EngineArgs

    engine_args = EngineArgs(
        model=model_source,
        disable_log_stats=True,
        served_model_name=model_source,
    )

    engine = LLMEngine.from_engine_args(engine_args)


def register_routes() -> None:
    """Register all HTTP routes."""
    global app

    # Health check
    @app.route("/health", methods=["GET"])
    def health_check():
        global _loading_thread, _loading_complete, _loading_success
        loading_status = "idle"
        if _loading_thread is not None and _loading_thread.is_alive():
            loading_status = "loading"
        elif _loading_complete.is_set():
            loading_status = "success" if _loading_success else "error"

        return jsonify({
            "status": "healthy",
            "inference_paused": _inference_paused,
            "model_loading": loading_status,
        })

    # OpenAI-compatible endpoints
    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """List available models (OpenAI compatible)."""
        global _loading_thread, _loading_complete, _loading_success
        loading_status = "idle"
        if _loading_thread is not None and _loading_thread.is_alive():
            loading_status = "loading"
        elif _loading_complete.is_set():
            loading_status = "success" if _loading_success else "error"

        return jsonify({
            "object": "list",
            "data": [{
                "id": engine.engine_args.model if engine else "",
                "loading_status": loading_status,
                "loading_error": _loading_error if not _loading_success and loading_status == "error" else None,
            }],
        })

    @app.route("/v1/completions", methods=["POST"])
    def create_completion():
        """Generate completions (OpenAI compatible)."""
        if _inference_paused:
            return jsonify({"error": {"message": "Inference is paused"}}), 503

        # Check if model is loading
        global _loading_thread, _loading_complete
        if _loading_thread is not None and _loading_thread.is_alive():
            return jsonify({"error": {"message": "Model is currently loading"}}), 503

        # Check if previous load failed
        if _loading_complete.is_set() and not _loading_success:
            return jsonify({"error": {"message": f"Model loading failed: {_loading_error}"}}), 500

        data = flask_request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": {"message": "Invalid request body"}}), 400

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            return jsonify({"error": {"message": "prompt must be a string"}}), 400

        # Build sampling parameters
        sampling_params = SamplingParams(
            temperature=float(data.get("temperature", 1.0)),
            top_p=float(data.get("top_p", 0.9)),
            top_k=int(data.get("top_k", 0)),
            max_tokens=int(data.get("max_tokens", 512)),
        )

        # Generate completion
        with _model_lock:
            if engine is None:
                return jsonify({"error": {"message": "Engine not initialized"}}), 500

            # Tokenize prompt
            tokenizer = engine.get_tokenizer()
            prompt_token_ids = tokenizer.encode(prompt)

            # Generate
            from vllm import SamplingType

            results = engine.generate(
                prompt_token_ids,
                sampling_params=sampling_params,
            )

        # Format response
        choices = []
        for i, result in enumerate(results):
            output_token_ids = result.outputs[0].token_ids
            output_text = tokenizer.decode(output_token_ids)

            # Calculate logprobs
            logprobs_obj = result.outputs[0].logprobs
            if hasattr(logprobs_obj, "__len__") and len(logprobs_obj) > 0:
                token_logprobs = [float(lp) for lp in logprobs_obj]
            else:
                token_logprobs = []

            cumulative_logprob = sum(token_logprobs) if token_logprobs else None

            choices.append(
                {
                    "text": output_text,
                    "index": i,
                    "finish_reason": result.outputs[0].finish_reason,
                    "token_ids": output_token_ids,
                    "prompt_token_ids": prompt_token_ids,
                    "logprobs": {"token_logprobs": token_logprobs},
                    "cumulative_logprob": cumulative_logprob,
                }
            )

        return jsonify({"choices": choices})

    @app.route("/tokenize", methods=["POST"])
    def tokenize_endpoint():
        """Tokenize text."""
        data = flask_request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": {"message": "Invalid request body"}}), 400

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            return jsonify({"error": {"message": "prompt must be a string"}}), 400

        with _model_lock:
            if engine is None:
                return jsonify({"error": {"message": "Engine not initialized"}}), 500

            tokenizer = engine.get_tokenizer()
            tokens = tokenizer.encode(prompt)

        return jsonify({"tokens": tokens})

    # Custom endpoints for FlashRL
    @app.route("/v1/load_weights_from_disk", methods=["POST"])
    def load_weights_from_disk():
        """Load model weights from disk without restarting process.

        Returns immediately and loads model in background thread.
        """
        data = flask_request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": {"message": "Invalid request body"}}), 400

        model_source = data.get("model_source")
        if not isinstance(model_source, str):
            return jsonify({"error": {"message": "model_source must be a string"}}), 400

        # Check if already loading
        global _loading_thread, _loading_complete
        if _loading_thread is not None and _loading_thread.is_alive():
            return jsonify(
                {"status": "loading_in_progress", "message": "Model loading is already in progress"}
            ), 202

        # Start loading in background thread
        _loading_thread = threading.Thread(
            target=_load_model_async,
            args=(model_source,),
            daemon=True,
        )
        _loading_thread.start()

        return jsonify({
            "status": "loading_started",
            "message": "Model loading started in background",
            "model_source": model_source,
        })

    @app.route("/v1/load_status", methods=["GET"])
    def load_status():
        """Check model loading status."""
        global _loading_thread, _loading_complete, _loading_success, _loading_error

        if _loading_thread is None:
            return jsonify({"status": "idle"})

        if _loading_thread.is_alive():
            return jsonify({"status": "loading", "message": "Model loading in progress"})

        if _loading_complete.is_set():
            if _loading_success:
                return jsonify({"status": "success", "message": "Model loaded successfully"})
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Model loading failed: {_loading_error}"
                }), 500

        return jsonify({"status": "unknown"})


def _load_model_async(model_source: str) -> None:
    """Load model asynchronously in background thread."""
    global engine, _loading_complete, _loading_success, _loading_error

    try:
        # Load new model with weights
        load_model(model_source)
        _loading_success = True
    except Exception as exc:
        _loading_success = False
        _loading_error = str(exc)
    finally:
        _loading_complete.set()

    # Admin endpoints
    @app.route("/admin/pause", methods=["POST"])
    def pause_inference():
        """Pause inference."""
        global _inference_paused
        _inference_paused = True
        return jsonify({"status": "inference_paused"})

    @app.route("/admin/resume", methods=["POST"])
    def resume_inference():
        """Resume inference."""
        global _inference_paused
        _inference_paused = False
        return jsonify({"status": "inference_resumed"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashRL custom vLLM server")
    parser.add_argument("--model", type=str, required=True, help="Model source (path or HuggingFace name)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    args = parser.parse_args()

    create_server(args.model, args.host, args.port)
