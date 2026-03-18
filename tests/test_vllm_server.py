"""Unit tests for vLLM custom server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

try:
    from flask import Flask, jsonify
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

import pytest


class TestVLLMServer:
    """Test suite for vLLM custom HTTP server."""

    def test_health_check(self, monkeypatch: pytest.MonkeyPatch):
        """Health check should return healthy status."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server.engine", MagicMock()):
            # Mock engine
            mock_engine = MagicMock()
            mock_engine.engine_core = MagicMock(model="test-model")

            with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                from flashrl.framework.serving.vllm.server import list_models
                response = list_models()

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["inference_paused"] == False
        assert "model_loading" not in data  # No loading initially

    def test_list_models(self, monkeypatch: pytest.MonkeyPatch):
        """Models endpoint should list available models."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server.engine", MagicMock()):
            # Mock engine
            mock_engine = MagicMock()
            mock_engine.engine_core = MagicMock(model="test-model")

            with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                from flashrl.framework.serving.vllm.server import list_models
                response = list_models()

        assert response.status_code == 200
        data = response.get_json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert "loading_status" not in data["data"][0]  # No loading initially

    def test_list_models_with_loading_status(self, monkeypatch: pytest.MonkeyPatch):
        """Models endpoint should include loading status when loading."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server.engine", MagicMock()):
            with patch("flashrl.framework.serving.vllm.server._loading_thread", MagicMock(is_alive=lambda: True)):
                with patch("flashrl.framework.serving.vllm.server._loading_complete", MagicMock(is_set=lambda: False)):
                    with patch("flashrl.framework.serving.vllm.server._loading_success", True):
                        with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                            from flashrl.framework.serving.vllm.server import list_models
                            response = list_models()

        assert response.status_code == 200
        data = response.get_json()
        assert data["object"] == "list"
        assert data["data"][0]["loading_status"] == "loading"
        assert data["data"][0]["loading_error"] is None

    def test_list_models_with_error_status(self, monkeypatch: pytest.MonkeyPatch):
        """Models endpoint should include error status when loading fails."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server.engine", MagicMock()):
            with patch("flashrl.framework.serving.vllm.server._loading_thread", MagicMock(is_alive=lambda: True)):
                with patch("flashrl.framework.serving.vllm.server._loading_complete", MagicMock(is_set=lambda: False)):
                    with patch("flashrl.framework.serving.vllm.server._loading_success", False):
                        with patch("flashrl.framework.serving.vllm.server._loading_error", "Load failed"):
                            with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                                from flashrl.framework.serving.vllm.server import list_models
                                response = list_models()

        assert response.status_code == 200
        data = response.get_json()
        assert data["object"] == "list"
        assert data["data"][0]["loading_status"] == "error"
        assert data["data"][0]["loading_error"] == "Load failed"

    def test_load_weights_from_disk(self, monkeypatch: pytest.MonkeyPatch):
        """Load weights endpoint should start loading model."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        mock_engine = MagicMock()
        mock_engine.engine_core = MagicMock(model="old-model")

        with patch("flashrl.framework.serving.vllm.server.engine", mock_engine):
            with patch("flashrl.framework.serving.vllm.server.load_model", MagicMock()):
                with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                    from flashrl.framework.serving.vllm.server import create_server

                    # Start server
                    create_server("test-model", "127.0.0.1", 8000)

                    # Load weights from disk
                    response = self.client.post("/v1/load_weights_from_disk",
                        json={"model_source": "/path/to/weights"},
                        headers={"Content-Type": "application/json"},
                    )

                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["status"] == "loading_started"
                    assert data["model_source"] == "/path/to/weights"

    def test_load_weights_from_disk_while_loading(self, monkeypatch: pytest.MonkeyPatch):
        """Load weights should reject request if already loading."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        mock_engine = MagicMock()

        with patch("flashrl.framework.serving.vllm.server.engine", mock_engine):
            with patch("flashrl.framework.serving.vllm.server.load_model", MagicMock()):
                with patch("flashrl.framework.serving.vllm.server._loading_thread", MagicMock(is_alive=lambda: True)):
                    with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                        from flashrl.framework.serving.vllm.server import create_server

                        # Start server
                        create_server("test-model", "127.0.0.1", 8000)

                        # Try to load again (should be rejected)
                        response = self.client.post("/v1/load_weights_from_disk",
                            json={"model_source": "/path/to/weights2"},
                            headers={"Content-Type": "application/json"},
                        )

                    assert response.status_code == 202
                    data = response.get_json()
                    assert data["status"] == "loading_in_progress"
                    assert "already in progress" in data["message"].lower()

    def test_load_status_idle(self, monkeypatch: pytest.MonkeyPatch):
        """Load status should return idle when not loading."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server._loading_thread", None):
            with patch("flashrl.framework.serving.vllm.server._loading_complete", MagicMock(is_set=lambda: False)):
                with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                    from flashrl.framework.serving.vllm.server import load_status
                    response = load_status()

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "idle"

    def test_load_status_loading(self, monkeypatch: pytest.MonkeyPatch):
        """Load status should return loading when in progress."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        mock_thread = MagicMock(is_alive=lambda: True)
        with patch("flashrl.framework.serving.vllm.server._loading_thread", mock_thread):
            with patch("flashrl.framework.serving.vllm.server._loading_complete", MagicMock(is_set=lambda: False)):
                with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                    from flashrl.framework.serving.vllm.server import load_status
                    response = load_status()

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "loading"
        assert data["message"] == "Model loading in progress"

    def test_load_status_success(self, monkeypatch: pytest.MonkeyPatch):
        """Load status should return success after loading completes."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        mock_thread = MagicMock(is_alive=lambda: False)
        with patch("flashrl.framework.serving.vllm.server._loading_thread", mock_thread):
            with patch("flashrl.framework.serving.vllm.server._loading_complete", MagicMock(is_set=lambda: True)):
                with patch("flashrl.framework.serving.vllm.server._loading_success", True):
                    with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                        from flashrl.framework.serving.vllm.server import load_status
                        response = load_status()

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["message"] == "Model loaded successfully"

    def test_load_status_error(self, monkeypatch: pytest.MonkeyPatch):
        """Load status should return error after loading fails."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        mock_thread = MagicMock(is_alive=lambda: False)
        with patch("flashrl.framework.serving.vllm.server._loading_thread", mock_thread):
            with patch("flashrl.framework.serving.vllm.server._loading_complete", MagicMock(is_set=lambda: True)):
                with patch("flashrl.framework.serving.vllm.server._loading_success", False):
                    with patch("flashrl.framework.serving.vllm.server._loading_error", "Load failed"):
                        with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                            from flashrl.framework.serving.vllm.server import load_status
                            response = load_status()

        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "error"
        assert "Load failed" in data["message"]

    def test_pause_inference(self, monkeypatch: pytest.MonkeyPatch):
        """Pause inference should set paused flag."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server._inference_paused", False):
            with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                from flashrl.framework.serving.vllm.server import pause_inference
                response = pause_inference()

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "inference_paused"

        # Verify flag is set
        from flashrl.framework.serving.vllm.server import _inference_paused
        assert _inference_paused == True

    def test_resume_inference(self, monkeypatch: pytest.MonkeyPatch):
        """Resume inference should clear paused flag."""
        if not _FLASK_AVAILABLE:
            pytest.skip("Flask not available")

        with patch("flashrl.framework.serving.vllm.server._inference_paused", True):
            with patch("flashrl.framework.serving.vllm.server.app", MagicMock()):
                from flashrl.framework.serving.vllm.server import resume_inference
                response = resume_inference()

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "inference_resumed"

        # Verify flag is cleared
        from flashrl.framework.serving.vllm.server import _inference_paused
        assert _inference_paused == False
