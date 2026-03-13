"""Background FastAPI server for the FlashRL admin API."""

from __future__ import annotations

import socket
from threading import Thread
import time

import uvicorn

from flashrl.framework.admin.app import create_admin_app
from flashrl.framework.admin.registry import AdminRegistry


_STARTUP_TIMEOUT_SECONDS = 5.0


class AdminServer:
    """Background admin API server bound to one registry."""

    def __init__(self, registry: AdminRegistry, host: str, port: int) -> None:
        self._registry = registry
        self._host = host
        self._port = port
        self._thread: Thread | None = None
        self._server: uvicorn.Server | None = None
        self._socket: socket.socket | None = None
        self._base_url: str | None = None
        self._startup_error: BaseException | None = None

    @property
    def base_url(self) -> str | None:
        """Return the bound base URL when the server is running."""
        return self._base_url

    def start(self) -> str:
        """Start serving in a background thread and return the bound base URL."""
        if self._server is not None:
            base_url = self.base_url
            assert base_url is not None
            return base_url

        app = create_admin_app(self._registry)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            access_log=False,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        sock = self._bind_socket()
        self._startup_error = None
        self._socket = sock
        self._server = server
        self._base_url = self._format_base_url(self._host, int(sock.getsockname()[1]))
        self._thread = Thread(
            target=self._run_server,
            args=(server, sock),
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                base_url = self.base_url
                assert base_url is not None
                return base_url
            if self._startup_error is not None:
                self.close()
                raise RuntimeError(
                    f"FlashRL admin server failed to start: {self._startup_error}"
                ) from self._startup_error
            if self._thread is not None and not self._thread.is_alive():
                self.close()
                raise RuntimeError("FlashRL admin server stopped before becoming ready.")
            time.sleep(0.01)

        self.close()
        raise RuntimeError("FlashRL admin server timed out during startup.")

    def close(self) -> None:
        """Stop the background server."""
        server = self._server
        thread = self._thread
        sock = self._socket

        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive() and server is not None:
            server.force_exit = True
            thread.join(timeout=2.0)
        if sock is not None:
            sock.close()

        self._server = None
        self._thread = None
        self._socket = None
        self._base_url = None
        self._startup_error = None

    def _run_server(self, server: uvicorn.Server, sock: socket.socket) -> None:
        try:
            server.run(sockets=[sock])
        except BaseException as exc:  # pragma: no cover - guarded by startup wait
            self._startup_error = exc

    def _bind_socket(self) -> socket.socket:
        family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        sock = socket.socket(family=family)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.set_inheritable(True)
        return sock

    def _format_base_url(self, host: str, port: int) -> str:
        if ":" in host and not host.startswith("["):
            return f"http://[{host}]:{port}"
        return f"http://{host}:{port}"
