from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from events import EventBus

HTML = Path(__file__).with_name("overlay.html")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def default_port() -> int:
    return int(os.environ.get("OVERLAY_PORT") or "8765")


def _ws_accept(key: str) -> str:
    raw = hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(raw).decode("ascii")


def _ws_encode(text: str) -> bytes:
    data = text.encode("utf-8")
    n = len(data)
    if n < 126:
        return bytes([0x81, n]) + data
    if n < 65536:
        return bytes([0x81, 126]) + struct.pack("!H", n) + data
    return bytes([0x81, 127]) + struct.pack("!Q", n) + data


def _ws_read(sock: socket.socket) -> bytes | None:
    hdr = _recv_exact(sock, 2)
    if hdr is None:
        return None
    opcode = hdr[0] & 0x0F
    masked = hdr[1] & 0x80
    n = hdr[1] & 0x7F
    if n == 126:
        ext = _recv_exact(sock, 2)
        if ext is None:
            return None
        n = struct.unpack("!H", ext)[0]
    elif n == 127:
        ext = _recv_exact(sock, 8)
        if ext is None:
            return None
        n = struct.unpack("!Q", ext)[0]
    mask = b""
    if masked:
        mask = _recv_exact(sock, 4) or b""
        if len(mask) < 4:
            return None
    data = _recv_exact(sock, n) if n else b""
    if data is None:
        return None
    if masked:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    if opcode == 0x8:
        return None
    if opcode == 0x9:
        sock.sendall(bytes([0x8A, len(data)]) + data)
        return b""
    return data


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except TimeoutError:
            raise
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


class OverlayServer:
    def __init__(
        self,
        bus: EventBus,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        runtime=None,
    ) -> None:
        self.bus = bus
        self.runtime = runtime
        self.host = host
        self.port = port if port is not None else default_port()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        bus = self.bus
        runtime = self.runtime

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args) -> None:
                return

            def _json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                if not raw:
                    return {}
                data = json.loads(raw.decode("utf-8"))
                return data if isinstance(data, dict) else {}

            def do_GET(self) -> None:
                if self.path in {"/", "/overlay", "/overlay.html"}:
                    body = HTML.read_bytes() if HTML.is_file() else b"<p>overlay.html missing</p>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.rstrip("/") == "/api/config":
                    if runtime is None:
                        self._json({"error": "runtime unavailable"}, 503)
                        return
                    self._json(runtime.snapshot())
                    return
                if self.path.rstrip("/") == "/ws":
                    self._ws()
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                path = self.path.rstrip("/")
                if runtime is None:
                    self._json({"error": "runtime unavailable"}, 503)
                    return
                try:
                    body = self._read_json()
                except json.JSONDecodeError:
                    self._json({"error": "invalid json"}, 400)
                    return
                if path == "/api/config":
                    try:
                        kwargs = {}
                        if "llm_model" in body:
                            kwargs["llm_model"] = body.get("llm_model")
                        if body.get("whisper_model"):
                            kwargs["whisper_model"] = body.get("whisper_model")
                        if body.get("theme"):
                            kwargs["theme"] = body.get("theme")
                        if "prompt" in body:
                            kwargs["prompt"] = body.get("prompt") or ""
                        snap = runtime.apply(**kwargs)
                    except ValueError as exc:
                        self._json({"error": str(exc)}, 400)
                        return
                    self._json(snap)
                    return
                if path == "/api/summary":
                    try:
                        snap = runtime.start_summary()
                    except ValueError as exc:
                        self._json({"error": str(exc)}, 400)
                        return
                    self._json(snap)
                    return
                if path == "/api/qa/delete":
                    q = str(body.get("question") or "").strip()
                    if q:
                        runtime.forget_question(q)
                    else:
                        runtime.clear_questions()
                    self._json({"ok": True})
                    return
                self.send_error(404)

            def _ws(self) -> None:
                key = self.headers.get("Sec-WebSocket-Key")
                if not key:
                    self.send_error(400)
                    return
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", _ws_accept(key))
                self.end_headers()
                self.wfile.flush()
                sock = self.connection
                sock.settimeout(0.25)
                q = bus.subscribe()
                try:
                    while True:
                        try:
                            while True:
                                ev = q.get_nowait()
                                sock.sendall(_ws_encode(json.dumps(ev, ensure_ascii=False)))
                        except queue.Empty:
                            pass
                        try:
                            if _ws_read(sock) is None:
                                break
                        except TimeoutError:
                            continue
                        except OSError:
                            break
                finally:
                    bus.unsubscribe(q)

        ThreadingHTTPServer.allow_reuse_address = True
        last_err: OSError | None = None
        wanted = self.port
        for port in range(wanted, wanted + 16):
            try:
                httpd = ThreadingHTTPServer((self.host, port), Handler)
                break
            except OSError as exc:
                last_err = exc
                httpd = None
        else:
            raise last_err or OSError("no free overlay port")
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        if self.port != wanted:
            print(f"overlay port {wanted} busy, using {self.port}", flush=True)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="overlay-http")
        self._thread.start()

    def close(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
