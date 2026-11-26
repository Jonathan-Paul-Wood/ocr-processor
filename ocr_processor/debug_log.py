"""Optional, bounded traffic capture. Only the API process owns the scratch file."""
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MAX_EVENT_BYTES = 1024 * 1024
MAX_EVENTS = 100
# Two snapshots can coexist during atomic replacement in the 64 MiB mount.
MAX_RETAINED_BYTES = 30 * 1024 * 1024
SENSITIVE_KEYS = {
    "authorization", "proxy_authorization", "cookie", "set_cookie", "x_api_key",
    "api_key", "access_token", "refresh_token", "token", "password", "secret",
    "ocr_api_key", "llm_api_key", "systemone_api_key",
}


def _sensitive(key):
    return str(key).lower().replace("-", "_") in SENSITIVE_KEYS


def redact(value, secrets=(), depth=0):
    if depth > 50:
        return "[depth limit]"
    if isinstance(value, dict):
        return {redact(str(key), secrets, depth + 1): "[REDACTED]" if _sensitive(key) else redact(item, secrets, depth + 1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets, depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"omitted": "binary_content", "byte_count": len(value)}
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[unsupported value]"


def safe_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path,
                       urlencode([(key, "[REDACTED]" if _sensitive(key) else value)
                                  for key, value in parse_qsl(parts.query, keep_blank_values=True)]), ""))


def body_for_log(raw, content_type=""):
    """Never turn transmitted binary files into text/base64 log bodies."""
    content_type = content_type.lower().split(";", 1)[0]
    binary = content_type.startswith(("image/", "audio/", "video/", "multipart/")) or content_type in {
        "application/pdf", "application/octet-stream", "application/zip",
    }
    if binary or bytes(raw[:5]).startswith((b"%PDF-", b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"II*\x00", b"MM\x00*")):
        return {"omitted": "binary_content", "byte_count": len(raw), "content_type": content_type}
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        return bytes(raw).decode("utf-8", errors="replace")


def _json_line(event):
    return (json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def encode_event(event, secrets=(), max_bytes=MAX_EVENT_BYTES):
    cleaned = redact(event, secrets)
    line = _json_line(cleaned)
    if len(line) <= max_bytes:
        return line
    summary = {key: cleaned.get(key) for key in ("timestamp", "event", "request_id")}
    summary.update(truncated=True, original_size_bytes=len(line))
    # A JSON string can expand when escaped again. Reserve headroom for that escape.
    budget = max(0, (max_bytes - len(_json_line(summary)) - 128) // 2)
    while True:
        summary["content_preview"] = line[:budget].decode("utf-8", errors="ignore")
        result = _json_line(summary)
        if len(result) <= max_bytes:
            return result
        budget //= 2


def new_event(event, request_id=None, **fields):
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event": event, "request_id": request_id, **fields}


def exception_causes(exc):
    causes, seen = [], set()
    while exc is not None and id(exc) not in seen and len(causes) < 10:
        seen.add(id(exc))
        causes.append({"type": type(exc).__name__, "message": str(exc)})
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return causes


def capture_llm_response(logger, response, raw, started, **fields):
    if logger:
        try:
            logger.emit("llm.response", status=response.status_code, headers=dict(response.headers),
                        body=body_for_log(raw, response.headers.get("content-type", "")),
                        duration_ms=round((time.monotonic() - started) * 1000, 3), **fields)
        except Exception:
            logger.emit("logging.capture_failed", stage="llm_response")


def capture_llm_error(logger, exc, started):
    if logger:
        try:
            logger.emit("llm.error", causes=exception_causes(exc),
                        duration_ms=round((time.monotonic() - started) * 1000, 3))
        except Exception:
            logger.emit("logging.capture_failed", stage="llm_error")


class DebugLog:
    def __init__(self, directory, secrets=(), *, max_events=MAX_EVENTS, max_event_bytes=MAX_EVENT_BYTES,
                 max_retained_bytes=MAX_RETAINED_BYTES):
        self.path = Path(directory) / "traffic.jsonl"
        self.secrets = secrets
        self.max_events = max_events
        self.max_event_bytes = max_event_bytes
        self.max_retained_bytes = max_retained_bytes
        self._queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._dropped = 0
        self._warned = False
        self._thread = None
        self._enabled = False
        self._records = deque()
        self._bytes = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.with_suffix(".tmp").unlink(missing_ok=True)
            self.path.write_bytes(b"")
            self.path.chmod(0o600)
            self._enabled = True
            self._thread = threading.Thread(target=self._write_loop, name="traffic-debug-writer", daemon=True)
            self._thread.start()
        except Exception as exc:
            self._enabled = False
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            self._warning(exc)

    def _warning(self, exc):
        if not self._warned:
            self._warned = True
            try:
                print(f"Debug traffic logging unavailable ({type(exc).__name__}); extraction continues.", file=sys.stderr)
            except Exception:
                pass

    def emit(self, event, request_id=None, **fields):
        if not self._enabled or self._stop.is_set():
            return
        try:
            self.accept(new_event(event, request_id, **fields))
        except Exception as exc:
            self._warning(exc)

    def accept(self, event):
        if not self._enabled or self._stop.is_set():
            return
        try:
            line = encode_event(event, self.secrets, self.max_event_bytes)
            self._queue.put_nowait(line)
        except queue.Full:
            with self._lock:
                self._dropped += 1
        except Exception as exc:
            self._warning(exc)

    def _persist(self, line):
        self._records.append(line)
        self._bytes += len(line)
        while len(self._records) > self.max_events or self._bytes > self.max_retained_bytes:
            self._bytes -= len(self._records.popleft())
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            for record in self._records:
                destination.write(record)
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def _write_loop(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                line = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    dropped, self._dropped = self._dropped, 0
                self._persist(line)
                if dropped:
                    self._persist(encode_event(new_event("logging.dropped_events", count=dropped)))
            except Exception as exc:
                self._enabled = False
                self._warning(exc)
            finally:
                self._queue.task_done()

    def close(self, timeout=2):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0, min(timeout, 2)))


class WorkerDebugLog:
    """Send events to the parent; do not touch or reset the debug directory."""
    def __init__(self, settings, request_id):
        self.secrets = (settings.api_key, settings.llm_api_key, settings.systemone_api_key)
        self.request_id = request_id

    def emit(self, event, **fields):
        try:
            line = encode_event(new_event(event, self.request_id, **fields), self.secrets)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        except Exception:
            pass


def forward_worker_events(stream, logger, request_id):
    """Drain the pipe even if debug storage fails; never enqueue an unbounded line."""
    try:
        with stream:
            while True:
                line = stream.readline(MAX_EVENT_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_EVENT_BYTES:
                    while line and not line.endswith(b"\n"):
                        line = stream.readline(MAX_EVENT_BYTES + 1)
                    if logger:
                        logger.emit("logging.invalid_worker_event", request_id, reason="event_exceeds_limit")
                    continue
                if logger:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError()
                        event["request_id"] = request_id
                        logger.accept(event)
                    except (ValueError, UnicodeError, RecursionError):
                        logger.emit("logging.invalid_worker_event", request_id, reason="invalid_json_event")
    except Exception as exc:
        if logger:
            logger.emit("logging.worker_pipe_error", request_id, exception_type=type(exc).__name__)
