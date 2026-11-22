import hmac
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from urllib.parse import urlsplit

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import Settings
from .errors import ServiceError, unexpected_error
from .schema import load_json


def run_worker(directory, settings):
    options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(
        [sys.executable, "-m", "ocr_processor.worker", str(directory)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "OCR_WORKER_SETTINGS": json.dumps(asdict(settings))},
        **options,
    )
    try:
        process.wait(timeout=settings.timeout + 5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
        process.wait()
        raise ServiceError("timeout", "Document processing exceeded its time limit.", 504,
                           details={"stage": "worker", "timeout_seconds": settings.timeout,
                                    "cleanup_allowance_seconds": 5}) from None
    result = directory / "result.json"
    if process.returncode != 0 or not result.exists():
        details = {"stage": "worker", "exit_code": process.returncode, "result_file_present": result.exists(),
                   "hint": "Check worker dependencies, temporary storage permissions/capacity, and container/host logs."}
        if os.name != "nt" and process.returncode < 0:
            details["signal"] = signal.Signals(-process.returncode).name
            if process.returncode == -signal.SIGKILL:
                details["hint"] = "Worker was killed. Check container OOM events and host logs; an OOM kill is one possible cause."
        raise ServiceError("worker_failed", "Document worker exited unexpectedly.", 500, details=details)
    with result.open(encoding="utf-8") as source:
        try:
            return load_json(source.read())
        except (ValueError, RecursionError) as exc:
            raise ServiceError("worker_failed", "Document worker returned an invalid result.", 500,
                               details={"stage": "worker_result", "exception_type": type(exc).__name__}) from None


def create_app(settings=None, worker_runner=run_worker):
    settings = settings or Settings.from_env()
    if len(settings.api_key) < 32:
        raise RuntimeError("Set OCR_API_KEY to a random secret of at least 32 characters.")
    if settings.timeout < 1:
        raise RuntimeError("REQUEST_TIMEOUT_SECONDS must be positive.")
    if settings.llm_base_url:
        endpoint = urlsplit(settings.llm_base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise RuntimeError("LLM_BASE_URL must be an HTTP(S) URL without credentials, query, or fragment.")
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=settings.max_upload_bytes, MAX_FORM_MEMORY_SIZE=512 * 1024, MAX_FORM_PARTS=12)
    slot = threading.BoundedSemaphore(1)

    @app.before_request
    def authorize():
        g.request_id = str(uuid.uuid4())
        if request.path != "/health":
            expected = "Bearer " + settings.api_key
            if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), expected.encode()):
                raise ServiceError("unauthorized", "A valid bearer API key is required.", 401,
                                   details={"stage": "authentication", "hint": "Send Authorization: Bearer with the configured OCR_API_KEY."})

    @app.after_request
    def headers(response):
        response.headers["X-Request-ID"] = g.request_id
        response.headers["Cache-Control"] = "no-store"
        if response.status_code in {429, 503}:
            response.headers["Retry-After"] = "5"
        if response.status_code == 401:
            response.headers["WWW-Authenticate"] = "Bearer"
        if response.status_code >= 400 and response.is_json:
            body = response.get_json()
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                details = body["error"].get("details") or {}
                details["request_id"] = g.request_id
                body["error"]["details"] = details
                response.set_data(app.json.dumps(body))
        return response

    @app.errorhandler(ServiceError)
    def service_error(exc):
        return jsonify(exc.payload()), exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        details = {"stage": "request", "http_status": exc.code}
        if exc.code == 413:
            details.update(request_bytes=request.content_length, max_upload_bytes=settings.max_upload_bytes)
            if request.content_length is not None and request.content_length > settings.max_upload_bytes:
                details.update(limit="upload_bytes", actual=request.content_length, maximum=settings.max_upload_bytes,
                               hint="Split the uploads or reduce their size; the limit includes multipart overhead.")
            else:
                details.update(limit="multipart_request", max_form_memory_bytes=app.config["MAX_FORM_MEMORY_SIZE"],
                               max_form_parts=app.config["MAX_FORM_PARTS"],
                               hint="The request exceeded a body, form-memory, or form-part limit; check these limits and field sizes.")
        return jsonify(ServiceError(exc.name.lower().replace(" ", "_"), exc.description, exc.code, details=details).payload()), exc.code

    @app.errorhandler(Exception)
    def unexpected_failure(exc):
        error = unexpected_error(exc, "api")
        return jsonify(error.payload()), error.status

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities():
        return {"extraction_configured": bool(settings.llm_base_url and settings.llm_model),
                "ocr_available": shutil.which(settings.tesseract_cmd) is not None,
                "limits": {"upload_bytes": settings.max_upload_bytes, "files": settings.max_files,
                           "pages": settings.max_pages, "pixels_per_page": settings.max_pixels,
                           "text_chars": settings.max_text_chars, "timeout_seconds": settings.timeout},
                "formats": ["pdf", "png", "jpeg", "tiff", "webp"]}

    def handle(mode):
        if mode == "extract" and not (settings.llm_base_url and settings.llm_model):
            missing = [name for name, value in [("LLM_BASE_URL", settings.llm_base_url), ("LLM_MODEL", settings.llm_model)] if not value]
            raise ServiceError("extraction_unavailable", "Configure LLM_BASE_URL and LLM_MODEL for structured extraction.", 503,
                               details={"stage": "configuration", "missing_settings": missing})
        if not slot.acquire(blocking=False):
            raise ServiceError("busy", "Another document is being processed; retry later.", 429,
                               details={"stage": "admission", "maximum_concurrent_requests": 1, "retry_after_seconds": 5})
        try:
            if request.mimetype != "multipart/form-data":
                raise ServiceError("invalid_request", "Use multipart/form-data.", 400,
                                   details={"stage": "request", "expected_content_type": "multipart/form-data"})
            if set(request.files) - {"documents"} or set(request.form) - {"instructions", "schema", "force_ocr"}:
                raise ServiceError("invalid_request", "Unknown request fields.", 400,
                                   details={"stage": "request", "allowed_file_fields": ["documents"],
                                            "allowed_form_fields": ["instructions", "schema", "force_ocr"]})
            uploads = request.files.getlist("documents")
            if not 1 <= len(uploads) <= settings.max_files:
                raise ServiceError("invalid_request", f"Supply 1 to {settings.max_files} documents.", 400,
                                   details={"stage": "request", "limit": "files", "actual": len(uploads), "minimum": 1, "maximum": settings.max_files})
            if any(len(request.form.getlist(key)) != 1 for key in request.form):
                raise ServiceError("invalid_request", "Duplicate form fields are not allowed.", 400,
                                   details={"stage": "request", "reason": "duplicate_form_fields"})
            instructions = request.form.get("instructions", "")
            schema_text = request.form.get("schema", "{}")
            if len(instructions) > 16_000 or len(schema_text) > 32_000:
                limits = [{"field": name, "actual": len(value), "maximum": maximum, "unit": "characters"}
                          for name, value, maximum in [("instructions", instructions, 16_000), ("schema", schema_text, 32_000)] if len(value) > maximum]
                raise ServiceError("invalid_request", "Instructions or schema exceed the limit.", 400,
                                   details={"stage": "request", "limit": "field_characters", "fields": limits})
            if mode == "extract" and (not instructions.strip() or "schema" not in request.form):
                raise ServiceError("invalid_request", "Supply instructions and a JSON Schema.", 400,
                                   details={"stage": "request", "required_fields": ["instructions", "schema"]})
            try:
                schema = load_json(schema_text)
            except (ValueError, RecursionError) as exc:
                details = {"stage": "schema_validation", "exception_type": type(exc).__name__}
                if isinstance(exc, json.JSONDecodeError):
                    details.update(line=exc.lineno, column=exc.colno, reason=exc.msg)
                raise ServiceError("invalid_schema", "Schema must be valid JSON.", 400, details=details) from None
            force = request.form.get("force_ocr", "false")
            if force not in {"true", "false"}:
                raise ServiceError("invalid_request", "force_ocr must be true or false.", 400,
                                   details={"stage": "request", "field": "force_ocr", "allowed_values": ["true", "false"]})
            with tempfile.TemporaryDirectory(prefix="ocr-") as temporary:
                directory = Path(temporary)
                paths = []
                for index, upload in enumerate(uploads):
                    # Caller filenames and MIME types never determine file paths or parsing.
                    path = directory / f"document-{index}"
                    upload.save(path)
                    if path.stat().st_size == 0:
                        raise ServiceError("invalid_document", "Empty documents are not supported.", 400,
                                           details={"stage": "upload", "document": index + 1, "bytes": 0})
                    paths.append(str(path))
                job = {"mode": mode, "files": paths, "instructions": instructions, "schema": schema, "force_ocr": force == "true"}
                (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
                output = worker_runner(directory, settings)
                return jsonify(output["body"]), output["status"]
        finally:
            slot.release()

    app.add_url_rule("/v1/text", "text", lambda: handle("text"), methods=["POST"])
    app.add_url_rule("/v1/extract", "extract", lambda: handle("extract"), methods=["POST"])
    return app

