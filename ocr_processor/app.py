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
import time
import uuid
from urllib.parse import urlsplit

from flask import Flask, current_app, g, has_app_context, has_request_context, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import Settings
from .debug_log import DebugLog, body_for_log, forward_worker_events
from .errors import ServiceError, unexpected_error
from .schema import load_json
from .systemone import validate_questions


def run_worker(directory, settings):
    debug = current_app.extensions.get("traffic_debug") if has_app_context() and settings.debug_mode else None
    request_id = g.request_id if has_request_context() else None
    options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(
        [sys.executable, "-m", "ocr_processor.worker", str(directory)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE if settings.debug_mode else subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "OCR_WORKER_SETTINGS": json.dumps(asdict(settings))},
        **options,
    )
    reader = None
    if settings.debug_mode:
        reader = threading.Thread(target=forward_worker_events, args=(process.stdout, debug, request_id),
                                  name="traffic-worker-reader", daemon=True)
        reader.start()
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
    finally:
        if reader is not None:
            reader.join(timeout=2)
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
    if min(settings.timeout, settings.ocr_timeout, settings.systemone_timeout) < 1:
        raise RuntimeError("Request, OCR and SystemOne timeouts must be positive.")
    if not 1 <= settings.systemone_port <= 65535:
        raise RuntimeError("SYSTEMONE_PORT must be between 1 and 65535.")
    if settings.systemone_ip:
        try:
            endpoint = urlsplit(settings.systemone_base_url)
            valid = (endpoint.hostname == settings.systemone_ip.lower() and endpoint.port == settings.systemone_port
                     and not endpoint.username and not endpoint.password and not endpoint.path
                     and not endpoint.query and not endpoint.fragment and not any(c.isspace() for c in settings.systemone_ip))
        except ValueError:
            valid = False
        if not valid:
            raise RuntimeError("SYSTEMONE_IP must be an IP address or hostname without a scheme, port, or path.")
    if settings.llm_base_url:
        endpoint = urlsplit(settings.llm_base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise RuntimeError("LLM_BASE_URL must be an HTTP(S) URL without credentials, query, or fragment.")
    app = Flask(__name__)
    app.debug = False
    debug = DebugLog(settings.debug_log_dir or os.path.join(tempfile.gettempdir(), "ocr-processor-debug"),
                     (settings.api_key, settings.llm_api_key, settings.systemone_api_key)) if settings.debug_mode else None
    app.extensions["traffic_debug"] = debug
    app.config.update(MAX_CONTENT_LENGTH=settings.max_upload_bytes, MAX_FORM_MEMORY_SIZE=512 * 1024, MAX_FORM_PARTS=12)
    slot = threading.BoundedSemaphore(1)

    @app.before_request
    def authorize():
        g.request_id = str(uuid.uuid4())
        if debug:
            g.debug_started = time.monotonic()
            g.debug_body_captured = False
            try:
                debug.emit("api.request", g.request_id, method=request.method, path=request.path,
                           query=request.args.to_dict(flat=False), headers=dict(request.headers),
                           content_type=request.content_type, byte_count=request.content_length)
            except Exception:
                debug.emit("logging.capture_failed", g.request_id, stage="api_request")
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
        if debug:
            try:
                body = ({"omitted": "streaming_response"} if response.is_streamed else
                        body_for_log(response.get_data(), response.content_type or ""))
                fields = {"status": response.status_code, "headers": dict(response.headers), "body": body,
                          "duration_ms": round((time.monotonic() - g.debug_started) * 1000, 3)}
                if request.method in {"POST", "PUT", "PATCH"} and not g.debug_body_captured:
                    fields["request_body_omitted_reason"] = "Request rejected before supported multipart fields could be captured; raw bodies and file bytes are never logged."
                debug.emit("api.response", g.request_id, **fields)
            except Exception:
                debug.emit("logging.capture_failed", g.request_id, stage="api_response")
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
                "systemone_configured": bool(settings.systemone_ip and settings.systemone_model),
                "transaction_extraction_configured": bool(settings.systemone_ip and settings.systemone_model),
                "ocr_available": shutil.which(settings.tesseract_cmd) is not None,
                "limits": {"upload_bytes": settings.max_upload_bytes, "files": settings.max_files,
                           "pages": settings.max_pages, "pixels_per_page": settings.max_pixels,
                           "text_chars": settings.max_text_chars, "timeout_seconds": settings.timeout},
                "formats": ["pdf", "png", "jpeg", "tiff", "webp", "csv"]}

    def handle(mode):
        if mode in {"systemone", "transactions"} and not (settings.systemone_ip and settings.systemone_model):
            missing = [name for name, value in [("SYSTEMONE_IP", settings.systemone_ip),
                                               ("SYSTEMONE_MODEL", settings.systemone_model)] if not value]
            raise ServiceError("systemone_unavailable", "Configure SYSTEMONE_IP and SYSTEMONE_MODEL for decision inference.", 503,
                               details={"stage": "configuration", "missing_settings": missing})
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
            if debug:
                # Use the normal bounded parser; never consume the raw upload for logging.
                try:
                    files = []
                    for field, upload in request.files.items(multi=True):
                        position = upload.stream.tell()
                        try:
                            byte_count = upload.stream.seek(0, os.SEEK_END)
                        finally:
                            upload.stream.seek(position)
                        files.append({"field": field, "filename": upload.filename, "content_type": upload.content_type,
                                      "byte_count": byte_count, "body_omitted": "file_content"})
                    debug.emit("api.input", g.request_id, fields=request.form.to_dict(flat=False), files=files)
                    g.debug_body_captured = True
                except HTTPException:
                    raise
                except Exception:
                    debug.emit("logging.capture_failed", g.request_id, stage="api_input")
            allowed_fields = ({"context", "output_format", "force_ocr"} if mode == "transactions" else
                              {"questions", "force_ocr"} if mode == "systemone" else {"instructions", "schema", "force_ocr"})
            if set(request.files) - {"documents"} or set(request.form) - allowed_fields:
                raise ServiceError("invalid_request", "Unknown request fields.", 400,
                                   details={"stage": "request", "allowed_file_fields": ["documents"],
                                            "allowed_form_fields": sorted(allowed_fields)})
            uploads = request.files.getlist("documents")
            if not 1 <= len(uploads) <= settings.max_files:
                raise ServiceError("invalid_request", f"Supply 1 to {settings.max_files} documents.", 400,
                                   details={"stage": "request", "limit": "files", "actual": len(uploads), "minimum": 1, "maximum": settings.max_files})
            if any(len(request.form.getlist(key)) != 1 for key in request.form):
                raise ServiceError("invalid_request", "Duplicate form fields are not allowed.", 400,
                                   details={"stage": "request", "reason": "duplicate_form_fields"})
            instructions = request.form.get("instructions", "")
            questions = None
            if mode == "systemone":
                questions_text = request.form.get("questions", "")
                if len(questions_text) > 32_000:
                    raise ServiceError("invalid_request", "Questions exceed the limit.", 400,
                                       details={"stage": "request", "field": "questions", "maximum": 32_000})
                try:
                    questions = load_json(questions_text)
                except (ValueError, RecursionError):
                    raise ServiceError("invalid_request", "Supply questions as a JSON object.", 400,
                                       details={"stage": "request", "field": "questions"}) from None
                validate_questions(questions)
            context = None
            if mode == "transactions":
                from .transactions import validate_context
                context_text = request.form.get("context", "{}")
                if len(context_text) > 64_000:
                    raise ServiceError("invalid_request", "Transaction context exceeds 64000 characters.", 400)
                try:
                    context = load_json(context_text)
                except (ValueError, RecursionError):
                    raise ServiceError("invalid_request", "Context must be JSON.", 400) from None
                validate_context(context)
                if "output_format" not in request.form:
                    raise ServiceError("invalid_request", "Supply output_format as a JSON Schema.", 400)
            schema_text = request.form.get("output_format" if mode == "transactions" else "schema", "{}")
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
                if mode == "systemone":
                    job["questions"] = questions
                if mode == "transactions":
                    job["context"] = context
                if debug:
                    job["request_id"] = g.request_id
                (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
                output = worker_runner(directory, settings)
                return jsonify(output["body"]), output["status"]
        finally:
            slot.release()

    app.add_url_rule("/v1/text", "text", lambda: handle("text"), methods=["POST"])
    app.add_url_rule("/v1/extract", "extract", lambda: handle("extract"), methods=["POST"])
    app.add_url_rule("/v1/systemone", "systemone", lambda: handle("systemone"), methods=["POST"])
    app.add_url_rule("/v1/transactions", "transactions", lambda: handle("transactions"), methods=["POST"])
    return app

