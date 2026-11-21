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
from .errors import ServiceError
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
        raise ServiceError("timeout", "Document processing exceeded its time limit.", 504) from None
    result = directory / "result.json"
    if process.returncode != 0 or not result.exists():
        raise ServiceError("worker_failed", "Document worker exited unexpectedly.", 500)
    return load_json(result.read_text(encoding="utf-8"))


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
    app.config.update(MAX_CONTENT_LENGTH=settings.max_upload_bytes, MAX_FORM_MEMORY_SIZE=64 * 1024, MAX_FORM_PARTS=12)
    slot = threading.BoundedSemaphore(1)

    @app.before_request
    def authorize():
        g.request_id = str(uuid.uuid4())
        if request.path != "/health":
            expected = "Bearer " + settings.api_key
            if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), expected.encode()):
                raise ServiceError("unauthorized", "A valid bearer API key is required.", 401)

    @app.after_request
    def headers(response):
        response.headers["X-Request-ID"] = g.request_id
        response.headers["Cache-Control"] = "no-store"
        if response.status_code in {429, 503}:
            response.headers["Retry-After"] = "5"
        if response.status_code == 401:
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.errorhandler(ServiceError)
    def service_error(exc):
        return jsonify(exc.payload()), exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error={"code": exc.name.lower().replace(" ", "_"), "message": exc.description}), exc.code

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
            raise ServiceError("extraction_unavailable", "Configure LLM_BASE_URL and LLM_MODEL for structured extraction.", 503)
        if not slot.acquire(blocking=False):
            raise ServiceError("busy", "Another document is being processed; retry later.", 429)
        try:
            if request.mimetype != "multipart/form-data":
                raise ServiceError("invalid_request", "Use multipart/form-data.", 400)
            if set(request.files) - {"documents"} or set(request.form) - {"instructions", "schema", "force_ocr"}:
                raise ServiceError("invalid_request", "Unknown request fields.", 400)
            uploads = request.files.getlist("documents")
            if not 1 <= len(uploads) <= settings.max_files:
                raise ServiceError("invalid_request", f"Supply 1 to {settings.max_files} documents.", 400)
            if any(len(request.form.getlist(key)) != 1 for key in request.form):
                raise ServiceError("invalid_request", "Duplicate form fields are not allowed.", 400)
            instructions = request.form.get("instructions", "")
            schema_text = request.form.get("schema", "{}")
            if len(instructions) > 16_000 or len(schema_text) > 32_000:
                raise ServiceError("invalid_request", "Instructions or schema exceed the limit.", 400)
            if mode == "extract" and (not instructions.strip() or "schema" not in request.form):
                raise ServiceError("invalid_request", "Supply instructions and a JSON Schema.", 400)
            try:
                schema = load_json(schema_text)
            except (ValueError, RecursionError):
                raise ServiceError("invalid_schema", "Schema must be valid JSON.", 400) from None
            force = request.form.get("force_ocr", "false")
            if force not in {"true", "false"}:
                raise ServiceError("invalid_request", "force_ocr must be true or false.", 400)
            with tempfile.TemporaryDirectory(prefix="ocr-") as temporary:
                directory = Path(temporary)
                paths = []
                for index, upload in enumerate(uploads):
                    # Caller filenames and MIME types never determine file paths or parsing.
                    path = directory / f"document-{index}"
                    upload.save(path)
                    if path.stat().st_size == 0:
                        raise ServiceError("invalid_document", "Empty documents are not supported.", 400)
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

