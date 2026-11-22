from dataclasses import replace
from io import BytesIO
import json
import subprocess
import time

import httpx
from PIL import Image
import pytest
from reportlab.pdfgen import canvas

from ocr_processor import documents, extraction
from ocr_processor.app import create_app
from ocr_processor.documents import recognize
from ocr_processor.errors import ServiceError
from ocr_processor.schema import validate_schema


def post(settings, data, path="/v1/text", runner=None):
    app = create_app(settings, runner) if runner else create_app(settings)
    return app.test_client().post(path, headers={"Authorization": "Bearer " + settings.api_key}, data=data)


def test_upload_size_reports_actual_and_limit(settings, pdf_bytes):
    response = post(replace(settings, max_upload_bytes=100), {"documents": (BytesIO(pdf_bytes), "private.pdf")})
    details = response.json["error"]["details"]
    assert response.status_code == 413
    assert details["limit"] == "upload_bytes"
    assert details["actual"] > details["maximum"] == 100
    assert details["request_id"] == response.headers["X-Request-ID"]
    assert "private.pdf" not in response.text


def test_pdf_page_limit_survives_subprocess(settings):
    output = BytesIO()
    pdf = canvas.Canvas(output)
    for _ in range(2):
        pdf.drawString(40, 750, "Private document content must not appear in error details.")
        pdf.showPage()
    pdf.save()
    response = post(replace(settings, max_pages=1), {"documents": (BytesIO(output.getvalue()), "private.pdf")})
    assert response.status_code == 413
    details = response.json["error"]["details"]
    assert details["document"] == 1
    assert (details["limit"], details["actual"], details["maximum"]) == ("pages", 2, 1)
    assert details["stage"] == "document_read"
    assert "Private document content" not in response.text


def test_pixel_limit_reports_image_dimensions(settings):
    output = BytesIO()
    Image.new("RGB", (100, 200)).save(output, format="PNG")
    response = post(replace(settings, max_pixels=50), {"documents": (BytesIO(output.getvalue()), "image.png")})
    details = response.json["error"]["details"]
    assert response.status_code == 413
    assert (details["document"], details["page"]) == (1, 1)
    assert (details["width_pixels"], details["height_pixels"], details["actual"], details["maximum"]) == (100, 200, 20000, 50)


def test_text_limit_reports_cumulative_characters(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "recognize", lambda *args: "private receipt text")
    path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(path)
    with pytest.raises(ServiceError) as raised:
        documents.read_documents([path], tmp_path, replace(settings, max_text_chars=5), time.monotonic() + 10)
    assert raised.value.details["actual"] == len("private receipt text")
    assert raised.value.details["maximum"] == 5
    assert "private receipt text" not in json.dumps(raised.value.payload())


def test_missing_configuration_names_only(settings):
    response = post(settings, {}, "/v1/extract")
    assert response.json["error"]["details"]["missing_settings"] == ["LLM_BASE_URL"]
    assert settings.api_key not in response.text


def test_provider_rejection_explains_failure(provider, pdf_bytes, schema):
    settings, state = provider
    state.update(status=400, headers={"X-Request-ID": "req-test123"}, response={"error": {
        "code": "invalid_json_schema", "type": "invalid_request_error", "param": "response_format",
        "message": "Private document text and provider credentials must not be forwarded.",
    }})
    response = post(settings, {"documents": (BytesIO(pdf_bytes), "bank.pdf"), "instructions": "Extract amount", "schema": json.dumps(schema)}, "/v1/extract")
    assert response.status_code == 502
    details = response.json["error"]["details"]
    assert details["provider_http_status"] == 400
    assert details["provider_error_code"] == "invalid_json_schema"
    assert details["provider_error_param"] == "response_format"
    assert details["provider_request_id"] == "req-test123"
    assert details["request_id"] == response.headers["X-Request-ID"]
    assert "Private document text" not in response.text


def test_schema_failure_reports_paths_without_values(provider, pdf_bytes, schema):
    settings, state = provider
    state["content"] = '{"amount":"private-invalid-value"}'
    response = post(settings, {"documents": (BytesIO(pdf_bytes), "bank.pdf"), "instructions": "Extract amount", "schema": json.dumps(schema)}, "/v1/extract")
    details = response.json["error"]["details"]
    assert details["stage"] == "output_validation"
    assert details["validator"] == "pattern"
    assert details["instance_path"] == ["amount"]
    assert details["schema_path"] == ["properties", "amount", "pattern"]
    assert "private-invalid-value" not in response.text


def test_network_failure_has_exception_type_without_secrets(settings, schema, monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("private-document " + settings.api_key)

    monkeypatch.setattr(httpx.Client, "stream", fail)
    with pytest.raises(ServiceError) as raised:
        extraction.extract([], "", schema, validate_schema(schema), replace(settings, llm_base_url="http://unreachable/v1"), time.monotonic() + 10)
    assert raised.value.details["exception_type"] == "ConnectError"
    assert settings.api_key not in json.dumps(raised.value.payload())
    assert "private-document" not in json.dumps(raised.value.payload())


def test_tesseract_failure_classifies_language_error(settings, tmp_path, monkeypatch):
    def fail(command, **kwargs):
        kwargs["stderr"].write(b"Error opening data file /private/path/eng.traineddata\nprivate document text")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ServiceError) as raised:
        recognize(Image.new("RGB", (10, 10)), tmp_path, settings, time.monotonic() + 10)
    assert raised.value.details["exit_code"] == 1
    assert raised.value.details["reason"] == "language_data_unavailable"
    assert "private" not in json.dumps(raised.value.payload())


def test_unexpected_worker_exception_keeps_stage_and_location(settings, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("private document text " + settings.api_key)

    monkeypatch.setattr(documents, "recognize", fail)
    path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(path)

    def runner(directory, config):
        # Exercise the worker's real fallback serialization with an injected failure.
        import sys
        from dataclasses import asdict
        from ocr_processor.worker import main
        monkeypatch.setenv("OCR_WORKER_SETTINGS", json.dumps(asdict(config)))
        monkeypatch.setattr(sys, "argv", ["worker", str(directory)])
        main()
        return json.loads((directory / "result.json").read_text())

    response = post(settings, {"documents": (BytesIO(path.read_bytes()), "image.png")}, runner=runner)
    assert response.status_code == 500
    details = response.json["error"]["details"]
    assert details["stage"] == "ocr"
    assert (details["document"], details["page"]) == (1, 1)
    assert details["exception_type"] == "RuntimeError"
    assert any(frame["file"] == "documents.py" for frame in details["locations"])
    assert settings.api_key not in response.text
    assert "private document text" not in response.text


def test_native_pdf_error_has_diagnostics(settings):
    response = post(settings, {"documents": (BytesIO(b"%PDF-broken"), "bank.pdf")})
    assert response.status_code == 422
    assert response.json["error"]["details"]["exception_type"] == "PdfiumError"

