from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import threading
import subprocess
import sys

import pytest

from ocr_processor.app import create_app


def upload(pdf_bytes, schema=None):
    data = {"documents": (BytesIO(pdf_bytes), "../../statement.pdf")}
    if schema is not None:
        data.update(schema=json.dumps(schema), instructions="Extract the purchase amount.")
    return data


def headers(settings):
    return {"Authorization": "Bearer " + settings.api_key}


def test_auth_health_and_limits(settings):
    client = create_app(settings).test_client()
    assert client.get("/health").json == {"status": "ok"}
    assert client.post("/v1/text").status_code == 401
    response = client.get("/v1/capabilities", headers=headers(settings))
    assert response.json["limits"]["pages"] == 30
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Request-ID"]
    with pytest.raises(RuntimeError):
        create_app(replace(settings, api_key=""))


def test_actual_pdf_worker_and_cleanup(settings, pdf_bytes):
    from ocr_processor.app import run_worker
    seen = []

    def runner(directory, config):
        seen.append(directory)
        assert (directory / "document-0").exists()
        assert not (directory / "statement.pdf").exists()
        return run_worker(directory, config)

    client = create_app(settings, runner).test_client()
    response = client.post("/v1/text", data=upload(pdf_bytes), headers=headers(settings))
    assert response.status_code == 200, response.json
    assert "12.34" in response.json["pages"][0]["text"]
    assert response.json["pages"][0]["method"] == "embedded"
    assert not seen[0].exists()


def test_actual_extraction_worker(provider, pdf_bytes, schema):
    settings, state = provider
    client = create_app(settings).test_client()
    response = client.post("/v1/extract", data=upload(pdf_bytes, schema), headers=headers(settings))
    assert response.status_code == 200, response.json
    assert response.json["data"] == {"amount": "12.34"}
    assert response.json["warnings"]
    sent = state["requests"][0]
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert "12.34" in sent["messages"][-1]["content"]


@pytest.mark.parametrize("content,finish,status,code", [
    ('{"amount":12.34}', "stop", 200, "invalid_extraction"),
    ('{"amount":"12.34","extra":1}', "stop", 200, "invalid_extraction"),
    ('not JSON', "stop", 200, "invalid_extraction"),
    ('{"amount":"12.34"}', "length", 200, "incomplete_extraction"),
    ('secret provider error', "stop", 401, "provider_error"),
])
def test_provider_failures(provider, pdf_bytes, schema, content, finish, status, code):
    settings, state = provider
    state.update(content=content, finish_reason=finish, status=status)
    response = create_app(settings).test_client().post("/v1/extract", data=upload(pdf_bytes, schema), headers=headers(settings))
    assert response.status_code == 502, response.json
    assert response.json["error"]["code"] == code
    assert "secret provider error" not in response.text


def test_schema_rejected_before_provider(provider, pdf_bytes, schema):
    settings, state = provider
    schema["properties"]["amount"] = {"$ref": "https://invalid.example/schema"}
    response = create_app(settings).test_client().post("/v1/extract", data=upload(pdf_bytes, schema), headers=headers(settings))
    assert response.status_code == 400
    assert state["requests"] == []


@pytest.mark.parametrize("document", [b"", b"not a PDF or image", b"%PDF-broken"])
def test_bad_documents(settings, document):
    response = create_app(settings).test_client().post("/v1/text", data=upload(document), headers=headers(settings))
    assert response.status_code in {400, 422}


def test_request_validation_and_unconfigured_extractor(settings, pdf_bytes):
    client = create_app(settings).test_client()
    assert client.post("/v1/extract", headers=headers(settings)).status_code == 503
    assert client.post("/v1/text", headers=headers(settings), json={}).status_code == 400
    assert client.post("/v1/text", headers=headers(settings), data={"force_ocr": "yes"}).status_code == 400
    client = create_app(replace(settings, max_upload_bytes=100)).test_client()
    assert client.post("/v1/text", headers=headers(settings), data=upload(pdf_bytes)).status_code == 413


def test_busy_request_does_not_queue(settings, pdf_bytes):
    started, release = threading.Event(), threading.Event()

    def runner(*args):
        started.set()
        assert release.wait(timeout=5)
        return {"status": 200, "body": {}}

    app = create_app(settings, runner)
    thread = threading.Thread(target=lambda: app.test_client().post("/v1/text", data=upload(pdf_bytes), headers=headers(settings)))
    thread.start()
    assert started.wait(timeout=5)
    try:
        response = app.test_client().post("/v1/text", data=upload(pdf_bytes), headers=headers(settings))
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "5"
    finally:
        release.set()
        thread.join()


def test_worker_crash_releases_slot_and_cleans_up(settings, pdf_bytes):
    from ocr_processor.errors import ServiceError
    paths = []

    def runner(directory, config):
        paths.append(directory)
        raise ServiceError("worker_failed", "Worker failed.", 500)

    client = create_app(settings, runner).test_client()
    for _ in range(2):
        assert client.post("/v1/text", data=upload(pdf_bytes), headers=headers(settings)).status_code == 500
    assert all(not path.exists() for path in paths)


def test_hard_deadline_kills_worker(settings, tmp_path, monkeypatch):
    from ocr_processor.app import run_worker
    from ocr_processor.errors import ServiceError
    original_popen = subprocess.Popen
    children = []

    def sleeper(command, **kwargs):
        if command[0] == "taskkill":
            return original_popen(command, **kwargs)
        child = original_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", sleeper)
    with pytest.raises(ServiceError) as exc:
        run_worker(tmp_path, replace(settings, timeout=0))
    assert exc.value.status == 504
    assert children[0].poll() is not None


def test_configuration_rejects_unsafe_provider_url(settings):
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        create_app(replace(settings, llm_base_url="file:///etc/passwd"))

