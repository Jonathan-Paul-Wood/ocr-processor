from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import threading
import time

import httpx
import pytest

from ocr_processor.app import create_app
from ocr_processor.config import Settings
from ocr_processor.debug_log import DebugLog, MAX_EVENT_BYTES, body_for_log, encode_event, forward_worker_events, new_event, safe_url
from ocr_processor.extraction import extract
from ocr_processor.schema import validate_schema


@pytest.mark.parametrize("value,expected", [(None, False), ("FALSE", False), ("TRUE", True), ("true", True),
                                           (" TrUe ", True), ("1", False), ("yes", False), ("", False)])
def test_debug_setting(value, expected, monkeypatch):
    monkeypatch.delenv("DEBUG_MODE", raising=False)
    if value is not None:
        monkeypatch.setenv("DEBUG_MODE", value)
    assert Settings.from_env().debug_mode is expected


@pytest.fixture
def debug_app(settings, tmp_path):
    apps = []

    def make(config=None, runner=None):
        config = replace(config or settings, debug_mode=True, debug_log_dir=str(tmp_path / f"logs-{len(apps)}"))
        app = create_app(config, runner) if runner else create_app(config)
        apps.append(app)
        return app, config

    yield make
    for app in apps:
        app.extensions["traffic_debug"].close()


def read_events(app):
    logger = app.extensions["traffic_debug"]
    logger.close()
    return [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]


def test_disabled_mode_has_no_capture_work(settings, tmp_path, monkeypatch, pdf_bytes):
    import ocr_processor.app as app_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Debug capture must not run")

    monkeypatch.setattr(app_module, "DebugLog", forbidden)
    monkeypatch.setattr(app_module, "body_for_log", forbidden)
    directory = tmp_path / "disabled"
    app = create_app(replace(settings, debug_log_dir=str(directory)), lambda *args: {"status": 200, "body": {"ok": True}})
    client = app.test_client()
    assert client.post("/v1/text", data={"documents": (BytesIO(pdf_bytes), "statement.pdf")},
                       headers={"Authorization": "Bearer " + settings.api_key}).json == {"ok": True}
    assert app.extensions["traffic_debug"] is None
    assert not directory.exists()


def test_api_bodies_and_file_metadata_without_file_bytes(debug_app, schema):
    raw = b"%PDF-raw-file-secret-must-not-be-recorded"

    def runner(directory, settings):
        job = json.loads((directory / "job.json").read_text())
        assert Path(job["files"][0]).read_bytes() == raw
        assert job["request_id"]
        return {"status": 200, "body": {"data": {"amount": "12.34"}}}

    app, config = debug_app(runner=runner)
    client = app.test_client()
    response = client.post("/v1/text?token=secret-query", data={
        "documents": (BytesIO(raw), "statement.pdf", "application/pdf"),
        "instructions": "Extract transactions", "schema": json.dumps(schema),
    }, headers={"Authorization": "Bearer " + config.api_key, "X-API-Key": "unknown-header-secret"})
    assert response.status_code == 200
    assert not app.debug
    events = read_events(app)
    assert [event["event"] for event in events] == ["api.request", "api.input", "api.response"]
    assert all(event["request_id"] == response.headers["X-Request-ID"] for event in events)
    assert all(event["timestamp"].endswith("Z") for event in events)
    assert events[1]["fields"]["instructions"] == ["Extract transactions"]
    assert events[1]["files"] == [{"field": "documents", "filename": "statement.pdf", "content_type": "application/pdf",
                                    "byte_count": len(raw), "body_omitted": "file_content"}]
    assert events[2]["body"] == response.json
    assert events[2]["duration_ms"] >= 0
    text = app.extensions["traffic_debug"].path.read_text()
    for secret in (raw.decode(), config.api_key, "unknown-header-secret", "secret-query"):
        assert secret not in text


@pytest.mark.parametrize("kind,status", [("unauthorized", 401), ("oversized", 413), ("wrong_type", 400)])
def test_rejections_record_why_body_was_not_captured(debug_app, settings, kind, status):
    app, config = debug_app(replace(settings, max_upload_bytes=100))
    headers = {} if kind == "unauthorized" else {"Authorization": "Bearer " + config.api_key}
    if kind == "wrong_type":
        response = app.test_client().post("/v1/text", json={"private": "unsupported-body"}, headers=headers)
    else:
        response = app.test_client().post("/v1/text", data={"documents": (BytesIO(b"binary-secret" * 100), "document.pdf")}, headers=headers)
    assert response.status_code == status
    events = read_events(app)
    assert events[-1]["event"] == "api.response"
    assert events[-1]["body"] == response.json
    assert "request_body_omitted_reason" in events[-1]
    assert "binary-secret" not in json.dumps(events)
    assert "unsupported-body" not in json.dumps(events)


@pytest.mark.parametrize("provider_status", [200, 400])
def test_real_worker_llm_traffic_correlates_with_api(debug_app, provider, pdf_bytes, schema, provider_status):
    config, state = provider
    config = replace(config, llm_api_key="llm-secret-value")
    state["status"] = provider_status
    state["headers"] = {"X-Request-ID": "provider-123", "Set-Cookie": "private-cookie"}
    if provider_status == 400:
        state["response"] = {"error": {"code": "bad_model", "message": "Specific model rejection. llm-secret-value"}}
    app, config = debug_app(config)
    response = app.test_client().post("/v1/extract", data={"documents": (BytesIO(pdf_bytes), "bank.pdf"),
        "instructions": "Extract purchase amount.", "schema": json.dumps(schema)},
        headers={"Authorization": "Bearer " + config.api_key})
    assert response.status_code == (200 if provider_status == 200 else 502), response.json
    events = read_events(app)
    outgoing = next(event for event in events if event["event"] == "llm.request")
    incoming = next(event for event in events if event["event"] == "llm.response")
    assert outgoing["url"] == config.llm_base_url + "/chat/completions"
    assert outgoing["body"] == state["requests"][0]
    assert "12.34" in outgoing["body"]["messages"][-1]["content"]
    assert incoming["status"] == provider_status
    assert incoming["headers"]["x-request-id"] == "provider-123"
    assert all(event["request_id"] == response.headers["X-Request-ID"] for event in events)
    assert events[-1]["event"] == "api.response"
    serialized = json.dumps(events)
    for secret in (config.api_key, config.llm_api_key, "private-cookie"):
        assert secret not in serialized
    if provider_status == 400:
        assert "Specific model rejection" in serialized


def test_network_error_cause_chain_is_recorded_and_redacted(settings, tmp_path, schema, monkeypatch):
    from ocr_processor.errors import ServiceError
    logger = DebugLog(tmp_path, (settings.api_key,))

    class BoundLogger:
        def emit(self, event, **fields):
            logger.emit(event, "network-request", **fields)

    def fail(*args, **kwargs):
        try:
            raise ConnectionRefusedError(111, "Connection refused " + settings.api_key)
        except ConnectionRefusedError as cause:
            raise httpx.ConnectError("All connection attempts failed") from cause

    monkeypatch.setattr(httpx.Client, "stream", fail)
    try:
        with pytest.raises(ServiceError) as error:
            extract([], "", schema, validate_schema(schema), replace(settings, llm_base_url="http://model/v1"),
                    time.monotonic() + 10, debug_log=BoundLogger())
        assert error.value.code == "provider_error"
    finally:
        logger.close()
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [cause["type"] for cause in events[-1]["causes"]] == ["ConnectError", "ConnectionRefusedError"]
    assert "Connection refused" in events[-1]["causes"][-1]["message"]
    assert settings.api_key not in logger.path.read_text()


def test_event_cap_is_valid_utf8_json_with_original_size_and_redaction():
    event = new_event("large", "req", body="secret-value" + "\U0001f600\"\\\n" * 400000)
    original = encode_event(event, ("secret-value",), max_bytes=20 * 1024 * 1024)
    capped = encode_event(event, ("secret-value",))
    assert len(capped) <= MAX_EVENT_BYTES
    parsed = json.loads(capped)
    assert parsed["truncated"] is True
    assert parsed["original_size_bytes"] == len(original)
    assert parsed["request_id"] == "req"
    assert "secret-value" not in capped.decode()


def test_fifo_keeps_exactly_latest_100_events(tmp_path):
    logger = DebugLog(tmp_path)
    for index in range(105):
        logger.emit("number", number=index)
        logger._queue.join()
    logger.close()
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert len(events) == 100
    assert [event["number"] for event in events] == list(range(5, 105))
    assert not logger.path.with_suffix(".tmp").exists()


def test_storage_budget_evicts_earlier(tmp_path):
    logger = DebugLog(tmp_path, max_retained_bytes=900)
    for index in range(20):
        logger.emit("number", number=index, body="x" * 100)
        logger._queue.join()
    logger.close()
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert logger.path.stat().st_size <= 900
    assert 1 <= len(events) < 20
    assert events[-1]["number"] == 19


def test_startup_resets_only_debug_files(tmp_path):
    first = DebugLog(tmp_path)
    first.emit("previous_session")
    first.close()
    marker = tmp_path / "unrelated.txt"
    marker.write_text("keep")
    first.path.with_suffix(".tmp").write_text("interrupted write")
    second = DebugLog(tmp_path)
    second.emit("new_session")
    second.close()
    assert [json.loads(line)["event"] for line in second.path.read_text().splitlines()] == ["new_session"]
    assert marker.read_text() == "keep"


def test_concurrent_emit_and_snapshot_reads_stay_valid(tmp_path):
    logger = DebugLog(tmp_path)
    failures = []

    def work(index):
        logger.emit("concurrent", number=index)
        try:
            for line in logger.path.read_text().splitlines():
                json.loads(line)
        except Exception as exc:
            failures.append(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(100)))
    logger.close()
    assert not failures
    assert all(isinstance(json.loads(line), dict) for line in logger.path.read_text().splitlines())


def test_queue_overflow_is_nonblocking_and_counted(tmp_path, monkeypatch):
    logger = DebugLog(tmp_path)
    entered, release = threading.Event(), threading.Event()
    persist = logger._persist

    def slow(line):
        entered.set()
        release.wait(timeout=3)
        persist(line)

    monkeypatch.setattr(logger, "_persist", slow)
    logger.emit("first")
    assert entered.wait(timeout=2)
    for index in range(12):
        logger.emit("queued", number=index)
    assert logger._queue.qsize() == 8
    release.set()
    logger.close()
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert sum(event["count"] for event in events if event["event"] == "logging.dropped_events") == 4


def test_unwritable_directory_does_not_break_extraction(debug_app, settings, tmp_path, capsys):
    obstacle = tmp_path / "not-a-directory"
    obstacle.write_text("x")
    config = replace(settings, debug_mode=True, debug_log_dir=str(obstacle))
    app = create_app(config, lambda *args: {"status": 200, "body": {"ok": True}})
    try:
        response = app.test_client().post("/v1/text", data={"documents": (BytesIO(b"test"), "image.png")},
                                          headers={"Authorization": "Bearer " + config.api_key})
        assert response.json == {"ok": True}
        assert "extraction continues" in capsys.readouterr().err
    finally:
        app.extensions["traffic_debug"].close()


def test_disk_write_failure_does_not_break_requests(debug_app, monkeypatch, capsys):
    app, config = debug_app()
    logger = app.extensions["traffic_debug"]

    def fail(*args):
        raise OSError("private OS error must not leak into fallback stderr")

    monkeypatch.setattr(logger, "_persist", fail)
    response = app.test_client().get("/health")
    assert response.json == {"status": "ok"}
    logger.close()
    stderr = capsys.readouterr().err
    assert "extraction continues" in stderr
    assert "private OS error" not in stderr


def test_shutdown_wait_is_bounded(tmp_path, monkeypatch):
    logger = DebugLog(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = logger._persist

    def slow(line):
        entered.set()
        release.wait(timeout=3)
        original(line)

    monkeypatch.setattr(logger, "_persist", slow)
    logger.emit("pending")
    assert entered.wait(timeout=1)
    started = time.monotonic()
    logger.close(timeout=0.05)
    assert time.monotonic() - started < 0.5
    release.set()
    logger.close()


def test_transmitted_binary_bodies_are_metadata_only():
    assert body_for_log(b"%PDF-binary-document", "application/pdf") == {
        "omitted": "binary_content", "byte_count": 20, "content_type": "application/pdf"}
    assert body_for_log(b"image-content", "image/png")["omitted"] == "binary_content"
    assert safe_url("http://user:password@model/v1?token=secret&name=model") == "http://model/v1?token=%5BREDACTED%5D&name=model"


def test_worker_pipe_discards_invalid_or_oversized_lines_and_keeps_draining(tmp_path):
    logger = DebugLog(tmp_path)
    stream = BytesIO(b"non-json private message\n" + b"x" * (MAX_EVENT_BYTES + 20) + b"\n" +
                     encode_event(new_event("llm.request", "wrong-id", body={"model": "test"})))
    forward_worker_events(stream, logger, "real-id")
    logger.close()
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["logging.invalid_worker_event", "logging.invalid_worker_event", "llm.request"]
    assert all(event["request_id"] == "real-id" for event in events)
    assert "non-json private message" not in logger.path.read_text()


def test_body_capture_failure_preserves_llm_success(provider, schema, monkeypatch):
    import ocr_processor.debug_log as log_module
    config, _ = provider
    records = []

    class Recorder:
        def emit(self, event, **fields):
            records.append((event, fields))

    def fail(*args):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(log_module, "body_for_log", fail)
    result = extract([], "Extract amount", schema, validate_schema(schema), config,
                     time.monotonic() + 10, debug_log=Recorder())
    assert result == {"amount": "12.34"}
    assert records[-1] == ("logging.capture_failed", {"stage": "llm_response"})


def test_body_capture_failure_preserves_api_response(debug_app, monkeypatch):
    import ocr_processor.app as app_module
    app, _ = debug_app()

    def fail(*args):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(app_module, "body_for_log", fail)
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}
    assert read_events(app)[-1]["event"] == "logging.capture_failed"
