from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time

import httpx
import pytest

from ocr_processor.app import create_app
from ocr_processor.config import Settings
from ocr_processor.debug_log import encode_event
from ocr_processor.errors import ServiceError
from ocr_processor.systemone import infer


QUESTIONS = {
    "category": {"type": "choice", "instructions": "Categorize this purchase.",
                 "criteria": {"groceries": "Food shopping", "travel": "Transportation"}},
    "urgent": {"type": "noul", "instructions": "Does this need review?"},
    "priority": {"type": "score", "criteria": ["low", "medium", "high"]},
}
RESPONSE = {"model": "kev-latest", "answers": {
    "category": {"type": "choice", "choice": "groceries", "confidence": 0.8,
                 "probabilities": {"groceries": 0.9, "travel": 0.1}},
    "urgent": {"type": "noul", "noul": 0.2},
    "priority": {"type": "score", "score": 1.4}}, "usage": {"input_tokens": 100}, "latency_ms": 10}


@pytest.fixture
def ranker(settings):
    state = {"status": 200, "response": RESPONSE, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, self.headers.get("Authorization"), body))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(state.get("raw", json.dumps(state["response"]).encode()))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield replace(settings, llm_model="", systemone_ip="127.0.0.1", systemone_port=server.server_port,
                  systemone_api_key="ranker-secret"), state
    server.shutdown()
    server.server_close()
    thread.join()


def post(settings, pdf_bytes, questions=QUESTIONS, **fields):
    return create_app(settings).test_client().post(
        "/v1/systemone", headers={"Authorization": "Bearer " + settings.api_key},
        data={"documents": (BytesIO(pdf_bytes), "statement.pdf"), "questions": json.dumps(questions), **fields})


def test_systemone_actual_worker_without_llm(ranker, pdf_bytes):
    settings, state = ranker
    response = post(settings, pdf_bytes)
    assert response.status_code == 200, response.json
    assert response.json["data"] == RESPONSE
    assert response.json["warnings"]
    path, authorization, body = state["requests"][0]
    assert path == "/v1/systemone"
    assert authorization == "Bearer ranker-secret"
    assert body == {"model": "kev-latest", "state": {"pages": response.json["pages"]}, "questions": QUESTIONS}
    assert "12.34" in body["state"]["pages"][0]["text"]
    client = create_app(settings).test_client()
    headers = {"Authorization": "Bearer " + settings.api_key}
    capabilities = client.get("/v1/capabilities", headers=headers).json
    assert capabilities["systemone_configured"] is True
    assert capabilities["extraction_configured"] is False
    assert client.post("/v1/extract", headers=headers).status_code == 503


def test_systemone_auth_and_configuration(settings):
    client = create_app(settings).test_client()
    assert client.post("/v1/systemone").status_code == 401
    response = client.post("/v1/systemone", headers={"Authorization": "Bearer " + settings.api_key})
    assert response.status_code == 503
    assert response.json["error"]["details"]["missing_settings"] == ["SYSTEMONE_IP"]


@pytest.mark.parametrize("questions", [None, [], {}, {"a": {"type": "text"}},
    {"a": {"type": "choice", "criteria": {}}}, {"a": {"type": "score", "criteria": {"x": None}}},
    {"a": {"type": "noul", "label": True}}, {"a": {"type": "score", "criteria": list(range(256))}}])
def test_invalid_questions_rejected_before_worker(ranker, pdf_bytes, questions):
    settings, state = ranker
    response = post(settings, pdf_bytes, questions)
    assert response.status_code == 400, response.json
    assert state["requests"] == []


def test_systemone_rejects_llm_fields(ranker, pdf_bytes):
    settings, state = ranker
    assert post(settings, pdf_bytes, instructions="Ignored?").status_code == 400
    assert post(settings, pdf_bytes, schema="{}").status_code == 400
    assert state["requests"] == []


@pytest.mark.parametrize("status", [302, 401, 422, 429, 503])
def test_ranker_rejections_are_sanitized(ranker, pdf_bytes, status):
    settings, state = ranker
    state.update(status=status, response={"detail": "private document ranker-secret"})
    response = post(settings, pdf_bytes)
    assert response.status_code == 502
    assert response.json["error"]["details"]["provider_http_status"] == status
    assert "private document" not in response.text
    assert "ranker-secret" not in response.text


@pytest.mark.parametrize("raw,code", [(b"invalid", "invalid_inference"), (b"{}", "invalid_inference"),
    (b'{"model":"kev-latest","answers":{}}', "invalid_inference"),
    (b"x" * (1024 * 1024 + 1), "provider_error")])
def test_bad_and_oversized_ranker_responses(ranker, raw, code):
    settings, state = ranker
    state["raw"] = raw
    with pytest.raises(ServiceError) as exc:
        infer([], QUESTIONS, settings, time.monotonic() + 10)
    assert exc.value.code == code


@pytest.mark.parametrize("exception,status", [(httpx.ReadTimeout("secret"), 504), (httpx.ConnectError("secret"), 502)])
def test_ranker_network_errors(settings, monkeypatch, exception, status):
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(httpx.Client, "stream", fail)
    with pytest.raises(ServiceError) as exc:
        infer([], QUESTIONS, settings, time.monotonic() + 10)
    assert exc.value.status == status
    assert "secret" not in json.dumps(exc.value.payload())
    with pytest.raises(ServiceError) as expired:
        infer([], QUESTIONS, settings, time.monotonic() - 1)
    assert expired.value.status == 504


def test_systemone_env(monkeypatch):
    for key, value in {"SYSTEMONE_IP": "10.0.0.2", "SYSTEMONE_PORT": "9000",
                       "SYSTEMONE_API_KEY": "different-secret", "SYSTEMONE_MODEL": "custom-model"}.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.systemone_base_url == "http://10.0.0.2:9000"
    assert settings.systemone_api_key == "different-secret"
    assert settings.systemone_model == "custom-model"
    assert b"different-secret" not in encode_event({"systemone_api_key": "different-secret"})


@pytest.mark.parametrize("host", ["http://127.0.0.1", "host:8009", "host/path", "host?key=secret", "user@host", "host name"])
def test_bad_systemone_hosts(settings, host):
    with pytest.raises(RuntimeError, match="SYSTEMONE_IP"):
        create_app(replace(settings, systemone_ip=host))


def test_systemone_ipv6(settings):
    settings = replace(settings, systemone_ip="::1")
    assert settings.systemone_base_url == "http://[::1]:8009"
    create_app(settings)


def test_systemone_debug_redacts_ranker_credentials(ranker, pdf_bytes, tmp_path):
    settings, state = ranker
    settings = replace(settings, debug_mode=True, debug_log_dir=str(tmp_path))
    app = create_app(settings)
    try:
        response = app.test_client().post(
            "/v1/systemone", headers={"Authorization": "Bearer " + settings.api_key},
            data={"documents": (BytesIO(pdf_bytes), "statement.pdf"), "questions": json.dumps(QUESTIONS)})
        assert response.status_code == 200, response.json
    finally:
        app.extensions["traffic_debug"].close()
    raw = (tmp_path / "traffic.jsonl").read_text(encoding="utf-8")
    assert "ranker-secret" not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    assert {"systemone.request", "systemone.response"} <= {event["event"] for event in events}
    assert state["requests"][0][1] == "Bearer ranker-secret"


@pytest.mark.parametrize("answer", [
    {"category": {"type": "choice", "choice": "unknown"}},
    {"urgent": {"type": "noul", "noul": True}},
    {"urgent": {"type": "noul", "noul": 1.1}},
    {"priority": {"type": "score", "score": 3}},
])
def test_invalid_answer_values(ranker, answer):
    settings, state = ranker
    state["response"] = {**RESPONSE, "answers": {**RESPONSE["answers"], **answer}}
    with pytest.raises(ServiceError) as exc:
        infer([], QUESTIONS, settings, time.monotonic() + 10)
    assert exc.value.code == "invalid_inference"


def test_nonfinite_metadata_rejected(ranker):
    settings, state = ranker
    state["raw"] = json.dumps(RESPONSE).replace('"latency_ms": 10', '"latency_ms": 1e999').encode()
    with pytest.raises(ServiceError) as exc:
        infer([], QUESTIONS, settings, time.monotonic() + 10)
    assert exc.value.code == "invalid_inference"
