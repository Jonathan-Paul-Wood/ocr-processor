from dataclasses import replace
from io import BytesIO
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest
from reportlab.pdfgen import canvas

from ocr_processor.config import Settings


@pytest.fixture
def settings():
    return Settings(api_key="test-secret-" * 4, llm_model="test-model")


@pytest.fixture
def pdf_bytes():
    output = BytesIO()
    pdf = canvas.Canvas(output)
    pdf.drawString(40, 750, "Example Bank account ending 1234. Statement November 2026.")
    pdf.drawString(40, 720, "2026-11-15 Example Market purchase USD 12.34")
    pdf.save()
    return output.getvalue()


@pytest.fixture
def schema():
    return {"type": "object", "properties": {"amount": {"type": "string", "pattern": r"^\d+\.\d{2}$"}},
            "required": ["amount"], "additionalProperties": False}


@pytest.fixture
def provider(settings):
    state = {"content": '{"amount":"12.34"}', "finish_reason": "stop", "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            for key, value in state.get("headers", {}).items():
                self.send_header(key, value)
            self.end_headers()
            payload = state.get("response", {"choices": [{"finish_reason": state["finish_reason"],
                                                         "message": {"content": state["content"]}}]})
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield replace(settings, llm_base_url=f"http://127.0.0.1:{server.server_port}/v1"), state
    server.shutdown()
    server.server_close()
    thread.join()

