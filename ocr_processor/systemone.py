"""OCR text adapter for rank-categorizer's typed decision API."""
import json
import math
import time

import httpx
from jsonschema import Draft202012Validator

from .debug_log import body_for_log, exception_causes
from .errors import ServiceError
from .schema import load_json


QUESTION_SCHEMA = {
    "type": "object", "minProperties": 1,
    "additionalProperties": {
        "type": "object", "required": ["type"], "additionalProperties": False,
        "properties": {"type": {"enum": ["choice", "score", "noul"]}, "instructions": {}, "criteria": {}},
        "allOf": [
            {"if": {"properties": {"type": {"const": "choice"}}},
             "then": {"required": ["criteria"], "properties": {"criteria": {
                 "type": "object", "minProperties": 1, "maxProperties": 255}}}},
            {"if": {"properties": {"type": {"const": "score"}}},
             "then": {"required": ["criteria"], "properties": {"criteria": {
                 "type": "array", "minItems": 1, "maxItems": 255}}}},
            {"if": {"properties": {"type": {"const": "noul"}}},
             "then": {"properties": {"criteria": {"type": ["object", "null"]}}}},
        ],
    },
}


def validate_questions(questions):
    try:
        json.dumps(questions, allow_nan=False)
        error = next(Draft202012Validator(QUESTION_SCHEMA).iter_errors(questions), None)
    except (ValueError, RecursionError):
        error = True
    if error is not None:
        raise ServiceError("invalid_request", "Questions must contain typed choice, score, or noul decisions with valid criteria.", 400,
                           details={"stage": "request", "field": "questions"})


def _valid_response(data, questions):
    if not isinstance(data, dict) or not isinstance(data.get("model"), str):
        return False
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        return False
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            return False
        value = answer.get(kind)
        if kind == "choice":
            if not isinstance(value, str) or value not in question["criteria"]:
                return False
        elif type(value) not in {int, float} or not math.isfinite(value):
            return False
        elif not 0 <= value <= (1 if kind == "noul" else len(question["criteria"]) - 1):
            return False
    return True


def infer(pages, questions, settings, deadline, debug_log=None, state=None, client=None):
    url = settings.systemone_base_url + "/v1/systemone"
    body = {"model": settings.systemone_model, "state": state if state is not None else {"pages": pages}, "questions": questions}
    headers = {"Authorization": f"Bearer {settings.systemone_api_key}"} if settings.systemone_api_key else {}
    remaining = min(settings.systemone_timeout, deadline - time.monotonic())
    if remaining <= 0:
        raise ServiceError("timeout", "Request exceeded its time limit.", 504,
                           details={"stage": "systemone_request", "timeout_seconds": settings.timeout})
    started = time.monotonic()

    def log(event, **fields):
        if debug_log:
            try:
                debug_log.emit("systemone." + event, **fields)
            except Exception:
                pass

    log("request", method="POST", url=url, headers=headers, body=body, timeout_seconds=round(remaining, 3))
    try:
        from contextlib import nullcontext
        with (nullcontext(client) if client is not None else httpx.Client(timeout=remaining, trust_env=False)) as connection:
            with connection.stream("POST", url, headers=headers, json=body, timeout=remaining) as response:
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 1024 * 1024:
                        log("response", status=response.status_code, incomplete=True, reason="provider_response_limit_exceeded")
                        raise ServiceError("provider_error", "SystemOne response exceeds the limit.", 502,
                                           details={"stage": "systemone_response", "maximum": 1024 * 1024})
                log("response", status=response.status_code, headers=dict(response.headers),
                    body=body_for_log(raw, response.headers.get("content-type", "")),
                    duration_ms=round((time.monotonic() - started) * 1000, 3))
                if response.status_code >= 300:
                    raise ServiceError("provider_error", "SystemOne service rejected the request.", 502,
                                       details={"stage": "systemone_request", "provider_http_status": response.status_code,
                                                "hint": "Check SYSTEMONE_API_KEY, SYSTEMONE_MODEL, service readiness, and question/context limits."})
    except httpx.TimeoutException as exc:
        log("error", causes=exception_causes(exc))
        raise ServiceError("timeout", "SystemOne service timed out.", 504,
                           details={"stage": "systemone_request", "exception_type": type(exc).__name__}) from None
    except httpx.HTTPError as exc:
        log("error", causes=exception_causes(exc))
        raise ServiceError("provider_error", "SystemOne service could not be reached.", 502,
                           details={"stage": "systemone_request", "exception_type": type(exc).__name__,
                                    "hint": "Check SYSTEMONE_IP, SYSTEMONE_PORT, and network reachability from the processor container."}) from None
    try:
        data = load_json(raw)
        json.dumps(data, allow_nan=False)
        if not _valid_response(data, questions):
            raise ValueError()
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise ServiceError("invalid_inference", "SystemOne returned an invalid decision response.", 502,
                           details={"stage": "systemone_response"}) from None
    return data
