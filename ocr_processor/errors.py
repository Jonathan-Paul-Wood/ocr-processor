from contextlib import contextmanager
from pathlib import Path
import traceback


class ServiceError(Exception):
    def __init__(self, code, message, status=422, *, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = dict(details or {})

    def add_context(self, **context):
        for key, value in context.items():
            self.details.setdefault(key, value)
        return self

    def payload(self):
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def unexpected_error(exc, stage):
    # Exception messages and traceback source/locals can contain document text or secrets.
    frames = traceback.extract_tb(exc.__traceback__)
    locations = [
        {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in frames if Path(frame.filename).parent.name == "ocr_processor"
    ][-5:]
    details = {"stage": stage, "exception_type": type(exc).__name__, "locations": locations}
    details.update(getattr(exc, "_ocr_context", {}))
    if isinstance(exc, OSError):
        details["errno"] = exc.errno
    return ServiceError("processing_failed", "Document processing failed.", 500, details=details)


@contextmanager
def error_context(**context):
    try:
        yield
    except ServiceError as exc:
        exc.add_context(**context)
        raise
    except Exception as exc:
        exc._ocr_context = {**context, **getattr(exc, "_ocr_context", {})}
        raise

