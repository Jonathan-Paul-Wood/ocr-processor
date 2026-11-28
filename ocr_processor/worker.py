"""One request per subprocess; native parsers never run in the API process."""
import json
import os
from pathlib import Path
import sys
import time

from .config import Settings
from .debug_log import WorkerDebugLog
from .documents import read_documents
from .errors import ServiceError, error_context, unexpected_error
from .extraction import extract
from .schema import load_json, validate_schema
from .systemone import infer, validate_questions
from .transactions import extract_transactions


def process(job, directory, settings):
    debug = WorkerDebugLog(settings, job.get("request_id")) if settings.debug_mode else None
    deadline = time.monotonic() + settings.timeout
    def progress(status, completed=0, total=None):
        target = Path(directory) / 'progress.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status=status, completed=completed, total=total)), encoding='utf-8')
        # Windows readers may briefly prevent replacement of an open destination.
        # Progress is advisory; a status poll must never fail the extraction.
        for attempt in range(20):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if os.name != 'nt':
                    raise
                time.sleep(0.005)
    if job["mode"] == "systemone":
        validate_questions(job["questions"])
    with error_context(stage="schema_validation"):
        validator = validate_schema(job["schema"]) if job["mode"] in {"extract", "transactions"} else None
    with error_context(stage="document_read"):
        pages, warnings = read_documents(job["files"], directory, settings, deadline, job["force_ocr"], progress=progress)
    result = {"pages": pages, "warnings": warnings}
    if job["mode"] == "systemone":
        with error_context(stage="systemone_inference"):
            result["data"] = infer(pages, job["questions"], settings, deadline, debug_log=debug)
        result["warnings"].append("Machine decisions require review; confidence does not establish factual accuracy.")
    if job["mode"] == "transactions":
        result["data"], transaction_warnings = extract_transactions(
            pages, job["context"], validator, settings, deadline, debug_log=debug, progress=progress)
        result["warnings"].extend(transaction_warnings)
    if job["mode"] == "extract":
        with error_context(stage="model_extraction"):
            result["data"] = extract(pages, job["instructions"], job["schema"], validator, settings, deadline, debug_log=debug)
        result["warnings"].append("Machine-extracted data requires review before financial import; schema validation does not establish factual accuracy.")
    return result


def main():
    directory = Path(sys.argv[1])
    try:
        job = load_json((directory / "job.json").read_text(encoding="utf-8"))
        settings = Settings(**load_json(os.environ["OCR_WORKER_SETTINGS"]))
        output = {"status": 200, "body": process(job, directory, settings)}
    except ServiceError as exc:
        output = {"status": exc.status, "body": exc.payload()}
    except Exception as exc:
        error = unexpected_error(exc, "worker")
        output = {"status": error.status, "body": error.payload()}
    (directory / "result.json").write_text(json.dumps(output, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

