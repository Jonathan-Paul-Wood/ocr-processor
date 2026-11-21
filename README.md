# OCR Processor

Independent HTTP service for extracting structured data from PDFs and receipt images.
It accepts documents, caller instructions, and a JSON Schema; it returns validated
JSON plus page-level source text.

## Stack and hardware choice

- **Python 3.12, Flask, Waitress**: small API, one process and four HTTP threads.
- **PDFium (`pypdfium2`)**: read existing PDF text without OCR. Pages with fewer than
  40 text characters are rendered at 250 DPI and sent to OCR. `force_ocr=true`
  handles broken text layers and scans with substantial text headers.
- **Tesseract 5, English**: CPU OCR for scanned PDFs, PNG, JPEG, TIFF and WebP.
  Pages run sequentially with one OpenMP thread. Multi-frame images are supported.
- **A separately configured model endpoint**: interprets instructions, identifies
  fields, suggests categories, and returns JSON constrained by the supplied schema.
  Responses are independently validated with JSON Schema draft 2020-12 and format checks.

Tesseract recognizes text; it does not understand arbitrary extraction instructions
or reliably infer transaction direction and categories by itself. On a server sharing
a few GB of RAM with Hoard Counter, the recommended starting point is local OCR plus
a hosted model for structuring, **if sending statement text to that provider is acceptable**.
There is no default provider, no model download, and no external call until configured.
Only extracted page text and caller instructions/schema go to the configured endpoint,
not original PDFs/images. Original text can still contain sensitive account information.

Compose caps this service at **1 GB RAM and one CPU**, with 256 MB temporary storage.
These are deployment limits, not measured usage guarantees. PDF decoding can exceed
the cap on hostile or unusually complex files; the disposable worker may then fail.
Benchmark representative statements before changing limits. One request runs at a time;
busy callers receive 429 rather than accumulating an unbounded work queue.

Other options:

| Option | When to consider it | Tradeoff |
| --- | --- | --- |
| [PaddleOCR](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html) | Tesseract struggles with receipt detection, rotation, or layout | Supports CPU inference; additional models and inference dependencies need benchmarking on this server |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | Financial text must stay on your infrastructure | OpenAI-compatible schema output; small quantized models may fit, but weights, context cache and CPU latency compete with Hoard Counter; accuracy must be evaluated |
| Hosted structured-output model | General instructions and varied statement formats | No local model RAM; network access, provider charges and data handling terms apply |

References: [Tesseract](https://github.com/tesseract-ocr/tesseract),
[OCR image quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html),
[PDFium Python API](https://pypdfium2.readthedocs.io/en/stable/python_api.html),
[structured output API](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
Complex tables, handwriting, rotated photographs, ambiguous dates and poor scans
still require review. This service uses plain text, not a layout or vision model.

## Run independently with Docker

```sh
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
# Put the generated value in OCR_API_KEY in .env.
# For structured extraction, also set LLM_BASE_URL, LLM_MODEL and any LLM_API_KEY.
docker compose up --build -d --wait
curl http://127.0.0.1:8090/health
```

An example hosted configuration is `LLM_BASE_URL=https://api.openai.com/v1`,
`LLM_MODEL=gpt-4o-mini`, and your provider key in `LLM_API_KEY`. Verify that the
selected provider/model supports Chat Completions `response_format.type=json_schema`
with `strict=true` and your schema keywords. Compatible endpoints vary; unsupported
schemas return a provider error rather than silently falling back to unconstrained text.
There are no automatic retries or repair calls that could hide extra provider charges.

For local llama.cpp, point `LLM_BASE_URL` at its `/v1` endpoint and set `LLM_MODEL`
to the loaded model. It runs separately and has its own RAM budget. A model on the
Docker host needs a reachable host address; `localhost` inside the container refers
to this service. On Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]`
to a local Compose override if using `host.docker.internal`.

The default published port is loopback-only. For other machines, use a private network
and a TLS reverse proxy with the same upload limit and a response timeout over 185
seconds. All document endpoints require `Authorization: Bearer <OCR_API_KEY>`.
The health endpoint is public and checks API liveness, not provider readiness.
Authenticated `/v1/capabilities` reports whether OCR and extraction are configured.
There are no browser cookies or CORS permissions; callers should be backend services.

Install additional Tesseract language packs in a derived image and set `OCR_LANGUAGE`
(for example `eng+spa`). Never pass OCR flags or language paths from untrusted callers.

## API

| Method/path | Request | Result |
| --- | --- | --- |
| `GET /health` | None; no auth | `{"status":"ok"}` |
| `GET /v1/capabilities` | Bearer auth | Configuration flags, formats, limits |
| `POST /v1/text` | Multipart `documents`, optional `force_ocr` | Page text and warnings; works without a model |
| `POST /v1/extract` | Multipart `documents`, `instructions`, `schema`, optional `force_ocr` | Schema-validated `data`, page text and warnings |

Repeat the `documents` part for each upload, in the desired document order. Supply
`schema` as a form **string containing JSON Schema**, not a file upload or example
JSON instance. `instructions` is a form string. `force_ocr` is `true` or `false`.
File contents determine parsing; caller filenames and content types are not trusted.

```sh
curl --fail-with-body http://127.0.0.1:8090/v1/extract \
  -H "Authorization: Bearer $OCR_API_KEY" \
  -F 'documents=@statement.pdf' \
  -F 'documents=@receipt.png' \
  --form-string "instructions=$(cat examples/transactions.instructions.txt)" \
  -F 'schema=<examples/transactions.schema.json'
```

For Windows PowerShell or a Python backend, the provided client avoids shell-specific
multipart quoting: `python examples/client.py statement.pdf receipt.png`.
It reads `OCR_API_KEY` and optional `OCR_URL` from the environment.
`.env` is loaded by Compose, not automatically by Python processes.

Successful extraction has this envelope (the contents of `data` follow your schema):

```json
{
  "data": {"transactions": []},
  "pages": [{"document": 1, "page": 1, "method": "embedded", "text": "Statement text..."}],
  "warnings": ["Machine-extracted data requires review before financial import; schema validation does not establish factual accuracy."]
}
```

Document and page numbers are one-based. `method` is `embedded` or `ocr`. The text
endpoint omits `data`. Blank pages in an otherwise readable document produce warnings;
an entirely unreadable upload fails. There is no partial success or silent truncation.

Schemas must have an object root. Draft 2020-12 is supported locally; use the subset
accepted by your model provider. Only fragment-local references are accepted; external
references, schema IDs, and dynamic anchors are rejected. For strict provider schemas,
use `additionalProperties: false`, make fields required, and allow `null` for unknowns.
Some format checkers ignore unknown format names. Schema validation checks structure
and declared supported formats, not truth, evidence quotes, deduplication, or completeness.

Default limits: 20 MiB **total HTTP request**, five files, 30 total pages, 16 million
pixels per page, 100,000 extracted text characters, 16,000 instruction characters,
32,000 schema characters, and 1 MiB provider response. OCR is limited to 30 seconds
per page; the processing budget is 180 seconds (`REQUEST_TIMEOUT_SECONDS`), with five
seconds of parent-process cleanup allowance. The provider call has a maximum 90-second
network timeout and 8,192 output tokens. Split large statements into smaller requests;
reconcile cross-request duplicates in the caller. Upload reception time is separate
from the processing budget; configure upstream connection/upload timeouts as needed.

Errors are `{"error":{"code":"...","message":"..."}}`. Every response has
`X-Request-ID` and `Cache-Control: no-store`.

| HTTP status | Meaning |
| --- | --- |
| 400 | Invalid fields, empty upload, or invalid schema |
| 401 | Missing/invalid service API key |
| 413 | Request/page/pixel/text limit exceeded |
| 415 | Recognized but unsupported image format |
| 422 | Corrupt/password-protected document, OCR failure, or no readable text |
| 429 | Busy; respect `Retry-After` |
| 500 | Worker crash or unexpected processing failure |
| 502 | Provider failure, refusal, truncated output, or schema-invalid result |
| 503 | OCR/model not configured; inspect capabilities/configuration |
| 504 | OCR, provider, or overall processing timeout |

The service does not save jobs/results, import transactions, or make financial writes.
Uploads, OCR intermediates, instructions, and results use request-scoped temporary
directories removed after completion/failure. In Compose they live in tmpfs. The
service does not log document bodies or provider responses; reverse-proxy logging
and provider retention policies are separate. A hard host/process crash can leave
temporary files until cleanup/restart outside Compose. Treat the API key as access
to all extraction functionality; this version has no tenants, per-client quotas,
separate client keys, asynchronous jobs, or persistent audit history.

## Calling from Hoard Counter

Use the API from Hoard Counter's backend, with the service key stored on that backend.
Pass a transaction schema, allowed category IDs, and known account IDs/masked account
numbers in the caller instructions. The generic service has no account lookup access;
unknown source/destination IDs stay null. The example schema uses category names and
decimal amount strings as a starting point; adapt it to your application's IDs and
currency precision. Keep the result as an editable preview requiring user review;
apply Hoard Counter's own account, currency, balance, duplicate and import validation
before persisting anything. An account's identity and a payee's name are different fields.

For host processes use `http://127.0.0.1:8090`. To call from a Hoard Counter container,
attach its backend and this service to an explicit shared Docker network. For example,
create `document-services`, then use separate Compose overrides:

```yaml
# ocr-processor/compose.network.yaml
services:
  processor:
    networks: [default, documents]
networks:
  documents:
    external: true
    name: document-services
```

```yaml
# Hoard Counter compose.network.yaml
services:
  api:
    networks: [default, documents]
    environment:
      OCR_URL: http://processor:8090
      OCR_API_KEY: ${OCR_API_KEY}
networks:
  documents:
    external: true
    name: document-services
```

Run each with `docker compose -f compose.yaml -f compose.network.yaml up -d`.
These examples only establish connectivity and environment variables; the Hoard
Counter upload UI and caller/import integration are a separate change. No files in
Hoard Counter are changed by this service implementation. Use a response timeout
of at least 200 seconds. Retry 429 with backoff; do not blindly retry provider failures
as an interrupted request may already have incurred provider costs.

## Development and verification

Python 3.11+ and a native Tesseract installation with English trained data are required.
On Debian/Ubuntu, install `tesseract-ocr tesseract-ocr-eng`. On Windows, install a
Tesseract build linked by its official documentation and add it to PATH or set
`TESSERACT_CMD` to its executable's absolute path.

```sh
python -m venv .venv
# Linux/macOS: . .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
export OCR_API_KEY='replace-with-at-least-32-random-characters'
# PowerShell: $env:OCR_API_KEY = '...'
python -m ocr_processor
```

`python -m pytest -q` tests real generated text PDFs and subprocess extraction with
a local HTTP model stub, malformed documents, auth, resource limits, concurrency,
schema validation and failure handling. Native OCR is additionally tested when
Tesseract is installed; otherwise that test is skipped. No paid model is called.
`docker build --target test -t ocr-processor-test .` runs the tests with Tesseract
installed in the Linux image. `docker compose build` builds the production target.
Evaluate real model accuracy separately with redacted statements and receipts before
using the output for imports. Rebuild images regularly for parser/security updates.

