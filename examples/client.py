"""Run from repository root: python examples/client.py statement.pdf [receipt.png]."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

import httpx


with ExitStack() as stack:
    files = [("documents", (Path(name).name, stack.enter_context(open(name, "rb")))) for name in sys.argv[1:]]
    examples = Path(__file__).parent
    response = httpx.post(
        os.getenv("OCR_URL", "http://127.0.0.1:8081") + "/v1/extract",
        headers={"Authorization": "Bearer " + os.environ["OCR_API_KEY"]},
        files=files,
        data={"instructions": (examples / "transactions.instructions.txt").read_text(),
              "schema": (examples / "transactions.schema.json").read_text()},
        timeout=200,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))

