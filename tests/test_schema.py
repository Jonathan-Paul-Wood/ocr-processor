import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from ocr_processor.errors import ServiceError
from ocr_processor.schema import load_json, validate_schema


def test_example_schema_and_dates():
    schema = json.loads((Path(__file__).parents[1] / "examples/transactions.schema.json").read_text())
    validator = validate_schema(schema)
    validator.validate({"transactions": []})
    transaction = {name: None for name in schema["properties"]["transactions"]["items"]["required"]}
    transaction.update(date="2026-02-30", evidence={"document": 1, "page": 1, "quote": "TOTAL 12.34"})
    with pytest.raises(ValidationError):
        validator.validate({"transactions": [transaction]})


@pytest.mark.parametrize("schema", [True, {}, {"type": "array"}, {"type": "object", "$ref": "file:///etc/passwd"},
                                     {"type": "object", "properties": {"x": {"type": "bogus"}}}])
def test_invalid_schema(schema):
    with pytest.raises(ServiceError):
        validate_schema(schema)


def test_nonfinite_json_rejected():
    with pytest.raises(ValueError):
        load_json('{"amount": NaN}')

