import json

from jsonschema import Draft202012Validator, FormatChecker, SchemaError

from .errors import ServiceError


def load_json(value):
    def reject_constant(_):
        raise ValueError("Non-finite JSON number")

    return json.loads(value, parse_constant=reject_constant)


def validate_schema(schema):
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ServiceError("invalid_schema", "JSON Schema must have an object root.", 400)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise ServiceError("invalid_schema", "Only local schema references are allowed.", 400)
                if key in {"$id", "$dynamicAnchor"}:
                    raise ServiceError("invalid_schema", "Schema IDs and dynamic anchors are unsupported.", 400)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    if schema.get("$schema", "https://json-schema.org/draft/2020-12/schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ServiceError("invalid_schema", "Use JSON Schema draft 2020-12.", 400)
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, RecursionError):
        raise ServiceError("invalid_schema", "Invalid JSON Schema.", 400) from None
    return Draft202012Validator(schema, format_checker=FormatChecker())

