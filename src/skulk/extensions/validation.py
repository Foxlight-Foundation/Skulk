# pyright: reportAny=false, reportUnknownMemberType=false
"""Bounded JSON Schema validation for capability payloads (Phase 2b).

Call payloads and results are validated against the schemas a provider
published in its descriptor. Validation is **bounded**: schemas are treated as
self-contained documents and remote ``$ref`` resolution is never performed, so
a malicious or broken schema cannot make a node fetch arbitrary URLs or hang on
network I/O mid-call (a #510 requirement). An unresolvable reference simply
fails validation with a clear message.
"""

from __future__ import annotations

from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

# referencing's default registry performs NO network retrieval: an unknown or
# remote $ref raises Unresolvable instead of fetching. Draft202012Validator
# uses that default, which is exactly the bounded behavior required; this
# module exists so every call-path validation shares one guarded entry point.


def validate_against_schema(
    payload: object, schema: dict[str, object], *, what: str
) -> str | None:
    """Validate ``payload`` against ``schema``; return an error string or ``None``.

    Never raises: a broken schema, an unresolvable reference, or a failing
    payload all come back as a message so the call path can answer with a
    typed error instead of a 500.

    Args:
        payload: The JSON-shaped object to validate.
        schema: The JSON Schema (2020-12) to validate against.
        what: Label used in the error message (``"payload"`` / ``"result"``).

    Returns:
        ``None`` when valid; a human-readable failure message otherwise.
    """
    try:
        validator = Draft202012Validator(schema)
        error = next(validator.iter_errors(cast(Any, payload)), None)
    except SchemaError as exc:
        return f"{what} schema is not a valid JSON Schema: {exc.message}"
    except Exception as exc:  # noqa: BLE001 - unresolvable refs etc. must not raise
        return f"{what} schema could not be evaluated: {exc}"
    if error is not None:
        path = "/".join(str(part) for part in error.absolute_path) or "(root)"
        return f"{what} failed schema validation at {path}: {error.message}"
    return None
