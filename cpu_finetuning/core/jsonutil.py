"""JSON helpers for values that browsers must be able to parse.

Python's ``json`` module writes ``NaN`` and ``Infinity`` by default.  Those
tokens are not valid JSON, so ``response.json()`` fails in the Web UI when a
training metric is missing (for example a probe window with no loss tokens).
Represent non-finite values as JSON ``null`` instead.
"""

from __future__ import annotations

import json
import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dumps(value: Any, **kwargs) -> str:
    """Serialize valid JSON; non-finite floats become ``null``."""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)


def dump(value: Any, fp, **kwargs) -> None:
    """Write valid JSON to a file object; non-finite floats become ``null``."""
    kwargs.setdefault("ensure_ascii", False)
    json.dump(json_safe(value), fp, allow_nan=False, **kwargs)
