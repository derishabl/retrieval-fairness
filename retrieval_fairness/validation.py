"""Central validation helpers for public inputs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence, Sized
from numbers import Integral, Real
from typing import Any


def require_positive_int(value: int, name: str) -> int:
    """Return *value* when it is a positive, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def require_non_negative_int(value: int, name: str) -> int:
    """Return *value* when it is a non-negative, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def require_integral(value: object, name: str) -> int:
    """Accept an integer-valued real without lossy coercion; reject bool/text."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an integral number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integral number, got {value!r}")
    return int(numeric)


def validate_vector(vector: Sequence[float], *, name: str, dim: int | None = None) -> None:
    """Validate a non-empty finite numeric array-like vector.

    NumPy arrays and other sized iterables are accepted. Strings, booleans and
    numeric-looking strings are deliberately rejected.
    """
    if isinstance(vector, (str, bytes)) or not isinstance(vector, (Iterable, Sized)):
        raise ValueError(f"{name} must be an array-like sequence of numbers")
    try:
        size = len(vector)
    except TypeError as exc:
        raise ValueError(f"{name} must be an array-like sequence of numbers") from exc
    if size == 0:
        raise ValueError(f"{name} must not be empty")
    if dim is not None and size != dim:
        raise ValueError(f"{name} dimension {size} != expected {dim}")
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name}[{index}] must be a finite number, got {value!r}")


def validate_unique_ids(ids: Sequence[str], *, name: str) -> None:
    """Require non-empty string IDs and reject duplicates with examples."""
    bad = [value for value in ids if not isinstance(value, str) or not value]
    if bad:
        raise ValueError(f"{name} must contain non-empty string IDs; invalid examples: {bad[:5]!r}")
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{name} contains duplicate IDs ({len(duplicates)}): {duplicates[:5]!r}")


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Small public-boundary helper for JSON objects."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value
