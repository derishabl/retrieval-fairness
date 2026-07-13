"""Central validation helpers for public inputs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def require_positive_int(value: int, name: str) -> int:
    """Return *value* when it is a positive, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def require_non_negative_int(value: int, name: str) -> int:
    """Return *value* when it is a non-negative, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def validate_vector(vector: Sequence[float], *, name: str, dim: int | None = None) -> None:
    """Validate that a vector is non-empty, finite and optionally has *dim*."""
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise ValueError(f"{name} must be a sequence of numbers")
    if not vector:
        raise ValueError(f"{name} must not be empty")
    if dim is not None and len(vector) != dim:
        raise ValueError(f"{name} dimension {len(vector)} != expected {dim}")
    for index, value in enumerate(vector):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{name}[{index}] must be a finite number, got {value!r}")


def validate_unique_ids(ids: Sequence[str], *, name: str) -> None:
    """Require non-empty string IDs and reject duplicates with examples."""
    bad = [value for value in ids if not isinstance(value, str) or not value]
    if bad:
        raise ValueError(f"{name} must contain non-empty string IDs; invalid examples: {bad[:5]!r}")
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{name} contains duplicate IDs ({len(duplicates)}): {duplicates[:5]!r}")
