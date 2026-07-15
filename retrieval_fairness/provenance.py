"""Typed, credential-safe provenance for probe artifacts."""

from __future__ import annotations

import os
import platform as platform_module
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "credential",
    "database_url",
    "password",
    "secret",
    "token",
)


def _contains_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def sanitize_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe metadata while rejecting credential-shaped keys.

    Arbitrary caller metadata remains supported, but secrets are rejected rather
    than silently copied into a long-lived baseline artifact.
    """

    def clean(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                if _contains_sensitive_key(key):
                    raise ValueError(f"metadata field {path + key!r} may contain credentials")
                output[key] = clean(raw_value, f"{path}{key}.")
            return output
        if isinstance(item, (list, tuple)):
            return [clean(child, path) for child in item]
        raise ValueError(f"metadata value at {path.rstrip('.')} is not JSON-safe: {type(item).__name__}")

    return clean(value, "metadata.")


@dataclass(frozen=True)
class ProbeMetadata:
    """Reproducibility metadata with an explicit, secret-free field set."""

    adapter: str
    top_k: int
    python_version: str = field(default_factory=platform_module.python_version)
    platform: str = field(default_factory=platform_module.platform)
    adapter_version: str | None = None
    adapter_config: Mapping[str, Any] = field(default_factory=dict)
    embedder: str | None = None
    embedder_revision: str | None = None
    distance_metric: str | None = None
    normalized: bool | None = None
    search_params: Mapping[str, Any] = field(default_factory=dict)
    corpus_revision: str | None = None
    workload_revision: str | None = None
    run_id: str | None = None
    git_commit: str | None = None

    @classmethod
    def for_run(
        cls,
        store: object,
        *,
        top_k: int,
        embedder: str | None = None,
        embedder_revision: str | None = None,
        corpus_revision: str | None = None,
        workload_revision: str | None = None,
        run_id: str | None = None,
        git_commit: str | None = None,
    ) -> ProbeMetadata:
        provider = getattr(store, "provenance_metadata", None)
        supplied = provider() if callable(provider) else {}
        if not isinstance(supplied, Mapping):
            raise ValueError("adapter provenance_metadata() must return a mapping")
        safe = sanitize_metadata(supplied)
        return cls(
            adapter=str(safe.get("adapter") or type(store).__name__),
            adapter_version=_optional_str(safe.get("adapter_version")),
            adapter_config=_mapping(safe.get("adapter_config")),
            distance_metric=_optional_str(safe.get("distance_metric")),
            normalized=_optional_bool(safe.get("normalized")),
            search_params=_mapping(safe.get("search_params")),
            top_k=top_k,
            embedder=embedder,
            embedder_revision=embedder_revision,
            corpus_revision=corpus_revision,
            workload_revision=workload_revision,
            run_id=run_id or os.environ.get("GITHUB_RUN_ID"),
            git_commit=git_commit or os.environ.get("GITHUB_SHA"),
        )

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "python_version": self.python_version,
            "python_implementation": platform_module.python_implementation(),
            "platform": self.platform,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "adapter_config": dict(self.adapter_config),
            "embedder": self.embedder,
            "embedder_revision": self.embedder_revision,
            "distance_metric": self.distance_metric,
            "normalized": self.normalized,
            "top_k": self.top_k,
            "search_params": dict(self.search_params),
            "corpus_revision": self.corpus_revision,
            "workload_revision": self.workload_revision,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
        }
        # Keep explicit None values: they make missing provenance visible instead
        # of pretending a field was never considered.
        return sanitize_metadata(values)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def runtime_package_versions() -> dict[str, str]:
    """Small runtime version block used by serializers and diagnostics."""
    return {
        "python": platform_module.python_version(),
        "implementation": platform_module.python_implementation(),
        "executable": sys.implementation.name,
    }
