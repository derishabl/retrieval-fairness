"""FAISS adapter with a cryptographically bound row-to-ID manifest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.types import Hit
from retrieval_fairness.validation import validate_unique_ids, validate_vector

_MANIFEST_SCHEMA_VERSION = 2


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


class FaissAdapter(BaseVectorStoreAdapter):
    """Search a FAISS index whose ordered IDs are protected by a manifest."""

    def __init__(
        self,
        index_path: str,
        ids_map_path: str | None = None,
        *,
        allow_legacy_ids_map: bool = False,
    ):
        super().__init__()
        import faiss

        self._faiss = faiss
        self._index_path = str(index_path)
        self._index = faiss.read_index(self._index_path)
        self._metric = self._metric_name(self._index.metric_type)
        self._normalized: bool | None = None
        self._manifest: dict[str, object] | None = None

        if ids_map_path:
            with open(ids_map_path, encoding="utf-8") as source:
                manifest = json.load(source)
            if not isinstance(manifest, dict) or not isinstance(manifest.get("ids"), list):
                raise ValueError("FAISS manifest must be an object containing an ids list")
            if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
                if not allow_legacy_ids_map:
                    raise ValueError(
                        "legacy FAISS ids-map is not bound to the index; rebuild the manifest "
                        "or pass allow_legacy_ids_map=True explicitly"
                    )
            else:
                self._validate_manifest(manifest)
                self._manifest = manifest
                self._normalized = manifest.get("normalized")
            self._ids = list(manifest["ids"])
        else:
            self._ids = [str(index) for index in range(self._index.ntotal)]

        if len(self._ids) != self._index.ntotal:
            raise ValueError(f"ids-map length {len(self._ids)} != index ntotal {self._index.ntotal}")
        if any(not isinstance(chunk_id, str) for chunk_id in self._ids):
            raise ValueError("FAISS ids-map must contain string IDs")
        validate_unique_ids(self._ids, name="FAISS IDs")

    def _metric_name(self, metric_type: int) -> str:
        if metric_type == self._faiss.METRIC_INNER_PRODUCT:
            return "ip"
        if metric_type == self._faiss.METRIC_L2:
            return "l2"
        return f"faiss:{metric_type}"

    def _validate_manifest(self, manifest: dict[str, object]) -> None:
        expected_checksum = manifest.get("index_sha256")
        actual_checksum = _sha256_file(self._index_path)
        if expected_checksum != actual_checksum:
            raise ValueError("FAISS manifest index_sha256 does not match the index file")
        if manifest.get("dimension") != self._index.d:
            raise ValueError(
                f"FAISS manifest dimension {manifest.get('dimension')} != index dimension {self._index.d}"
            )
        if manifest.get("metric") != self._metric:
            raise ValueError(
                f"FAISS manifest metric {manifest.get('metric')!r} != index metric {self._metric!r}"
            )

    @property
    def index_mapping_fingerprint(self) -> str | None:
        if self._manifest is None:
            return None
        digest = hashlib.sha256()
        digest.update(str(self._manifest["index_sha256"]).encode("ascii"))
        for chunk_id in self._ids:
            encoded = chunk_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return f"sha256:{digest.hexdigest()}"

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        validate_vector(query_vec, name="query vector", dim=self._index.d)
        query = np.array([query_vec], dtype="float32")
        count = min(top_k, self._index.ntotal)
        if count == 0:
            return []
        scores, indices = self._index.search(query, count)
        selected = [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
        if self._metric == "l2":
            selected.sort(key=lambda pair: (pair[1], self._ids[pair[0]]))
        else:
            selected.sort(key=lambda pair: (-pair[1], self._ids[pair[0]]))
        return [
            Hit(chunk_id=self._ids[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(selected, start=1)
        ]

    def _list_chunk_ids(self) -> Iterator[str]:
        yield from self._ids

    def provenance_metadata(self) -> dict[str, object]:
        return {
            "adapter": "faiss",
            "adapter_version": getattr(self._faiss, "__version__", None),
            "adapter_config": {
                "dimension": self._index.d,
                "index_sha256": self._manifest.get("index_sha256") if self._manifest else None,
                "index_mapping_fingerprint": self.index_mapping_fingerprint,
            },
            "distance_metric": self._metric,
            "normalized": self._normalized,
            "search_params": {
                "tie_policy": "score_desc_chunk_id_asc",
                "boundary_ties_backend_limited": True,
            },
        }

    @property
    def size(self) -> int:
        return self._index.ntotal


def build_flat_index(
    vectors: list[list[float]],
    ids: list[str],
    index_path: str,
    ids_map_path: str,
    metric: str = "ip",
    *,
    normalized: bool = False,
) -> None:
    """Atomically write an IndexFlat index and checksum-bound manifest."""
    if not vectors:
        raise ValueError("vectors must not be empty")
    if len(vectors) != len(ids):
        raise ValueError(f"len(vectors)={len(vectors)} != len(ids)={len(ids)}")
    validate_unique_ids(ids, name="FAISS IDs")
    if metric not in {"ip", "l2"}:
        raise ValueError(f"unknown metric: {metric}")
    dimension = len(vectors[0])
    for index, vector in enumerate(vectors):
        validate_vector(vector, name=f"vectors[{index}]", dim=dimension)

    index_target = Path(index_path)
    manifest_target = Path(ids_map_path)
    index_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)

    import faiss

    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(dimension) if metric == "ip" else faiss.IndexFlatL2(dimension)
    index.add(matrix)

    index_temp: str | None = None
    manifest_temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{index_target.name}.", dir=index_target.parent, delete=False
        ) as temporary:
            index_temp = temporary.name
        faiss.write_index(index, index_temp)
        manifest = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "ids": ids,
            "index_sha256": _sha256_file(index_temp),
            "dimension": dimension,
            "metric": metric,
            "normalized": normalized,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{manifest_target.name}.",
            dir=manifest_target.parent,
            delete=False,
        ) as temporary:
            manifest_temp = temporary.name
            json.dump(manifest, temporary, ensure_ascii=False, separators=(",", ":"))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(index_temp, index_target)
        index_temp = None
        os.replace(manifest_temp, manifest_target)
        manifest_temp = None
    finally:
        for temporary in (index_temp, manifest_temp):
            if temporary:
                Path(temporary).unlink(missing_ok=True)
