# Reproducibility and identity policies

A regression gate is meaningful only when it proves what stayed constant. Schema
v3 therefore keeps separate identities instead of overloading one hash.

## Workload

- `workload_ids_fingerprint`: canonical query-ID set; row order is irrelevant.
- `workload_content_fingerprint`: query ID plus normalized source text.
- `workload_revision`: caller revision for precomputed workloads without text.

The default `same-content` policy accepts row reordering and new query vectors
from another embedder, but rejects changed query text. If source text is absent,
provide `--workload-revision` when probing or explicitly use
`--workload-policy same-ids`.

## Corpus

- `corpus_set_fingerprint`: canonical logical ID set.
- `corpus_order_fingerprint`: physical adapter order.
- `corpus_content_fingerprint`: chunk ID plus normalized source text.
- `corpus_revision`: caller revision for stores that cannot return content.
- `index_mapping_fingerprint`: FAISS index/ordered-sidecar binding metadata.

pgvector returns IDs using `ORDER BY id`; Qdrant scroll order is irrelevant to
the logical set hash. FAISS keeps ordered mapping integrity separately from the
logical corpus identity.

## Provenance

`ProbeMetadata` automatically records Python/platform, adapter and dependency
version, safe adapter configuration, distance metric, normalization, top-k,
search parameters, embedder/model revision, corpus/workload revisions, run ID,
and source commit when available.

Credential-shaped metadata keys are rejected. Database URLs, passwords, tokens,
and API keys are never serialized. Endpoint hostnames are omitted by default.

## Artifacts

A full probe is an integrity-checked raw source for `load_probe`, `diff`, and
`gate`. A summary is a bounded presentation artifact: it omits raw hits,
frequencies, query IDs, and dark IDs, and samples the Lorenz curve to at most 512
quantile points by default. `load_probe()` deliberately rejects summaries.

Run the fixed-size performance check with:

```bash
python -m scripts.benchmark_quality --size 1000000 --assert-targets
```

Targets are a summary below 1 MiB and `build_report(detail="summary")` below
three seconds on the project benchmark runner.
