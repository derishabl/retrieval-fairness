# retrieval-fairness

**Exposure audit for vector search / RAG.** Shows what share of your
vector corpus is actually reachable by queries and what share is never
retrieved (**dark matter** — the antihub inventory of your index);
measures exposure concentration (Gini), hub capture, and regression
diff when you change the embedder or chunking. Think of it as a
health/coverage report for your *index*, not your pipeline.

> Status: early development. This is **packaging novelty** (the Gini /
> retrievability metrics are honestly borrowed from IR-fairness /
> T-Retrievability research), not a from-scratch invention.

## Why your index has dark matter (it's not a bug in your code)

Hubs and never-retrieved chunks are a **structural property of
high-dimensional nearest-neighbor search**, not a symptom of a bad
embedder. Radovanović, Nanopoulos & Ivanović (JMLR 2010, [“Hubs in
Space”](https://www.jmlr.org/papers/v11/radovanovic10a.html)) showed
that as dimensionality grows, the distribution of k-occurrences becomes
strongly right-skewed: a few points (hubs) appear in a disproportionate
share of neighbor lists, while others (antihubs) appear in almost none.
Modern ANN indexes (HNSW) amplify the effect — hub nodes are the
highway entry points that make the graph navigable. Left unmeasured,
this silently collapses the *effective* size of your corpus. This tool
makes it measurable — and gateable in CI.

## Installation

```bash
pip install retrieval-fairness
```

Optional store adapters are extras (installed only if you use them):

```bash
pip install 'retrieval-fairness[faiss]'       # FAISS
pip install 'retrieval-fairness[pgvector]'    # pgvector (PostgreSQL)
pip install 'retrieval-fairness[qdrant]'      # Qdrant
pip install 'retrieval-fairness[models]'     # sentence-transformers embedder
pip install 'retrieval-fairness[fastembed]'   # fastembed (BGE) embedder
```

From source (development):

```bash
pip install -e '.[dev]'
```

## Quick start

```bash
retrieval-fairness demo --top-k 5          # demo on a synthetic corpus
retrieval-fairness demo-diff --top-k 5     # regression diff for an embedder migration
```

## Usage

### Run against real queries

```bash
# corpus.jsonl: {"id": "...", "text": "...", "vector": [...]}
# queries.jsonl: {"id": "...", "text": "...", "vector": [...]}
retrieval-fairness probe --corpus corpus.jsonl --queries queries.jsonl \
    --top-k 10 --json report.json --html dashboard.html
```

### No query logs: antihub self-query audit (synthetic queries)

For each chunk, generate the query that *should* retrieve it (its own
top TF-IDF terms) and check whether it actually surfaces. A chunk that
cannot be found even by a query aimed at it is invisible from any
reasonable query direction — dark matter from day one:

```bash
retrieval-fairness synth --corpus corpus.jsonl --top-k 10 --html dashboard.html
```

### Regression diff (embedder/chunking change)

```bash
retrieval-fairness diff --baseline before.json --candidate after.json
# For precomputed inputs without source text, opt into ID-only comparison:
retrieval-fairness diff --baseline before.json --candidate after.json \
    --workload-policy same-ids --corpus-policy same-ids
# For an intentional rechunking migration:
retrieval-fairness diff --baseline before.json --candidate after.json \
    --corpus-policy allow-change
```

Schema v3 separates logical ID sets, physical order, semantic content/revision,
and FAISS index mapping identities. `same-content` is the CLI/gate default:
changing query or chunk text under the same ID is rejected, while reordering
rows or changing embedder vectors is allowed. Precomputed workloads/stores
without source text must supply `--workload-revision` / `--corpus-revision` at
probe time or explicitly opt into `same-ids`. Legacy schema v1/v2 baselines
remain readable.

Every full baseline also records typed, credential-safe provenance: Python and
adapter versions, metric/normalization, search parameters, top-k, model
revision, and caller run/commit IDs. Database URLs and API keys are rejected
from metadata and never serialized. Reports are always rebuilt from raw hits
and frequencies on load; saved metrics are not a source of truth.

### CI gate

```bash
retrieval-fairness gate --baseline v1.json --candidate new.json --strict \
    --max-coverage-drop 0.05 --max-dark-matter-rise 0.05
# exit 1 in strict mode if coverage dropped > 5 pp -> CI blocks the deploy
```

### Compact summary artifact

```bash
retrieval-fairness probe --corpus corpus.jsonl --queries queries.jsonl \
    --summary-json summary.json --max-lorenz-points 512
```

A summary contains exact metrics/counts and a deterministic quantile-sampled
Lorenz curve, but no raw hits, frequencies, full query IDs, or dark IDs by
default. It is intentionally rejected by `load_probe()` and cannot be used as
a regression source. The typical 1M-corpus summary stays below 1 MiB; rerun the
benchmark with `python -m scripts.benchmark_quality --assert-targets`.

### Cross-check dark matter against qrels ("lost gold")

If you have relevance judgments (qrels), cross-check which dark-matter
chunks are actually relevant — the corpus contains material the
retriever never surfaces. No competitor exposure tool ships this:

```bash
retrieval-fairness qrels --probe report.json --qrels qrels.json \
    --min-relevance-grade 1 --json qrels_report.json
# --queries is only required for a legacy schema-v1 probe
```

A qrels pair is relevant only when `grade >= --min-relevance-grade` (default
1), so zero and negative judgments are ignored. The output reports **micro
recall@k** over all relevant query/document pairs and **macro recall@k** over
queries that have at least one relevant in-corpus document. `recall_at_k` is
a read-only compatibility alias for micro recall in JSON and Python.

## Metrics

| Metric | What it shows |
|---|---|
| Coverage % | share of the corpus retrieved at least once |
| Of reachable ceiling % | coverage as a share of what the workload can physically reach (n_queries × top_k); distinguishes a bad retriever from a small workload |
| Dark matter % | share NEVER retrieved |
| Gini | exposure concentration (0 = uniform, 1 = all in one) |
| Hub capture top5/10 | share of exposure captured by top-N hubs |
| Lorenz curve | inequality visualization |
| Per-query overlap | result stability across a migration |
| Lost gold / micro & macro Recall@k | positive-relevance dark-matter chunks and qrels retrieval quality |

## How it works

- `retrieval_fairness/types.py` — the `VectorStore` contract (Protocol).
  Any store is bridged to it via an adapter (InMemory, FAISS, pgvector,
  Qdrant today; Pinecone/Weaviate on the roadmap).
- `adapters/inmemory.py` — `InMemoryVectorStore` (cosine, numpy) for
  dev/tests/demos; `adapters/{faiss,pgvector,qdrant}.py` — store adapters.
- `metrics.py` — coverage, gini, lorenz, hub_capture, FairnessReport.
- `probe.py` — run a workload → retrieval frequency → report.
- `diff.py` — regression diff between two runs.
- `gate.py` — CI gate with configurable rules.
- `synth.py` — antihub self-query audit (synthetic queries from the corpus).
- `qrels.py` — dark-matter vs qrels cross-check ("lost gold").
- `dashboard.py` — self-contained HTML report (Lorenz, histogram, PCA map).
- `embedders.py` — Embedder contract (TF-IDF / sentence-transformers / fastembed).

Real-scale case study (BEIR NQ, ~50% dark matter, lexical→dense
regression diff): `docs/case_study_nq.md`. Store adapters: `docs/adapters.md`.
Comparison with related work: `docs/comparison.md`. Identity, provenance, and
artifact contracts: `docs/reproducibility.md`.

## Tests

```bash
pytest tests/ -q
```

## License

MIT
