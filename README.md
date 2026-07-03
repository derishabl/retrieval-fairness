# retrieval-fairness

**Code coverage for retrieval.** Shows what share of your vector corpus
is actually reachable by queries and what share is never retrieved
(dark matter); measures exposure concentration (Gini), hub capture,
and regression diff when you change the embedder or chunking.

> Status: early development. This is **packaging novelty** (the Gini /
> retrievability metrics are honestly borrowed from IR-fairness /
> T-Retrievability research), not a from-scratch invention. See
> `RETRIEVAL_FAIRNESS_PLAN.md`.

## Installation

```bash
pip install -e .            # or pip install retrieval-fairness
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
# queries.jsonl: {"id": "...", "vector": [...]}
retrieval-fairness probe --corpus corpus.jsonl --queries queries.jsonl \
    --top-k 10 --json report.json --html dashboard.html
```

### No query logs: synthetic queries from the corpus

```bash
retrieval-fairness synth --corpus corpus.jsonl --top-k 10 --html dashboard.html
```

### Regression diff (embedder/chunking change)

```bash
retrieval-fairness diff --baseline before.json --candidate after.json
```

### CI gate

```bash
retrieval-fairness gate --baseline v1.json --candidate new.json --strict \
    --max-coverage-drop 0.05 --max-dark-matter-rise 0.05
# exit 1 in strict mode if coverage dropped > 5 pp -> CI blocks the deploy
```

### Cross-check dark matter against qrels ("lost gold")

If you have relevance judgments (qrels), cross-check which dark-matter
chunks are actually relevant — the corpus contains material the
retriever never surfaces. No competitor exposure tool ships this:

```bash
retrieval-fairness qrels --probe report.json --qrels qrels.json \
    --queries queries.jsonl --json qrels_report.json
# prints: lost gold count, recall@k, dark_relevant_ids
```

## Metrics

| Metric | What it shows |
|---|---|
| Coverage % | share of the corpus retrieved at least once |
| Dark matter % | share NEVER retrieved |
| Gini | exposure concentration (0 = uniform, 1 = all in one) |
| Hub capture top5/10 | share of exposure captured by top-N hubs |
| Lorenz curve | inequality visualization |
| Per-query overlap | result stability across a migration |
| Lost gold / Recall@k | dark-matter chunks that are actually relevant (qrels cross-check) |

## How it works

- `retrieval_fairness/types.py` — the `VectorStore` contract (Protocol).
  Any store (FAISS, Qdrant, Pinecone, pgvector) is bridged to it via
  an adapter.
- `stores.py` — `InMemoryVectorStore` for dev/tests/demos.
- `metrics.py` — coverage, gini, lorenz, hub_capture, FairnessReport.
- `probe.py` — run a workload → retrieval frequency → report.
- `diff.py` — regression diff between two runs.
- `gate.py` — CI gate with configurable rules.
- `synth.py` — synthetic queries generated from the corpus.
- `dashboard.py` — self-contained HTML report (Lorenz, histogram, PCA map).

Real-scale case study (BEIR NQ, ~50% dark matter, lexical→dense
regression diff): `docs/case_study_nq.md`. Store adapters:
`docs/adapters.md`. Comparison with related work: `docs/comparison.md`.

## Tests

```bash
pytest tests/ -q
```

## License

MIT
