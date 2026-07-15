# Changelog

## 0.2.0 — 2026-07-16

- Add schema-v3 semantic workload and corpus identities with strict content/revision gate policies.
- Record typed, credential-safe adapter/embedder/runtime provenance automatically.
- Make summary artifacts bounded: no raw IDs/hits/frequencies and at most 512 Lorenz points by default.
- Reuse a validated frequency calculation context and add a reproducible 1M benchmark.
- Define deterministic score tie-breaking and bind FAISS manifests to indexes with SHA-256.
- Include CLI in coverage, raise the threshold to 80%, and add CLI end-to-end tests.
- Add Dependabot, CodeQL, dependency review, optional-extra audits, pyright, and broader Ruff rules.
- Ship complete test fixtures in the sdist and harden vector/qrels/numeric validation.

## 0.1.1 — 2026-07-13

- Add integrity-checked baseline schema v2 with query IDs and stable fingerprints.
- Align diffs by query ID and validate corpus compatibility.
- Correct qrels grade semantics and report micro/macro recall.
- Centralize input validation and harden dashboard, CI, and release workflows.

## 0.1.0

- Initial exposure metrics, adapters, dashboard, qrels validation, and CI gate.
