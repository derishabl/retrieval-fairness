# Quality milestone 0.2.0 verification

Verified on 2026-07-16 against the post-release 0.1.1 audit.

## Local acceptance

- Ruff expanded rules: pass.
- Pyright basic on core modules: pass, 0 diagnostics.
- `pytest -W error --cov=retrieval_fairness --cov-fail-under=80`:
  127 passed, 3 optional-integration skips, 87.38% coverage.
- Schema v1 and v2 compatibility fixtures: pass.
- Wheel/sdist build and Twine validation: pass.
- Test suite executed from the unpacked sdist: 127 passed, 3 skipped.
- Fresh wheel install, CLI help, demo, and version smoke: pass.
- Base plus `faiss`, `pgvector`, `qdrant`, `models`, and `fastembed`
  dependency resolutions: no known vulnerabilities.
- 1M benchmark: `build_report(detail="summary")` 1.62 s; compact summary
  9,268 bytes and 512 Lorenz points on the audit machine. Timing is runner
  dependent; use `python -m scripts.benchmark_quality --assert-targets`.

## Repository controls

- Dependabot alerts/security updates enabled.
- Dependabot config, dependency review, CodeQL, and optional-extra audit matrix added.
- Immutable `v*` tag ruleset enabled (update/delete denied, no bypass actors).
- PyPI environment accepts only custom tag policy `v*`; admin bypass disabled.
- Solo-maintainer governance uses zero required approvals while retaining strict
  required checks, PR flow, linear history, conversation resolution, and force
  push/deletion denial.

The CodeQL workflow becomes visible in GitHub Security after this branch is
merged and its first analysis completes.

## Owner action still required

The repository currently names `derishabl` as the MIT copyright holder. Legal
holder confirmation cannot be inferred by automation and remains an explicit
owner decision before release.
