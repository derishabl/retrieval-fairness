# Contributing

1. Create a virtual environment.
2. Install development dependencies: `pip install -e '.[dev,faiss]'`.
3. Run `ruff check .` and `ruff format --check .`.
4. Run `pytest -W error --cov=retrieval_fairness --cov-fail-under=75`.
5. Add a regression test for behavioral changes and open a pull request.

Build checks: `python -m build && twine check dist/*`.
