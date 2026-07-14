# Repository Guidelines

## Project Structure & Module Organization

The installable package is under `src/ppi_benchmark/`. Put command-line entry
points in `cli/`, source-specific loading in `datasets/`, framework adapters in
`backends/`, PPI/PTM behavior in `tasks/`, and reusable encoder code in
`protein_encoders/`. Shared splitting, evaluation, features, and reporting live
at the package root. Tests are in `tests/` and generally mirror these modules.
Use `scripts/` for thin runnable workflows, `configs/` for versioned benchmark
settings, and `docs/` for connected-change checklists. Generated datasets,
embeddings, checkpoints, and benchmark outputs belong in ignored `processed/`
or `results/` paths, not source directories.

## Build, Test, and Development Commands

Always use the project’s `ppi` conda environment:

```bash
conda activate ppi
python -m pip install -e '.[dev]'
python -m pytest -q -m 'not slow'
python -m pytest -q
ruff check src tests
BENCHMARK_PROFILE=laptop bash scripts/run_toy_ppi_example.sh
```

Install `.[torch]` only for Torch backends and `.[plm]` for Hugging Face PLM
work. The `not slow` selection is the fast inner loop; run the unfiltered suite
before review. `python -m pytest -q tests/test_sampling.py` runs one test file.
The laptop toy workflow is the preferred end-to-end smoke test.

## Coding Style & Naming Conventions

Target Python 3.11, use four-space indentation, and keep Ruff clean. Use
`snake_case` for functions, modules, and variables; `PascalCase` for classes;
and `UPPER_CASE` for constants. Add type hints and concise docstrings to public
contracts. Keep task-specific collation, heads, losses, predictions, and metrics
inside `tasks/`; shared trainers and backends must not assume PPI columns.
Prefer deterministic seeds, stable paths, batched computation, and cached
expensive features.

## Testing Guidelines

Pytest discovers `tests/test_*.py`; name tests `test_<observable_behavior>`.
Every behavior change needs a focused regression test. Changes to runners,
splits, caching, or checkpoints should also test deterministic artifacts and
data signatures. Leakage-sensitive changes must verify train/validation/test
separation explicitly. There is no numeric coverage gate, but the full suite
must pass before review.

## Commit & Pull Request Guidelines

History uses short, direct subjects; prefer an imperative message such as
`Add PTM residue alignment checks` and keep each commit focused. Pull requests
should explain the behavior and scientific/leakage implications, list commands
run, and link relevant issues. Include sample CLI output or plots when output
formats change. Do not commit credentials, large biological archives, embedding
caches, checkpoints, or generated benchmark results.
