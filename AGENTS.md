# Repository Guidelines

Goal: Develop a benchmarking framework which will produce results for "models x features x splits x tasks". Models and features will include baselines (e.g., predicts positives), simple models (e.g., TF-IDF features), and SOTA models (e.g., protein language models). Splits will be different leakage-aware splits such as C1/C2/C3, homology-aware splitting, species, and temporal splitting. Tasks will be protein-protein interactions, PTM prediction, and more. Each step in the process (e.g., dataset prepration, splitting, task training and prediction, report generation) should be written in an abstraction that allows for adapters specific to each model/featire/split/task.

## Project Structure & Module Organization

The installable package is under `src/ppi_benchmark/`. Put command-line entry
points in `cli/`, source-specific loading in `datasets/`, framework adapters in
`backends/`, PPI/PTM behavior in `tasks/`, and reusable encoder code in
`protein_encoders/`. Shared split construction lives in `splitting/`, reporting
and plot generation in `reporting/`, and evaluation in `evaluation/`. Shared
feature construction remains in `features.py`. Tests are in `tests/` and
generally mirror these modules.
Generic artifact hashing and locked writers live in `artifact_io.py`; do not
import dataset preparation or reporting modules solely for those primitives.
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

Before modifying split construction, grouping, negative sampling,
canonicalization, or leakage audits, read `docs/split_semantics.md` and the
relevant task, protocol, and grouping sections of
`docs/split_protocol_catalog.md`. Read the entire catalog when a change affects
cross-cutting rules, protocol selection or registration, or multiple protocols.
Read `docs/split_engineering_guide.md` only for split architecture, artifact,
migration, or conformance-test changes.

When adding or changing a split strategy or public protocol, follow the
connected-change checklist in `docs/change_checklists.md` and update the
protocol catalog entry or implementation status as applicable.

Preserve the selected protocol's prediction unit, assignment entity, projection
rules, forbidden overlaps, drop accounting, audit requirements, and semantic
version. Do not claim that a protocol is implemented unless its required
invariants have regression tests.
