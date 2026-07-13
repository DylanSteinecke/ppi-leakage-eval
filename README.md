# PPI Leakage
Testing how robust models and datasets for protein-protein interactions are to leakage.

## Installation

Activate the project environment and install the package in editable mode:

```bash
conda activate ppi
python -m pip install -e .
```

This provides `ppi-train`, `ppi-prepare`, `ppi-aggregate`, and
`ppi-make-toy-data`.

The installable implementation lives under `src/ppi_benchmark/`. Command-line
orchestration is in `cli/`, dataset preparation and source-specific loaders are
in `datasets/`, and reusable benchmark modules are at the package root.

## Tests
Run the lightweight regression suite with:

```bash
python -m pytest
```

For connected-change checklists, see
[`docs/change_checklists.md`](docs/change_checklists.md).

## Run the toy PPI example

The toy runner generates a synthetic dataset, then runs the benchmark grid:

```bash
conda activate ppi
bash scripts/run_toy_ppi_example.sh --no-metrics-plots
```

Set `GENERATE_TOY_DATA=0` to reuse the existing files under `processed/`.

## Run the yeast BioGRID example

The local UniProt FASTA uses headers such as `sp|P04387|GAL80_YEAST`, while
BioGRID stores the matching accession as `P04387`. The example runner prepares
the current local files, samples negatives within the yeast taxon, and runs a
small sparse benchmark:

```bash
conda activate ppi
bash scripts/run_yeast_biogrid_ppi_example.sh --no-metrics-plots
```

Prepared data are written under `processed/biogrid_yeast_physical/` as
`pairs.csv`, `proteins.fasta`, `protein_metadata.csv`, and
`dataset_metadata.json`. The protein sidecar stores one NCBI `taxon_id` per
protein without repeating species data on every interaction row. The benchmark
also accepts it directly with `--protein-metadata`; a sidecar beside canonical
`pairs.csv` is discovered automatically.

The loader reads the large archive in chunks. Sampled negatives are unobserved
protein pairs from taxonomy-pair strata represented by positives, not
experimentally confirmed non-interactions. The archive member includes a
BioGRID release number and must be updated when the `LATEST` download changes.

## Leakage-aware splits

Use `--split-strategy c1`, `c2`, or `c3` to select a standard PPI
generalization regime:

- `c1` holds out edges while keeping every test protein represented in train.
- `c2` gives each test edge one train group and one held-out group.
- `c3` separates train and test protein groups completely.

C2 and C3 discard edges that do not match the selected regime. The runner
evaluates 100 deterministic candidate group assignments by default; adjust
this with `--n-split-trials`. These modes currently require `--val-size 0`.
Detailed retention, class-balance, degree, overlap, and invariant diagnostics
are written under `split_audit` in `split_metadata.json`.
