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
laptop-sized sparse benchmark. By default, it selects a deterministic,
label-stratified cohort of at most 10,000 pairs before creating each split:

```bash
conda activate ppi
MAX_PAIRS=10000 bash scripts/run_yeast_biogrid_ppi_example.sh \
    --no-metrics-plots
```

Set `MAX_PAIRS` to a different compute budget, or to an empty string to use the
entire eligible cohort. `SAMPLING_SEED` controls cohort selection independently
of the train/test split seed. To reuse already prepared yeast files, add
`PREPARE_YEAST_DATA=0` before the command.

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

## Sample a whole benchmark cohort

`ppi-train` can select a deterministic subset after input validation and before
random or C1/C2/C3 split construction:

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --max-pairs 10000 \
    --sampling-seed 17 \
    --split-strategy c3 \
    --train-size 0.8 \
    --features tfidf \
    --classifier logistic \
    --max-iter 100 \
    --run-dir results/yeast_c3_sample
```

Use `--sample-fraction` instead of `--max-pairs` for a proportional cohort.
The selected source rows and deterministic ranks are written to
`sampling/selected_examples.csv`; audit details are stored under `sampling` in
`splits/split_metadata.json`.

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
