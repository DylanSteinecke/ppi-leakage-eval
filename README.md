# PPI Leakage
Testing how robust models and datasets for protein-protein interactions are to leakage.

## Installation

Activate the project environment and install the package in editable mode:

```bash
conda activate ppi
python -m pip install -e .
```

Install the optional Torch backend with:

```bash
python -m pip install -e '.[torch]'
```

This provides `ppi-train`, `ppi-prepare`, `ppi-aggregate`, and
`ppi-make-toy-data`.

The installable implementation lives under `src/ppi_benchmark/`. Command-line
orchestration is in `cli/`, dataset preparation and source-specific loaders are
in `datasets/`, model-framework adapters are in `backends/`, and reusable
benchmark modules are at the package root. The runner uses a backend-neutral
fit/predict contract. Classical classifiers retain the existing sklearn
factory and scoring path, while `torch_mlp` exercises the batched Torch path.

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
bash scripts/run_toy_ppi_example.sh
```

The toy runner defaults to the `exhaustive` profile. Set
`BENCHMARK_PROFILE=laptop` for the smaller grid, or `GENERATE_TOY_DATA=0` to
reuse the existing files under `processed/`.

## Run the yeast BioGRID example

The local UniProt FASTA uses headers such as `sp|P04387|GAL80_YEAST`, while
BioGRID stores the matching accession as `P04387`. The example runner prepares
the current local files, samples negatives within the yeast taxon, and runs a
laptop-sized sparse benchmark. By default, the `laptop` profile selects a
deterministic, label-stratified cohort of at most 10,000 pairs before creating
each split:

```bash
conda activate ppi
BENCHMARK_PROFILE=laptop bash scripts/run_yeast_biogrid_ppi_example.sh
```

The runner profiles are:

- `laptop`: TF-IDF and count features with SGD logistic regression, at most
  10,000 pairs, and 25 C-split trials.
- `exhaustive`: TF-IDF, BM25, count, binary, and their combined feature set with
  logistic regression, linear SVM, and SGD logistic regression; the full
  cohort and 100 C-split trials are used by default.

Both profiles include the constant baselines and random/C1/C2/C3 splits. Set
`MAX_PAIRS` or `N_SPLIT_TRIALS` explicitly to override a profile default. An
explicitly empty `MAX_PAIRS` uses the entire eligible cohort. `SAMPLING_SEED`
controls cohort selection independently of the data-split seed. `SPLIT_SEEDS`
accepts a quoted, space-separated list and creates a separate run directory for
every split strategy/seed combination. `MODEL_SEED` controls the first model
seed; `NUM_RERUNS` advances model seeds within each fixed split. For example:

```bash
SPLIT_SEEDS="0 1 2 3 4" MODEL_SEED=100 NUM_RERUNS=3 \
    PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh
```

The grid defaults to a genuine 80/10/10 train/validation/test split. Test
remains held out unless `--eval-test-set` is supplied. After aggregation, the
current grid invocation is summarized in
`benchmark_train_val_f1.png`: each split strategy has its own panel, marker
positions show absolute F1, and the train-to-validation arrow shows the
generalization gap. Strategy panels are stacked vertically on one shared F1
scale, so better or worse performance can be compared by horizontal position.
To reuse already prepared yeast files, add
`PREPARE_YEAST_DATA=0` before the command.

For example, run the exhaustive model grid on a bounded cohort with:

```bash
BENCHMARK_PROFILE=exhaustive MAX_PAIRS=50000 PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh
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
    --val-size 0.1 \
    --features tfidf \
    --classifier logistic \
    --max-iter 100 \
    --run-dir results/yeast_c3_sample
```

Use `--sample-fraction` instead of `--max-pairs` for a proportional cohort.
The selected source rows and deterministic ranks are written to
`sampling/selected_examples.csv`; audit details are stored under `sampling` in
`splits/split_metadata.json`.

## Performance reporting

Every `ppi-train` invocation appends one record to `performance.jsonl`. Each
record contains stage durations, process peak resident memory, feature-matrix
shape/density/storage statistics, and per-model fit/evaluation timings and
solver iteration counts. Per-run metric CSVs also include `fit_seconds`,
`evaluation_seconds`, and `solver_iterations` columns.

## Validation-selected thresholds

Learned models select their binary decision threshold by maximizing F1 on the
validation split. That one threshold is then used for train, validation, and,
when explicitly enabled, test metrics; test labels are never consulted during
selection. Baselines retain their fixed behavior. Use
`--threshold-selection fixed` to retain the backend default of 0.5 for
probabilities or 0.0 for decision scores.

The selected/default thresholds, selection strategy, and validation F1 are
recorded in every metric row and in each `performance.jsonl` model record;
test prediction rows also carry the selected threshold and strategy. Runs
without a validation split fall back to the fixed backend threshold.

## Batched Torch MLP

`torch_mlp` is a small one-hidden-layer classifier over the same sparse k-mer
pair matrices used by the sklearn models. It slices sparse rows by batch and
densifies only the current batch for training, validation, and prediction. The
default 80/10/10 split therefore supports validation-AUPRC early stopping while
keeping test held out unless `--eval-test-set` is requested.

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --max-pairs 10000 \
    --features tfidf \
    --classifier torch_mlp \
    --torch-max-epochs 50 \
    --torch-batch-size 256 \
    --torch-hidden-dim 64 \
    --torch-patience 5 \
    --run-dir results/yeast_torch_mlp
```

Epoch losses, validation AUPRC, and batch counts are written to
`training_history.csv`. The best-AUPRC weights are restored before benchmark
evaluation. Each run writes both `*.best.pt` (inference weights) and
`*.last.pt` (model, optimizer, RNG, history, and early-stopping state) below
`checkpoints/<execution-id>/`. Per-model performance records report both paths,
the device, completed epochs, optimizer steps, best epoch, early-stopping
status, and the largest batch ever densified.

Resume an interrupted run from its last checkpoint by keeping the data, split,
model seed, and Torch configuration unchanged while increasing the total epoch
target:

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --features tfidf \
    --classifier torch_mlp \
    --split-seed 0 \
    --model-seed 0 \
    --torch-max-epochs 100 \
    --torch-resume-from results/yeast_torch_mlp/checkpoints/<execution-id>/tfidf__torch_mlp__run_1.last.pt \
    --run-dir results/yeast_torch_mlp_resumed
```

`--torch-max-epochs` is the total target, not the number of additional epochs.
Resume validates the model seed, training configuration, and exact sparse
train/validation data signatures before restoring optimizer and RNG state.

To add the model to every feature set in an existing laptop or exhaustive
benchmark grid without changing that profile's defaults:

```bash
INCLUDE_TORCH_MLP=1 PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh \
    --torch-max-epochs 30 --torch-batch-size 256
```

## Leakage-aware splits

Use `--split-strategy c1`, `c2`, or `c3` to select a standard PPI
generalization regime:

- `c1` holds out edges while keeping every held-out protein represented in
  train.
- `c2` gives each held-out edge one train group and one held-out group.
- `c3` separates train and held-out protein groups completely.

C2 and C3 discard edges that do not match the selected regime. For three-way
splits, validation and test receive distinct novel protein groups under C2,
while all three splits receive mutually disjoint protein groups under C3. The
runner evaluates 100 deterministic candidate group assignments by default;
adjust this with `--n-split-trials`.
Detailed retention, class-balance, degree, overlap, and invariant diagnostics
are written under `split_audit` in `split_metadata.json`.

For homology-aware C2/C3 splits, provide a precomputed sequence-cluster CSV:

```csv
protein_id,cluster_id
P04387,cluster_0001
P12345,cluster_0001
Q99999,cluster_0002
```

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --sequence-clusters processed/biogrid_yeast_physical/sequence_clusters.csv \
    --split-strategy c3 \
    --run-dir results/yeast_homology_c3
```

Every eligible cohort protein must have exactly one mapping; coverage is
validated before optional cohort sampling. C2/C3 then treat each cluster as an
indivisible group. The normalized mapping is copied to
`splits/sequence_cluster_assignments.csv`, and its path, hash, counts, and split
audit are recorded in `split_metadata.json`. The pipeline consumes cluster
assignments but does not run a clustering tool itself.
