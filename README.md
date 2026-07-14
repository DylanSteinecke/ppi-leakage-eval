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

Install frozen protein language-model features with:

```bash
python -m pip install -e '.[plm]'
```

This provides `ppi-train`, `ppi-grid`, `ppi-prepare`, `ppi-aggregate`, and
`ppi-make-toy-data`.

The installable implementation lives under `src/ppi_benchmark/`. Command-line
orchestration is in `cli/`, dataset preparation and source-specific loaders are
in `datasets/`, model-framework adapters are in `backends/`, and reusable
benchmark modules are at the package root. The runner uses a backend-neutral
fit/predict contract. Classical classifiers retain the existing sklearn
factory and scoring path, while `torch_mlp` exercises the batched Torch path.

The training engine is task-neutral: a backend fits and predicts, an evaluation
policy selects the operating point and computes metrics, and a task adapter
maps domain examples to those shared contracts. The PPI adapter owns pair
examples, a collator that deduplicates proteins within a batch, symmetric pair
composition (sum, absolute difference, and product), and PPI-specific output
fields. A future PTM adapter can provide residue/window examples and its own
head while reusing the backend contract, thresholding, checkpointing, metrics,
and runtime reporting. The shared Torch trainer accepts task connectors for
batch iteration, loss, metrics, task/data signatures, and checkpoint
components; it has no PPI or matrix assumptions.

Metric, prediction, training-history, split-metadata, and performance outputs
carry `evaluation_schema_version` and `task`. Prediction rows retain the legacy
PPI columns and also expose the common fields `split`, `example_id`, `target`,
`score`, and `prediction`. The compact `splits/split_assignments.csv` records
every train/validation/test source row; input hashes make those assignments
reconstructable without copying the full dataset three times.

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
accepts a quoted, space-separated list. `MODEL_SEEDS` explicitly lists every
independent model-fit seed. For example:

```bash
SPLIT_SEEDS="0 1 2 3 4" MODEL_SEEDS="100 101 102" \
    PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh
```

Each invocation creates one immutable benchmark root at
`<OUT_DIR>/<RUN_NAME>/`. Every split strategy, split seed, baseline group, and
feature set gets an independent run directory below `runs/`; no grid cell
appends into another cell's files. The resolved grid is recorded in
`benchmark_config.json`, and aggregation writes `benchmark_manifest.csv`,
`benchmark_summary.csv`, and `benchmark_train_val_f1.png` at the benchmark
root. Omit `RUN_NAME` to use a timestamped name.

The grid defaults to a genuine 80/10/10 train/validation/test split. Test
remains held out unless `--eval-test-set` is supplied. After aggregation, the
current grid invocation is summarized in
`benchmark_train_val_f1.png`: each split strategy has its own panel, marker
positions show absolute F1, and the train-to-validation arrow shows the
generalization gap. Strategy panels are stacked vertically on one shared F1
scale, so better or worse performance can be compared by horizontal position.
To reuse already prepared yeast files, add
`PREPARE_YEAST_DATA=0` before the command.

The Python grid runner owns profile expansion and can also be invoked directly:

```bash
ppi-grid \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --protein-metadata processed/biogrid_yeast_physical/protein_metadata.csv \
    --out-dir results \
    --run-name yeast_laptop_v1 \
    --profile laptop \
    --split-seeds 0 1 2 \
    --model-seeds 100 101
```

Use `--config configs/benchmark_grid.example.toml` for a versionable TOML
specification. CLI values override TOML values. Extra `ppi-train` options go
after `--`; grid-defining options cannot be overridden there. An existing
benchmark root is never overwritten.

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

## Frozen protein encoders and embedding cache

`ProteinEncoder` is a model-neutral interface; the first concrete adapter is
Hugging Face ESM-2. `--features plm` encodes each distinct sequence in the
complete sampled cohort once, before split-specific C2/C3 edge discards and
including proteins assigned to held-out test. This is safe because the encoder
is frozen and its API never receives labels or split assignments. Test features
and metrics are still not constructed unless `--eval-test-set` is supplied.

Cache entries are addressed by the normalized sequence SHA-256 inside a
namespace that covers the adapter version, exact model and tokenizer revisions,
pooling, output layer, maximum length, truncation policy, and precision. Remote
Hugging Face models therefore require immutable 40-character commit revisions;
floating names such as `main` are rejected. A cache hit does not load model
weights. “Frozen during encoding” and “label-independent” are separate
provenance properties: post-fine-tuning checkpoint identities must include both
the checkpoint SHA-256 and training-split SHA-256. They therefore cannot share
pretrained or cross-split cache namespaces, and mutable encoders cannot be
cached before a checkpoint is written.

For a local model directory, the resolved directory path plus the supplied
revision string form the model identity. Treat that directory as immutable and
change the revision whenever its contents change. The pipeline intentionally
does not re-hash multi-gigabyte local weights on every cache-only grid process.

By default, per-encoder SQLite caches live under
`~/.cache/ppi-leakage/protein_embeddings/`, outside timestamped run
directories. Override this globally with `PPI_EMBEDDING_CACHE_DIR` or per
command with `--embedding-cache-dir`. This shared location lets random,
C1, C2, C3, and repeated split-seed runs reuse the same frozen embeddings.
Writes are transactional, payload checksums are verified on read, and every run
writes cache/encoder provenance to `protein_encoder.json` and
`performance.jsonl`.

Uncached sequences are sorted by tokenized length and batched under
`--plm-max-batch-tokens`, with `--plm-max-batch-sequences` as a second safety
cap. The pooled protein rows are composed into symmetric PPI features using
sum, absolute difference, and elementwise product. Dense pair matrices are
allocated once and filled in chunks.

The ESM adapter also exposes `tokenize_with_alignment` and
`encode_token_batch`. Token-level results include attention masks and an exact
zero-based residue-to-token map. Pooled encoding remains the efficient cached
wrapper used by existing PPI runs; token tensors are produced only when a
residue task explicitly requests them and are not added to the pooled SQLite
cache.

Example for one split:

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --features plm \
    --plm-model facebook/esm2_t6_8M_UR50D \
    --plm-revision <40-character-hugging-face-commit> \
    --plm-max-batch-tokens 4096 \
    --classifier sgd_logistic \
    --run-dir results/yeast_esm2
```

Add the same frozen encoder to every split strategy in a benchmark grid with:

```bash
INCLUDE_PLM=1 \
PLM_REVISION=<40-character-hugging-face-commit> \
PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh
```

Set `PLM_MODEL` or `EMBEDDING_CACHE_DIR` to override their grid defaults.
Frozen PLM is opt-in for both laptop and exhaustive profiles so existing grid
sizes and runtimes remain unchanged.

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

Its matrix handling is now a task connector over the reusable Torch trainer.
The trainer owns optimization, device/autocast precision, configurable
validation monitoring, early stopping, scheduler/scaler state, RNG state, and
resumption. Use `--torch-validation-monitor loss` to monitor validation loss
instead of AUPRC and `--torch-precision` to select `float32`, `float16`, or
`bfloat16` where supported by the requested device.

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
    --model-seeds 0 \
    --torch-max-epochs 100 \
    --torch-resume-from results/yeast_torch_mlp/checkpoints/<execution-id>/tfidf__torch_mlp__run_1.last.pt \
    --run-dir results/yeast_torch_mlp_resumed
```

`--torch-max-epochs` is the total target, not the number of additional epochs.
Resume validates the model seed, training configuration, and exact sparse
train/validation data signatures before restoring optimizer and RNG state.
Resume must write to a new `--run-dir`; the source `*.last.pt` remains
immutable, and its path plus SHA-256 are recorded in the new performance
report. `*.best.pt` is inference-only and is intentionally rejected as a
resume source.

Checkpoints carry task/schema identity, exact task-owned split signatures, and
independent encoder/task-head component dictionaries. Last checkpoints also
carry optimizer, optional scheduler, autocast scaler, RNG, history, and
early-stopping state. The legacy single `model_state_dict` field remains for
compatibility with existing `torch_mlp` checkpoints.

## PTM task primitives

The package now provides `PTMResidueDataset`, protein-aware split assignment,
`PTMWindowCollator`, and `PTMResidueHead`. Windows are generated only after
protein-level assignment, and the collator resolves the target through the
encoder-provided residue map rather than assuming a fixed special-token
offset. `PPIPairHead` similarly owns Torch sum/absolute-difference/product
composition for a future end-to-end PPI PLM model. These are framework
connectors rather than a PTM CLI: dataset-specific PTM loading, negative-site
policy, split strategy, and benchmark metrics still need to be chosen before
shipping an end-to-end PTM command.

To add the model to every feature set in an existing laptop or exhaustive
benchmark grid without changing that profile's defaults:

```bash
INCLUDE_TORCH_MLP=1 PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh \
    --torch-max-epochs 30 --torch-batch-size 256
```

The canonical training CLI uses `--classifier`, `--split-seed`, and explicit
`--model-seeds`. The older `--classifiers`, `--seed`,
`--model-seed`/`--num-reruns`, `--append-results`, `--execution-id`, and
`--split-name` spellings remain accepted as hidden compatibility options but
emit deprecation warnings where applicable.

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
