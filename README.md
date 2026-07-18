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
in `datasets/`, model-framework adapters are in `backends/`, split construction
and audits are in `splitting/`, and result generation is in `reporting/`. The
runner uses a backend-neutral fit/predict contract. Classical classifiers
retain the existing sklearn factory and scoring path, while `torch_mlp`
exercises the batched Torch path.

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

The toy runner defaults to the `exhaustive` profile and runs six split
configurations: random, C1, identity-grouped C2/C3, and MMseqs2-grouped C2/C3.
MMseqs2 must be available as `mmseqs` on `PATH`. Set
`INCLUDE_SEQUENCE_CLUSTER_SPLITS=0` for only the dependency-free identity
grid, `BENCHMARK_PROFILE=laptop` for the smaller model grid, or
`GENERATE_TOY_DATA=0` to reuse the existing files under `processed/`.

## Run the yeast BioGRID example

The local UniProt FASTA uses headers such as `sp|P04387|GAL80_YEAST`, while
BioGRID stores the matching accession as `P04387`. The example runner prepares
the current local files, samples negatives within the yeast taxon, and runs a
laptop-sized sparse benchmark. The yeast wrapper selects a deterministic,
label-stratified cohort of 50,000 pairs before creating each split. The larger
workflow-specific cohort is needed for C2's train-side representation
invariant; a 10,000-pair yeast sample is too sparse even though that remains
the general `laptop` profile default:

```bash
conda activate ppi
BENCHMARK_PROFILE=laptop bash scripts/run_yeast_biogrid_ppi_example.sh
```

The runner profiles are:

- `laptop`: TF-IDF and count features with SGD logistic regression and 25
  C-split trials. The general profile limit is 10,000 pairs; the yeast wrapper
  overrides it to 50,000 for feasible C2 projection.
- `exhaustive`: TF-IDF, BM25, count, binary, and their combined feature set with
  logistic regression, linear SVM, and SGD logistic regression; the full
  cohort and 100 C-split trials are used by default.

Both profiles include the leakage-safe `degree_logistic` control and the
constant baselines. The fixed `degree_hgb` sensitivity control is registered
but excluded from default grids. The example scripts create an
`identity` child grid with random/C1/C2/C3 and an
`mmseqs2_id0.30_cov0.80_mode0` child grid with C2/C3; suite-level aggregation
compares all six configurations. Set
`MAX_PAIRS` or `N_SPLIT_TRIALS` explicitly to override a profile default. An
explicitly empty `MAX_PAIRS` uses the entire eligible cohort. `SAMPLING_SEED`
controls cohort selection independently of the data-split seed. `SPLIT_SEEDS`
accepts a quoted, space-separated list. `MODEL_SEEDS` explicitly lists every
independent model-fit seed. Negative construction has its own
`NEGATIVE_SAMPLING_SEED`; it is not controlled by `SAMPLING_SEED`.
`NEGATIVE_SAMPLING_POLICY` selects `taxon_pair_matched` (the default) or
`global` when the yeast dataset is prepared.

The grouped child accepts `SEQUENCE_CLUSTER_MIN_SEQ_ID`,
`SEQUENCE_CLUSTER_COVERAGE`, `SEQUENCE_CLUSTER_COV_MODE`,
`SEQUENCE_CLUSTER_EVALUE`, `SEQUENCE_CLUSTER_SENSITIVITY`,
`SEQUENCE_CLUSTER_CLUSTER_MODE`, `SEQUENCE_CLUSTER_THREADS`, and
`SEQUENCE_CLUSTER_CACHE_DIR` environment overrides. Set `SEQUENCE_CLUSTERS` to
use a supplied CSV for only the grouped child, or set
`INCLUDE_SEQUENCE_CLUSTER_SPLITS=0` to omit that child. Give every threshold
sweep a distinct `RUN_NAME` (and, when useful, `SEQUENCE_CLUSTER_GRID_NAME`).
For example:

```bash
SPLIT_SEEDS="0 1 2 3 4" MODEL_SEEDS="100 101 102" \
    PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh
```

Each example invocation creates one suite at `<OUT_DIR>/<RUN_NAME>/` with two
immutable child grids. Every split strategy, split seed, baseline group, and
feature set gets an independent run directory; no grid cell appends into
another cell's files. Each child records `benchmark_config.json`, while
aggregation writes `benchmark_manifest.csv`, `benchmark_summary.csv`,
`benchmark_degree_summary.csv`, `benchmark_degree_lift.csv`,
`benchmark_degree_control_selection.json`, and
`benchmark_train_val_f1.png` at the suite root. Omit `RUN_NAME` to use a
timestamped name.

The grid defaults to a genuine 80/10/10 train/validation/test split. Test
remains held out unless `--eval-test-set` is supplied. After aggregation, the
current grid invocation is summarized in
`benchmark_train_val_f1.png`: each split strategy and grouping instance has its
own panel, marker positions show absolute F1, and the train-to-validation arrow
shows the generalization gap. Distinct identity, supplied-cluster, and MMseqs2
groupings are never averaged together. Panels are stacked vertically on one
shared F1 scale, so better or worse performance can be compared by horizontal
position.
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
    --max-pairs 50000 \
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

The loader reads the large archive in chunks. The archive member includes a
BioGRID release number and must be updated when the `LATEST` download changes.

### PPI negative construction

The generated-negative flags are grouped under **Negative construction** in
`ppi-prepare`: `--sample-negatives`, `--negative-sampling-policy`,
`--negative-ratio`, and `--negative-sampling-seed`. The generic edge loader can
instead accept source rows with `--negative-pairs`.

`taxon_pair_matched` is the generated default. It samples unobserved unordered
pairs from taxonomy-pair strata represented by positives and requires a
`taxon_id` for every eligible protein; incomplete taxonomy fails rather than
silently switching policies. `global` is an explicit opt-in that samples from
the full eligible unordered-pair space. In both cases, label 0 means a sampled
unobserved pair under the recorded evidence snapshot, not an experimentally
confirmed non-interaction.

Generated negatives are currently prepared before split assignment. C1/C2/C3
then project positive and generated examples by the same rules, but this is not
partition-aware negative construction and does not establish full protocol
conformance. `dataset_metadata.json` records the policy, timing
(`before_split`), ratio, seed, candidate universe, and realized counts.

To compare policies, prepare two immutable dataset directories with different
dataset names, keep the evidence snapshot, ratio, negative-sampling seed,
split seeds, and model seeds fixed, then report the results as separate
negative-construction conditions. Use a multi-taxon dataset: on the yeast
single-taxon example, `taxon_pair_matched` and `global` have the same candidate
space and are not an informative comparison.

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

## Degree and hub-bias diagnostic

Every PPI run profiles the retained training graph without consulting
validation or test edges. Positive degree counts distinct positive training
partners; training exposure counts all retained training examples containing a
protein. The training-only profile is written to
`splits/training_positive_degree.csv`, with hashes, bin cutoffs, and provenance
under `diagnostics.degree_diagnostic` in `splits/split_metadata.json`.

Validation and, only with `--eval-test-set`, test reporting is written to
`val_degree_metrics.csv` / `test_degree_metrics.csv` and matching summary
files. These contain the threshold-free preferential-attachment reference,
global and degree-stratified metrics for every requested model, counts and
prevalence, and the single global validation-selected threshold. C2 reports
the familiar endpoint because the novel endpoint degree is structurally zero;
C3 reports global controls and marks degree stratification inapplicable. Use
`--degree-bin-quantiles 0.5 0.9` (or the matching grid/TOML option) to change
the default training-distribution cutoffs.

Benchmark aggregation joins biological models to the fitted control only when
the dataset, protocol instance, grouping, negative construction, assignments,
training graph, and exact evaluation cohort hashes agree. The headline AUPRC
quantity is `residual_over_degree_control`; it is a residual diagnostic, not a
claim of genuine biological signal. The task-owned matrix schema
`ppi.training_degree.v1` uses three ordered symmetric columns:
`degree_log_min`, `degree_log_max`, and `degree_log_product`, where each
endpoint degree is transformed with `log1p` first. Training positives use
leave-one-canonical-positive-edge-out degrees; negatives and held-out examples
use the complete positive training graph. Individual `degree_logistic`
coefficients should not be interpreted independently because these derived
features are correlated.

Aggregation locks the fitted-control decision in
`benchmark_degree_control_selection.json` using validation evidence only,
before reading test degree metrics. Re-aggregation validates the evidence hash
and refuses to change the choice. Use a new benchmark directory if validation
evidence or the pre-publication control decision changes. The first aggregation
is the lock event; if the five-seed HGB evidence is missing, the predeclared
fallback deliberately locks `degree_logistic`.

## Frozen protein encoders and embedding cache

`ProteinEncoder` is a model-neutral interface. Model families are selected
explicitly with `--plm-adapter`; the default and first concrete adapter is
Hugging Face ESM-2. Each adapter owns its tokenization, exact token accounting,
pooling, and residue alignment, while batching and caching remain shared.
`--features plm` encodes each distinct sequence in the
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

Every Hugging Face adapter also exposes `tokenize_with_alignment` and
`encode_token_batch`. Token-level results include attention masks and an exact
zero-based residue-to-token map. Pooled encoding remains the efficient cached
wrapper used by existing PPI runs; token tensors are produced only when a
residue task explicitly requests them and are not added to the pooled SQLite
cache.

The benchmark runners provide exact, self-supervised encoder presets:

| Preset | Adapter | Intended profile |
| --- | --- | --- |
| `esm2_8m` | ESM-2 8M | laptop or exhaustive |
| `esm2_35m` | ESM-2 35M | exhaustive with an accelerator |
| `esm2_150m` | ESM-2 150M | exhaustive with an accelerator |
| `esm2_650m` | ESM-2 650M | exhaustive with an accelerator |
| `esm2_3b` | ESM-2 3B | exhaustive with a large accelerator |
| `esm2_15b` | ESM-2 15B | exhaustive with very large accelerator memory |
| `protbert` | ProtBERT | laptop or exhaustive |
| `prott5_xl` | ProtT5-XL encoder | exhaustive with an accelerator |

Presets pin the model and tokenizer revisions and carry conservative batching,
precision, and device defaults. The laptop profile rejects larger ESM2 sizes
and `prott5_xl` before loading or downloading weights. The same ESM2 adapter
accepts every size; only the preset model identity and resource defaults vary.
ProtBERT is laptop-compatible, but appreciably
slower and larger than ESM-2 8M. Noncanonical residues `U`, `Z`, `O`, and `B`
are normalized to `X` for both ProtTrans adapters; ESM-2 retains its native
tokenization. Adapter implementations are part of the cache namespace, so
these family-specific preprocessing rules cannot produce cross-family cache
hits.

Example for one split:

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --features plm \
    --plm-adapter esm2 \
    --plm-model facebook/esm2_t6_8M_UR50D \
    --plm-revision <40-character-hugging-face-commit> \
    --plm-max-batch-tokens 4096 \
    --classifier sgd_logistic \
    --run-dir results/yeast_esm2
```

Run both laptop presets over the configured split strategies with:

```bash
PREPARE_YEAST_DATA=0 \
BENCHMARK_PROFILE=laptop \
MAX_PAIRS=1000 \
conda run -n ppi bash scripts/run_yeast_biogrid_ppi_example.sh \
    --plm-preset esm2_8m \
    --plm-preset protbert \
    --eval-test-set
```

ProtT5 is available only through a non-laptop grid profile:

```bash
PREPARE_YEAST_DATA=0 \
BENCHMARK_PROFILE=exhaustive \
    MAX_PAIRS=40000 \
conda run -n ppi bash scripts/run_yeast_biogrid_ppi_example.sh \
    --plm-preset prott5_xl \
    --eval-test-set
```

Larger ESM2 sizes use the same syntax, and multiple sizes may be compared in
one exhaustive grid:

```bash
PREPARE_YEAST_DATA=0 \
BENCHMARK_PROFILE=exhaustive \
MAX_PAIRS=40000 \
conda run -n ppi bash scripts/run_yeast_biogrid_ppi_example.sh \
    --plm-preset esm2_150m \
    --plm-preset esm2_650m \
    --eval-test-set
```

The singular `PLM_ADAPTER`, `PLM_MODEL`, and `PLM_REVISION` interface remains
available for deliberate custom runs. Approved benchmark presets are limited
to the self-supervised models above. Adding another model family requires only
a small adapter registered in `protein_encoders/factory.py`; caching, pair
composition, training backends, and evaluation remain shared. Frozen PLMs stay
opt-in for both profiles, preserving existing grid size and runtime by default.

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
CLASSIFIERS="degree_logistic always_positive always_negative logistic \
linear_svm sgd_logistic torch_mlp" PREPARE_YEAST_DATA=0 \
    bash scripts/run_yeast_biogrid_ppi_example.sh \
    --torch-max-epochs 30 --torch-batch-size 256
```

The canonical training CLI uses `--classifier`, `--split-seed`, and explicit
`--model-seeds`. The older `--classifiers`, `--seed`,
`--model-seed`/`--num-reruns`, `--append-results`, `--execution-id`, and
`--split-name` spellings remain accepted as hidden compatibility options but
emit deprecation warnings where applicable.

## Leakage-aware splits

The normative concepts, compatibility rules, versioning policy, and extension
contract are defined in
[`docs/split_semantics.md`](docs/split_semantics.md). Task-specific definitions
are in [`docs/split_protocol_catalog.md`](docs/split_protocol_catalog.md), and
implementation boundaries and test guidance are in
[`docs/split_engineering_guide.md`](docs/split_engineering_guide.md). Existing
short split names remain legacy CLI identifiers until an explicit conformance
review maps them to task-qualified protocol versions.

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

For sequence-cluster-aware C2/C3 splits, either generate MMseqs2 groups
automatically or provide a precomputed CSV. Automatic grouping is
task-independent; only the subsequent C2/C3 edge projection is PPI-specific:

```bash
ppi-train \
    --pairs processed/biogrid_yeast_physical/pairs.csv \
    --fasta processed/biogrid_yeast_physical/proteins.fasta \
    --sequence-cluster-method mmseqs2 \
    --sequence-cluster-min-seq-id 0.30 \
    --sequence-cluster-coverage 0.80 \
    --sequence-cluster-cov-mode 0 \
    --split-strategy c3 \
    --run-dir results/yeast_mmseqs2_c3
```

The defaults are identity `0.30`, coverage `0.80`, coverage mode `0`, E-value
`0.001`, automatic sensitivity and cluster mode, and one thread. Automatic
sensitivity omits `-s`; automatic cluster mode omits `--cluster-mode`.
MMseqs2 defines coverage modes as: `0` query and target coverage, `1` target
coverage, `2` query coverage, `3` minimum target/query length ratio, `4`
minimum query/target length ratio, and `5` minimum shorter/longer length ratio.
The `-c` value is interpreted under the selected mode.

For supplied groups:

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
    --run-dir results/yeast_sequence_cluster_c3
```

Every eligible cohort protein must have exactly one mapping; coverage is
validated before optional cohort sampling. C2/C3 then treat each cluster as an
indivisible group. The normalized mapping is copied to
`splits/sequence_cluster_assignments.csv`, and its path, hash, counts, and split
audit are recorded in `split_metadata.json`. Supplied and automatic grouping
are mutually exclusive, and either source is rejected for random, C1, and
provided-column protocols.

Generated mappings use a content-addressed cache selected by
`PPI_SEQUENCE_CLUSTER_CACHE_DIR`, then
`$XDG_CACHE_HOME/ppi-leakage/sequence_clusters`, then
`~/.cache/ppi-leakage/sequence_clusters`. Metadata records the exact tool
version and command, all effective parameters, universe and mapping hashes,
cluster-size statistics, cache fingerprint, and cache hit. One `ppi-grid`
invocation accepts one grouping configuration; use separate versioned TOML
files and run names for threshold sweeps. See
`configs/benchmark_grid.mmseqs2.example.toml`.

These runs are described as **MMseqs2 sequence-cluster-disjoint**. MMseqs2's
clustering threshold does not itself prove that every cross-partition sequence
pair is below that threshold; a strict homology-disjoint claim requires a
separate all-vs-all audit.
