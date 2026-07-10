#!/usr/bin/env bash
set -euo pipefail

# Prepare and run the local yeast BioGRID example with:
# YEAST_EXAMPLE=1 bash scripts/run_all_train_test_ppi_pred.sh --no-metrics-plots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

# Input data args: yeast or toy data
YEAST_EXAMPLE="${YEAST_EXAMPLE:-0}"
YEAST_DATASET_NAME="${YEAST_DATASET_NAME:-biogrid_yeast_physical}"
if [[ "$YEAST_EXAMPLE" == "1" ]]; then
    PAIRS_DEFAULT="processed/$YEAST_DATASET_NAME/pairs.csv"
    FASTA_DEFAULT="processed/$YEAST_DATASET_NAME/proteins.fasta"
    PROTEIN_METADATA_DEFAULT="processed/$YEAST_DATASET_NAME/protein_metadata.csv"
    OUT_DIR_DEFAULT="results/${YEAST_DATASET_NAME}_example"
    NUM_RERUNS_DEFAULT=1
    MAX_ITER_DEFAULT=100
    K_DEFAULT=2
else
    PAIRS_DEFAULT="processed/toy_pairs.csv"
    FASTA_DEFAULT="processed/toy_sequences.fasta"
    PROTEIN_METADATA_DEFAULT=""
    OUT_DIR_DEFAULT="results/toy_all_models"
    NUM_RERUNS_DEFAULT=5
    MAX_ITER_DEFAULT=10000
    K_DEFAULT=3
fi

PAIRS="${PAIRS:-$PAIRS_DEFAULT}"
FASTA="${FASTA:-$FASTA_DEFAULT}"
PROTEIN_METADATA="${PROTEIN_METADATA:-$PROTEIN_METADATA_DEFAULT}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
NUM_RERUNS="${NUM_RERUNS:-$NUM_RERUNS_DEFAULT}"
MAX_ITER="${MAX_ITER:-$MAX_ITER_DEFAULT}"
K="${K:-$K_DEFAULT}"
TRAIN_SIZE="${TRAIN_SIZE:-0.80}"
VAL_SIZE="${VAL_SIZE:-0.0}"
EXECUTION_ID="${EXECUTION_ID:-all_models_$(date -u +%Y-%m-%d_%H-%M-%S)}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
AGGREGATE_RESULTS="${AGGREGATE_RESULTS:-1}"
USER_ARGS=("$@")
BASELINE_CLASSIFIERS=(
    always_positive
    always_negative
)
    SPLIT_STRATEGIES=(
        random
        c1
        c2
        c3
    )
FEATURE_SETS=(
    tfidf
    bm25
    count
    binary
    "tfidf bm25 count binary"
)
LEARNED_CLASSIFIERS=(
    logistic
    linear_svm
    sgd_logistic
)

# Yeast PPIs
if [[ "$YEAST_EXAMPLE" == "1" ]]; then
    BIOGRID_ARCHIVE="${BIOGRID_ARCHIVE:-input/BIOGRID-ORGANISM-LATEST.tab3.zip}"
    BIOGRID_ARCHIVE_MEMBER="${BIOGRID_ARCHIVE_MEMBER:-BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-5.0.259.tab3.txt}"
    YEAST_FASTA="${YEAST_FASTA:-input/UP000002311_559292.fasta}"
    NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
    PREPARE_YEAST_DATA="${PREPARE_YEAST_DATA:-1}"
    if [[ "$PREPARE_YEAST_DATA" == "1" ]]; then
        "$PYTHON" scripts/prepare_ppi_dataset.py biogrid \
            --dataset-name "$YEAST_DATASET_NAME" \
            --interactions "$BIOGRID_ARCHIVE" \
            --archive-member "$BIOGRID_ARCHIVE_MEMBER" \
            --fasta "$YEAST_FASTA" \
            --fasta-id-format uniprot_accession \
            --protein-a-col "SWISS-PROT Accessions Interactor A" \
            --protein-b-col "SWISS-PROT Accessions Interactor B" \
            --organism-a-col "Organism ID Interactor A" \
            --organism-b-col "Organism ID Interactor B" \
            --organism-id 559292 \
            --experimental-system-type-col "Experimental System Type" \
            --allowed-system-types physical \
            --ambiguous-id-policy drop \
            --sample-negatives \
            --negative-ratio "$NEGATIVE_RATIO" \
            --out-dir processed \
            --overwrite
    fi
fi


PROTEIN_METADATA_ARGS=()
if [[ -n "$PROTEIN_METADATA" ]]; then
    PROTEIN_METADATA_ARGS=(--protein-metadata "$PROTEIN_METADATA")
fi

mkdir -p "$OUT_DIR"

# Try each split strategy
for split_strategy in "${SPLIT_STRATEGIES[@]}"; do
    STRATEGY_EXECUTION_ID="${EXECUTION_ID}__${split_strategy}"
    RUN_DIR="$OUT_DIR/${split_strategy}_${RUN_STAMP}"
    APPEND_ARGS=()

    run_ppi_benchmark() {
        "$PYTHON" scripts/train_test_ppi_pred.py \
            --pairs "$PAIRS" \
            --fasta "$FASTA" \
            "${PROTEIN_METADATA_ARGS[@]}" \
            --run-dir "$RUN_DIR" \
            --num-reruns "$NUM_RERUNS" \
            --max-iter "$MAX_ITER" \
            --k "$K" \
            --train-size "$TRAIN_SIZE" \
            --val-size "$VAL_SIZE" \
            --split-strategy "$split_strategy" \
            --execution-id "$STRATEGY_EXECUTION_ID" \
            "$@" \
            "${USER_ARGS[@]}"
    }

    # Baselines create a fresh canonical run directory.
    run_ppi_benchmark --classifier "${BASELINE_CLASSIFIERS[@]}"
    APPEND_ARGS=(--append-results)

    # Try each feature set
    for feature_set in "${FEATURE_SETS[@]}"; do
        IFS=" " read -r -a FEATURE_ARGS <<< "$feature_set"

        run_ppi_benchmark \
            --features "${FEATURE_ARGS[@]}" \
            --classifier "${LEARNED_CLASSIFIERS[@]}" \
            "${APPEND_ARGS[@]}"
    done

    echo "Finished strategy: $split_strategy"
    echo "Execution ID: $STRATEGY_EXECUTION_ID"
    echo "Run directory: $RUN_DIR"
    echo "Metrics: $RUN_DIR/train_metrics.csv"
    echo "Summaries: $RUN_DIR/*_metrics_summary.csv"
    echo "Plots: $RUN_DIR/plots/"
    echo "Predictions, when test is evaluated: $RUN_DIR/predictions.csv"
done

if [[ "$AGGREGATE_RESULTS" == "1" ]]; then
    "$PYTHON" scripts/aggregate_benchmark_results.py \
        --benchmark-dir "$OUT_DIR"
    echo "Benchmark manifest: $OUT_DIR/benchmark_manifest.csv"
    echo "Benchmark summary: $OUT_DIR/benchmark_summary.csv"
fi

echo "Finished all runs for all split strategies."
