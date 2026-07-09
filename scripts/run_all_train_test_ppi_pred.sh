#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
PAIRS="${PAIRS:-processed/toy_pairs.csv}"
FASTA="${FASTA:-processed/toy_sequences.fasta}"
OUT_DIR="${OUT_DIR:-results/toy_all_models}"
NUM_RERUNS="${NUM_RERUNS:-5}"
MAX_ITER="${MAX_ITER:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-0.80}"
VAL_SIZE="${VAL_SIZE:-0.0}"
EXECUTION_ID="${EXECUTION_ID:-all_models_$(date -u +%Y-%m-%d_%H-%M-%S)}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
AGGREGATE_RESULTS="${AGGREGATE_RESULTS:-1}"
USER_ARGS=("$@")
SPLIT_STRATEGIES=(
    random
    protein_disjoint_components
    #protein_disjoint_prune_edges
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
BASELINE_CLASSIFIERS=(
    always_positive
    always_negative
)

mkdir -p "$OUT_DIR"

for split_strategy in "${SPLIT_STRATEGIES[@]}"; do
    STRATEGY_EXECUTION_ID="${EXECUTION_ID}__${split_strategy}"
    RUN_DIR="$OUT_DIR/${split_strategy}_${RUN_STAMP}"
    APPEND_ARGS=()

    # Baselines create a fresh canonical run directory.
    "$PYTHON" scripts/train_test_ppi_pred.py \
        --pairs "$PAIRS" \
        --fasta "$FASTA" \
        --run-dir "$RUN_DIR" \
        --classifier "${BASELINE_CLASSIFIERS[@]}" \
        --num-reruns "$NUM_RERUNS" \
        --max-iter "$MAX_ITER" \
        --train-size "$TRAIN_SIZE" \
        --val-size "$VAL_SIZE" \
        --split-strategy "$split_strategy" \
        --execution-id "$STRATEGY_EXECUTION_ID" \
        "${USER_ARGS[@]}"
    APPEND_ARGS=(--append-results)

    # Learned classifiers append into the same canonical run directory.
    for feature_set in "${FEATURE_SETS[@]}"; do
        IFS=" " read -r -a FEATURE_ARGS <<< "$feature_set"

        "$PYTHON" scripts/train_test_ppi_pred.py \
            --pairs "$PAIRS" \
            --fasta "$FASTA" \
            --run-dir "$RUN_DIR" \
            --features "${FEATURE_ARGS[@]}" \
            --classifier "${LEARNED_CLASSIFIERS[@]}" \
            --num-reruns "$NUM_RERUNS" \
            --max-iter "$MAX_ITER" \
            --train-size "$TRAIN_SIZE" \
            --val-size "$VAL_SIZE" \
            --split-strategy "$split_strategy" \
            --execution-id "$STRATEGY_EXECUTION_ID" \
            "${APPEND_ARGS[@]}" \
            "${USER_ARGS[@]}"
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
