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
EXECUTION_ID="${EXECUTION_ID:-all_models_$(date +%Y%m%d_%H%M%S)}"
USER_ARGS=("$@")
SPLIT_STRATEGIES=(
    random
    protein_disjoint_components
)

FEATURE_SETS=(
    tfidf
    bm25
    count
    binary
    "tfidf bm25"
    "tfidf count"
    "tfidf binary"
    "bm25 count"
    "bm25 binary"
    "count binary"
    "tfidf bm25 count"
    "tfidf bm25 binary"
    "tfidf count binary"
    "bm25 count binary"
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

the_date=$(date +%m-%d-%y__%H_%M_%S)

for split_strategy in "${SPLIT_STRATEGIES[@]}"; do
    STRATEGY_EXECUTION_ID="${EXECUTION_ID}__${split_strategy}"

    PRED_OUT="$OUT_DIR/predictions_all_${split_strategy}_${the_date}.csv"
    TRAIN_METRICS_OUT="$OUT_DIR/train_metrics_runs_${split_strategy}_${the_date}.csv"
    TEST_METRICS_OUT="$OUT_DIR/test_metrics_runs_${split_strategy}_${the_date}.csv"
    TRAIN_SUMMARY_OUT="$OUT_DIR/train_metrics_summary_${split_strategy}_${the_date}.csv"
    TEST_SUMMARY_OUT="$OUT_DIR/test_metrics_summary_${split_strategy}_${the_date}.csv"
    TRAIN_PLOT_OUT="$OUT_DIR/train_metrics_summary_${split_strategy}_${the_date}.svg"
    TEST_PLOT_OUT="$OUT_DIR/test_metrics_summary_${split_strategy}_${the_date}.svg"

    APPEND_ARGS=()

    # Baseline classifiers
    "$PYTHON" scripts/train_test_ppi_pred.py \
        --pairs "$PAIRS" \
        --fasta "$FASTA" \
        --classifier "${BASELINE_CLASSIFIERS[@]}" \
        --num-reruns "$NUM_RERUNS" \
        --max-iter "$MAX_ITER" \
        --train-size "$TRAIN_SIZE" \
        --split-strategy "$split_strategy" \
        --execution-id "$STRATEGY_EXECUTION_ID" \
        --pred-out "$PRED_OUT" \
        --train-metrics-out "$TRAIN_METRICS_OUT" \
        --test-metrics-out "$TEST_METRICS_OUT" \
        --train-metrics-summary-out "$TRAIN_SUMMARY_OUT" \
        --test-metrics-summary-out "$TEST_SUMMARY_OUT" \
        --train-metrics-plot-out "$TRAIN_PLOT_OUT" \
        --test-metrics-plot-out "$TEST_PLOT_OUT" \
        "${USER_ARGS[@]}"

    APPEND_ARGS=(--append-results)

    # Learned classifiers
    for feature_set in "${FEATURE_SETS[@]}"; do
        IFS=" " read -r -a FEATURE_ARGS <<< "$feature_set"

        "$PYTHON" scripts/train_test_ppi_pred.py \
            --pairs "$PAIRS" \
            --fasta "$FASTA" \
            --features "${FEATURE_ARGS[@]}" \
            --classifier "${LEARNED_CLASSIFIERS[@]}" \
            --num-reruns "$NUM_RERUNS" \
            --max-iter "$MAX_ITER" \
            --train-size "$TRAIN_SIZE" \
            --split-strategy "$split_strategy" \
            --execution-id "$STRATEGY_EXECUTION_ID" \
            --pred-out "$PRED_OUT" \
            --train-metrics-out "$TRAIN_METRICS_OUT" \
            --test-metrics-out "$TEST_METRICS_OUT" \
            --train-metrics-summary-out "$TRAIN_SUMMARY_OUT" \
            --test-metrics-summary-out "$TEST_SUMMARY_OUT" \
            --train-metrics-plot-out "$TRAIN_PLOT_OUT" \
            --test-metrics-plot-out "$TEST_PLOT_OUT" \
            "${APPEND_ARGS[@]}" \
            "${USER_ARGS[@]}"
    done

    echo "Finished strategy: $split_strategy"
    echo "Execution ID: $STRATEGY_EXECUTION_ID"
    echo "Train per-run metrics: $TRAIN_METRICS_OUT"
    echo "Test per-run metrics: $TEST_METRICS_OUT"
    echo "Train metric summary: $TRAIN_SUMMARY_OUT"
    echo "Test metric summary: $TEST_SUMMARY_OUT"
    echo "Train metric plot: $TRAIN_PLOT_OUT"
    echo "Test metric plot: $TEST_PLOT_OUT"
    echo "Predictions: $PRED_OUT"
done

echo "Finished all runs for all split strategies."
