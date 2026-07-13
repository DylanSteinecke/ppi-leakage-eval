#!/usr/bin/env bash

# Shared benchmark grid for the example runners. This file is sourced, not run
# directly. The caller must set PAIRS, FASTA, OUT_DIR, and the default values
# for NUM_RERUNS, MAX_ITER, and K before sourcing it.

PAIRS="${PAIRS:?Set PAIRS before sourcing _run_ppi_benchmark_grid.sh}"
FASTA="${FASTA:?Set FASTA before sourcing _run_ppi_benchmark_grid.sh}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR before sourcing _run_ppi_benchmark_grid.sh}"
PROTEIN_METADATA="${PROTEIN_METADATA:-}"
NUM_RERUNS="${NUM_RERUNS:?Set NUM_RERUNS before sourcing _run_ppi_benchmark_grid.sh}"
MAX_ITER="${MAX_ITER:?Set MAX_ITER before sourcing _run_ppi_benchmark_grid.sh}"
K="${K:?Set K before sourcing _run_ppi_benchmark_grid.sh}"
MAX_PAIRS="${MAX_PAIRS:-}"
SAMPLING_SEED="${SAMPLING_SEED:-0}"
N_SPLIT_TRIALS="${N_SPLIT_TRIALS:-100}"
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

PROTEIN_METADATA_ARGS=()
if [[ -n "$PROTEIN_METADATA" ]]; then
    PROTEIN_METADATA_ARGS=(--protein-metadata "$PROTEIN_METADATA")
fi

COHORT_SAMPLING_ARGS=()
if [[ -n "$MAX_PAIRS" ]]; then
    COHORT_SAMPLING_ARGS=(
        --max-pairs "$MAX_PAIRS"
        --sampling-seed "$SAMPLING_SEED"
    )
fi

mkdir -p "$OUT_DIR"

for split_strategy in "${SPLIT_STRATEGIES[@]}"; do
    STRATEGY_EXECUTION_ID="${EXECUTION_ID}__${split_strategy}"
    RUN_DIR="$OUT_DIR/${split_strategy}_${RUN_STAMP}"
    APPEND_ARGS=()

    run_ppi_benchmark() {
        ppi-train \
            --pairs "$PAIRS" \
            --fasta "$FASTA" \
            "${PROTEIN_METADATA_ARGS[@]}" \
            "${COHORT_SAMPLING_ARGS[@]}" \
            --run-dir "$RUN_DIR" \
            --num-reruns "$NUM_RERUNS" \
            --max-iter "$MAX_ITER" \
            --k "$K" \
            --train-size "$TRAIN_SIZE" \
            --val-size "$VAL_SIZE" \
            --split-strategy "$split_strategy" \
            --n-split-trials "$N_SPLIT_TRIALS" \
            --execution-id "$STRATEGY_EXECUTION_ID" \
            "$@" \
            "${USER_ARGS[@]}"
    }

    run_ppi_benchmark --classifier "${BASELINE_CLASSIFIERS[@]}"
    APPEND_ARGS=(--append-results)

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
    ppi-aggregate --benchmark-dir "$OUT_DIR"
    echo "Benchmark manifest: $OUT_DIR/benchmark_manifest.csv"
    echo "Benchmark summary: $OUT_DIR/benchmark_summary.csv"
fi

echo "Finished all runs for all split strategies."
