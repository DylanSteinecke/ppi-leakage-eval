#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
PAIRS="${PAIRS:-processed/toy_pairs.csv}"
FASTA="${FASTA:-processed/toy_sequences.fasta}"
OUT_DIR="${OUT_DIR:-results/toy_all_models}"
NUM_RERUNS="${NUM_RERUNS:-3}"

FEATURES=(
    tfidf
    bm25
    count
    binary
)
CLASSIFIERS=(
    logistic
    linear_svm
    sgd_logistic
    always_positive
    always_negative
)

mkdir -p "$OUT_DIR"

"$PYTHON" scripts/train_test_ppi_pred.py \
    --pairs "$PAIRS" \
    --fasta "$FASTA" \
    --features "${FEATURES[@]}" \
    --classifier "${CLASSIFIERS[@]}" \
    --num-reruns "$NUM_RERUNS" \
    --pred-out "$OUT_DIR/predictions_all.csv" \
    --metrics-out "$OUT_DIR/metrics_runs.csv" \
    --metrics-summary-out "$OUT_DIR/metrics_summary.csv" \
    "$@"

echo "Finished all runs."
echo "Per-run metrics: $OUT_DIR/metrics_runs.csv"
echo "Metric summary: $OUT_DIR/metrics_summary.csv"
echo "Predictions: $OUT_DIR/predictions_all.csv"
