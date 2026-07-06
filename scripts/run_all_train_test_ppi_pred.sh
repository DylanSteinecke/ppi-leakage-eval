#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
PAIRS="${PAIRS:-processed/toy_pairs.csv}"
FASTA="${FASTA:-processed/toy_sequences.fasta}"
OUT_DIR="${OUT_DIR:-results/toy_all_models}"

FEATURES=(tfidf bm25 count binary)
CLASSIFIERS=(logistic linear_svm sgd_logistic)

mkdir -p "$OUT_DIR"

combined_metrics="$OUT_DIR/metrics_all.csv"
first_metrics=1

for features in "${FEATURES[@]}"; do
    for classifier in "${CLASSIFIERS[@]}"; do
        run_name="${features}_${classifier}"
        predictions_out="$OUT_DIR/predictions_${run_name}.csv"
        metrics_out="$OUT_DIR/metrics_${run_name}.csv"

        echo "Running features=${features} classifier=${classifier}"
        "$PYTHON" scripts/train_test_ppi_pred.py \
            --pairs "$PAIRS" \
            --fasta "$FASTA" \
            --features "$features" \
            --classifier "$classifier" \
            --out "$predictions_out" \
            --metrics-out "$metrics_out" \
            "$@"

        if [[ "$first_metrics" -eq 1 ]]; then
            cp "$metrics_out" "$combined_metrics"
            first_metrics=0
        else
            tail -n +2 "$metrics_out" >> "$combined_metrics"
        fi
    done
done

echo "Finished all runs."
echo "Combined metrics: $combined_metrics"
echo "Predictions and per-run metrics: $OUT_DIR"
