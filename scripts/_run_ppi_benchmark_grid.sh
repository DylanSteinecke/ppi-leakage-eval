#!/usr/bin/env bash

# Shared benchmark grid for the example runners. This file is sourced, not run
# directly. The caller must set PAIRS, FASTA, OUT_DIR, and the default values
# for NUM_RERUNS, MAX_ITER, and K before sourcing it. BENCHMARK_PROFILE selects
# a laptop or exhaustive feature/model grid.

PAIRS="${PAIRS:?Set PAIRS before sourcing _run_ppi_benchmark_grid.sh}"
FASTA="${FASTA:?Set FASTA before sourcing _run_ppi_benchmark_grid.sh}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR before sourcing _run_ppi_benchmark_grid.sh}"
PROTEIN_METADATA="${PROTEIN_METADATA:-}"
NUM_RERUNS="${NUM_RERUNS:?Set NUM_RERUNS before sourcing _run_ppi_benchmark_grid.sh}"
MAX_ITER="${MAX_ITER:?Set MAX_ITER before sourcing _run_ppi_benchmark_grid.sh}"
K="${K:?Set K before sourcing _run_ppi_benchmark_grid.sh}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-exhaustive}"
SAMPLING_SEED="${SAMPLING_SEED:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-0.80}"
VAL_SIZE="${VAL_SIZE:-0.10}"
SPLIT_SEEDS="${SPLIT_SEEDS:-0}"
MODEL_SEED="${MODEL_SEED:-0}"
EXECUTION_ID="${EXECUTION_ID:-${BENCHMARK_PROFILE}_models_$(date -u +%Y-%m-%d_%H-%M-%S)}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
AGGREGATE_RESULTS="${AGGREGATE_RESULTS:-1}"
INCLUDE_TORCH_MLP="${INCLUDE_TORCH_MLP:-0}"
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
read -r -a SPLIT_SEED_VALUES <<< "$SPLIT_SEEDS"
if [[ "${#SPLIT_SEED_VALUES[@]}" -eq 0 ]]; then
    echo "SPLIT_SEEDS must contain at least one integer seed." >&2
    return 2
fi
for split_seed in "${SPLIT_SEED_VALUES[@]}"; do
    if [[ ! "$split_seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid split seed '$split_seed'; expected an integer." >&2
        return 2
    fi
done

case "$BENCHMARK_PROFILE" in
    laptop)
        if [[ -z "${MAX_PAIRS+x}" ]]; then
            MAX_PAIRS=10000
        fi
        if [[ -z "${N_SPLIT_TRIALS+x}" ]]; then
            N_SPLIT_TRIALS=25
        fi
        FEATURE_SETS=(
            tfidf
            count
        )
        LEARNED_CLASSIFIERS=(
            sgd_logistic
        )
        ;;
    exhaustive)
        if [[ -z "${MAX_PAIRS+x}" ]]; then
            MAX_PAIRS=""
        fi
        if [[ -z "${N_SPLIT_TRIALS+x}" ]]; then
            N_SPLIT_TRIALS=100
        fi
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
        ;;
    *)
        echo "Unknown BENCHMARK_PROFILE '$BENCHMARK_PROFILE'; expected laptop or exhaustive." >&2
        return 2
        ;;
esac

if [[ "$INCLUDE_TORCH_MLP" == "1" ]]; then
    LEARNED_CLASSIFIERS+=(torch_mlp)
elif [[ "$INCLUDE_TORCH_MLP" != "0" ]]; then
    echo "INCLUDE_TORCH_MLP must be 0 or 1." >&2
    return 2
fi

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
echo "Benchmark profile: $BENCHMARK_PROFILE"
echo "Feature sets: ${FEATURE_SETS[*]}"
echo "Learned classifiers: ${LEARNED_CLASSIFIERS[*]}"
echo "Split fractions: train=$TRAIN_SIZE, val=$VAL_SIZE, test=remainder"
echo "Split seeds: ${SPLIT_SEED_VALUES[*]}"
echo "First model seed: $MODEL_SEED"

for split_strategy in "${SPLIT_STRATEGIES[@]}"; do
    for split_seed in "${SPLIT_SEED_VALUES[@]}"; do
        STRATEGY_EXECUTION_ID="${EXECUTION_ID}__${split_strategy}__split_seed_${split_seed}"
        RUN_DIR="$OUT_DIR/${split_strategy}_seed-${split_seed}_${RUN_STAMP}"
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
                --split-seed "$split_seed" \
                --model-seed "$MODEL_SEED" \
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

        echo "Finished strategy: $split_strategy; split seed: $split_seed"
        echo "Execution ID: $STRATEGY_EXECUTION_ID"
        echo "Run directory: $RUN_DIR"
        echo "Metrics: $RUN_DIR/train_metrics.csv, $RUN_DIR/val_metrics.csv"
        echo "Split assignments (including held-out test): $RUN_DIR/splits/split_assignments.csv"
        echo "Summaries: $RUN_DIR/*_metrics_summary.csv"
        echo "Plots: $RUN_DIR/plots/"
        echo "Performance: $RUN_DIR/performance.jsonl"
    done
done

if [[ "$AGGREGATE_RESULTS" == "1" ]]; then
    ppi-aggregate \
        --benchmark-dir "$OUT_DIR" \
        --plot-execution-id-prefix "${EXECUTION_ID}__"
    echo "Benchmark manifest: $OUT_DIR/benchmark_manifest.csv"
    echo "Benchmark summary: $OUT_DIR/benchmark_summary.csv"
    if [[ -f "$OUT_DIR/benchmark_train_val_f1.png" ]]; then
        echo "Train/validation F1 plot: $OUT_DIR/benchmark_train_val_f1.png"
    fi
fi

echo "Finished all runs for all split strategies."
