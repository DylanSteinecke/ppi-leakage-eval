#!/usr/bin/env bash

# Thin compatibility wrapper around the Python grid orchestrator. This file is
# sourced by the example runners so their environment-variable interface stays
# convenient; profile expansion and execution live in ppi_benchmark.cli.grid.

PAIRS="${PAIRS:?Set PAIRS before sourcing _run_ppi_benchmark_grid.sh}"
FASTA="${FASTA:?Set FASTA before sourcing _run_ppi_benchmark_grid.sh}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR before sourcing _run_ppi_benchmark_grid.sh}"
PROTEIN_METADATA="${PROTEIN_METADATA:-}"
SEQUENCE_CLUSTERS="${SEQUENCE_CLUSTERS:-}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-exhaustive}"
SAMPLING_SEED="${SAMPLING_SEED:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-0.80}"
VAL_SIZE="${VAL_SIZE:-0.10}"
SPLIT_STRATEGIES="${SPLIT_STRATEGIES:-random c1 c2 c3}"
SPLIT_SEEDS="${SPLIT_SEEDS:-0}"
MAX_ITER="${MAX_ITER:-1000}"
K="${K:-3}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
RUN_NAME="${RUN_NAME:-${BENCHMARK_PROFILE}_${RUN_STAMP}}"
AGGREGATE_RESULTS="${AGGREGATE_RESULTS:-1}"
INCLUDE_TORCH_MLP="${INCLUDE_TORCH_MLP:-0}"
INCLUDE_PLM="${INCLUDE_PLM:-0}"
PLM_MODEL="${PLM_MODEL:-facebook/esm2_t6_8M_UR50D}"
PLM_REVISION="${PLM_REVISION:-}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-}"
USER_ARGS=("$@")

read -r -a SPLIT_STRATEGY_VALUES <<< "$SPLIT_STRATEGIES"
read -r -a SPLIT_SEED_VALUES <<< "$SPLIT_SEEDS"
if [[ "${#SPLIT_STRATEGY_VALUES[@]}" -eq 0 ]]; then
    echo "SPLIT_STRATEGIES must contain at least one strategy." >&2
    return 2
fi
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

# MODEL_SEEDS is canonical. MODEL_SEED plus NUM_RERUNS remains a compatibility
# bridge for existing laptop commands and expands once here into explicit seeds.
if [[ -n "${MODEL_SEEDS:-}" ]]; then
    read -r -a MODEL_SEED_VALUES <<< "$MODEL_SEEDS"
else
    MODEL_SEED="${MODEL_SEED:-0}"
    NUM_RERUNS="${NUM_RERUNS:-1}"
    if [[ ! "$MODEL_SEED" =~ ^-?[0-9]+$ ]]; then
        echo "MODEL_SEED must be an integer." >&2
        return 2
    fi
    if [[ ! "$NUM_RERUNS" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_RERUNS must be a positive integer." >&2
        return 2
    fi
    MODEL_SEED_VALUES=()
    for ((seed_offset = 0; seed_offset < NUM_RERUNS; seed_offset++)); do
        MODEL_SEED_VALUES+=("$((MODEL_SEED + seed_offset))")
    done
fi
for model_seed in "${MODEL_SEED_VALUES[@]}"; do
    if [[ ! "$model_seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid model seed '$model_seed'; expected an integer." >&2
        return 2
    fi
done

GRID_ARGS=(
    --pairs "$PAIRS"
    --fasta "$FASTA"
    --out-dir "$OUT_DIR"
    --run-name "$RUN_NAME"
    --profile "$BENCHMARK_PROFILE"
    --sampling-seed "$SAMPLING_SEED"
    --train-size "$TRAIN_SIZE"
    --val-size "$VAL_SIZE"
    --split-strategies "${SPLIT_STRATEGY_VALUES[@]}"
    --split-seeds "${SPLIT_SEED_VALUES[@]}"
    --model-seeds "${MODEL_SEED_VALUES[@]}"
    --max-iter "$MAX_ITER"
    --k "$K"
)

if [[ -n "$PROTEIN_METADATA" ]]; then
    GRID_ARGS+=(--protein-metadata "$PROTEIN_METADATA")
fi
if [[ -n "$SEQUENCE_CLUSTERS" ]]; then
    GRID_ARGS+=(--sequence-clusters "$SEQUENCE_CLUSTERS")
fi
if [[ -v MAX_PAIRS ]]; then
    if [[ -n "$MAX_PAIRS" ]]; then
        GRID_ARGS+=(--max-pairs "$MAX_PAIRS")
    else
        GRID_ARGS+=(--full-cohort)
    fi
fi
if [[ -n "${N_SPLIT_TRIALS:-}" ]]; then
    GRID_ARGS+=(--n-split-trials "$N_SPLIT_TRIALS")
fi

case "$AGGREGATE_RESULTS" in
    1) GRID_ARGS+=(--aggregate-results) ;;
    0) GRID_ARGS+=(--no-aggregate-results) ;;
    *)
        echo "AGGREGATE_RESULTS must be 0 or 1." >&2
        return 2
        ;;
esac
case "$INCLUDE_TORCH_MLP" in
    1) GRID_ARGS+=(--include-torch-mlp) ;;
    0) GRID_ARGS+=(--no-include-torch-mlp) ;;
    *)
        echo "INCLUDE_TORCH_MLP must be 0 or 1." >&2
        return 2
        ;;
esac
case "$INCLUDE_PLM" in
    1)
        if [[ -z "$PLM_REVISION" ]]; then
            echo "INCLUDE_PLM=1 requires an immutable PLM_REVISION." >&2
            return 2
        fi
        GRID_ARGS+=(
            --include-plm
            --plm-model "$PLM_MODEL"
            --plm-revision "$PLM_REVISION"
        )
        if [[ -n "$EMBEDDING_CACHE_DIR" ]]; then
            GRID_ARGS+=(--embedding-cache-dir "$EMBEDDING_CACHE_DIR")
        fi
        ;;
    0) GRID_ARGS+=(--no-include-plm) ;;
    *)
        echo "INCLUDE_PLM must be 0 or 1." >&2
        return 2
        ;;
esac

echo "Benchmark profile: $BENCHMARK_PROFILE"
echo "Split fractions: train=$TRAIN_SIZE, val=$VAL_SIZE, test=remainder"
echo "Split seeds: ${SPLIT_SEED_VALUES[*]}"
echo "Model seeds: ${MODEL_SEED_VALUES[*]}"

ppi-grid "${GRID_ARGS[@]}" -- "${USER_ARGS[@]}"
