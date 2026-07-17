#!/usr/bin/env bash

# Thin compatibility wrapper around the Python grid orchestrator. This file is
# sourced by the example runners so their environment-variable interface stays
# convenient; profile expansion and execution live in ppi_benchmark.cli.grid.

PAIRS="${PAIRS:?Set PAIRS before sourcing _run_ppi_benchmark_grid.sh}"
FASTA="${FASTA:?Set FASTA before sourcing _run_ppi_benchmark_grid.sh}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR before sourcing _run_ppi_benchmark_grid.sh}"
PROTEIN_METADATA="${PROTEIN_METADATA:-}"
SEQUENCE_CLUSTERS="${SEQUENCE_CLUSTERS:-}"
INCLUDE_SEQUENCE_CLUSTER_SPLITS="${INCLUDE_SEQUENCE_CLUSTER_SPLITS:-1}"
SEQUENCE_CLUSTER_MIN_SEQ_ID="${SEQUENCE_CLUSTER_MIN_SEQ_ID:-0.30}"
SEQUENCE_CLUSTER_COVERAGE="${SEQUENCE_CLUSTER_COVERAGE:-0.80}"
SEQUENCE_CLUSTER_COV_MODE="${SEQUENCE_CLUSTER_COV_MODE:-0}"
SEQUENCE_CLUSTER_EVALUE="${SEQUENCE_CLUSTER_EVALUE:-0.001}"
SEQUENCE_CLUSTER_SENSITIVITY="${SEQUENCE_CLUSTER_SENSITIVITY:-auto}"
SEQUENCE_CLUSTER_CLUSTER_MODE="${SEQUENCE_CLUSTER_CLUSTER_MODE:-auto}"
SEQUENCE_CLUSTER_THREADS="${SEQUENCE_CLUSTER_THREADS:-4}"
SEQUENCE_CLUSTER_CACHE_DIR="${SEQUENCE_CLUSTER_CACHE_DIR:-}"
GROUPED_SPLIT_STRATEGIES="${GROUPED_SPLIT_STRATEGIES:-c2 c3}"
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
INCLUDE_SGD="${INCLUDE_SGD:-1}"
INCLUDE_TORCH_MLP="${INCLUDE_TORCH_MLP:-0}"
INCLUDE_PLM="${INCLUDE_PLM:-0}"
INCLUDE_LOW_RESOURCE_ESM2="${INCLUDE_LOW_RESOURCE_ESM2:-0}"
PLM_ADAPTER="${PLM_ADAPTER:-esm2}"
PLM_PRESETS="${PLM_PRESETS:-}"
PLM_MODEL="${PLM_MODEL:-facebook/esm2_t6_8M_UR50D}"
PLM_REVISION="${PLM_REVISION:-}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-}"
PLM_POOLING="${PLM_POOLING:-mean}"
PLM_DEVICE="${PLM_DEVICE:-cpu}"
PLM_PRECISION="${PLM_PRECISION:-float32}"
PLM_MAX_LENGTH="${PLM_MAX_LENGTH:-1024}"
PLM_TRUNCATION_POLICY="${PLM_TRUNCATION_POLICY:-truncate}"
PLM_MAX_BATCH_TOKENS="${PLM_MAX_BATCH_TOKENS:-1024}"
PLM_MAX_BATCH_SEQUENCES="${PLM_MAX_BATCH_SEQUENCES:-8}"

read -r -a PLM_PRESET_VALUES <<< "$PLM_PRESETS"

PASSTHROUGH_ARGS=()
while (($#)); do
    case "$1" in
        --include-low-resource-esm2)
            INCLUDE_LOW_RESOURCE_ESM2=1
            ;;
        --no-include-low-resource-esm2)
            INCLUDE_LOW_RESOURCE_ESM2=0
            ;;
        --include-plm)
            INCLUDE_PLM=1
            ;;
        --no-include-plm)
            INCLUDE_PLM=0
            ;;
        --include-sgd|--sgd)
            INCLUDE_SGD=1
            ;;
        --no-include-sgd|--no-sgd)
            INCLUDE_SGD=0
            ;;
        --include-torch-mlp)
            INCLUDE_TORCH_MLP=1
            ;;
        --no-include-torch-mlp)
            INCLUDE_TORCH_MLP=0
            ;;
        --include-sequence-cluster-splits)
            INCLUDE_SEQUENCE_CLUSTER_SPLITS=1
            ;;
        --no-include-sequence-cluster-splits)
            INCLUDE_SEQUENCE_CLUSTER_SPLITS=0
            ;;
        --plm-adapter)
            if (($# < 2)); then
                echo "--plm-adapter requires a value." >&2
                return 2
            fi
            PLM_ADAPTER="$2"
            shift
            ;;
        --plm-preset)
            if (($# < 2)); then
                echo "--plm-preset requires a value." >&2
                return 2
            fi
            PLM_PRESET_VALUES+=("$2")
            shift
            ;;
        --)
            shift
            PASSTHROUGH_ARGS+=("$@")
            break
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            ;;
    esac
    shift
done

LOW_RESOURCE_ESM2_ARGS=()
if [[ "$INCLUDE_LOW_RESOURCE_ESM2" == "1" \
        && "${#PLM_PRESET_VALUES[@]}" -gt 0 ]]; then
    echo "--include-low-resource-esm2 cannot be combined with --plm-preset." >&2
    return 2
fi
case "$INCLUDE_LOW_RESOURCE_ESM2" in
    1)
        INCLUDE_PLM=1
        if [[ "$PLM_ADAPTER" != "esm2" ]]; then
            echo "--include-low-resource-esm2 requires PLM_ADAPTER=esm2." >&2
            return 2
        fi
        if [[ -z "$PLM_REVISION" \
                && "$PLM_MODEL" == "facebook/esm2_t6_8M_UR50D" ]]; then
            PLM_REVISION="c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
        fi
        if [[ -z "$EMBEDDING_CACHE_DIR" ]]; then
            EMBEDDING_CACHE_DIR="results/embedding_cache"
        fi
        LOW_RESOURCE_ESM2_ARGS=(
            --plm-pooling "$PLM_POOLING"
            --plm-device "$PLM_DEVICE"
            --plm-precision "$PLM_PRECISION"
            --plm-max-length "$PLM_MAX_LENGTH"
            --plm-truncation-policy "$PLM_TRUNCATION_POLICY"
            --plm-max-batch-tokens "$PLM_MAX_BATCH_TOKENS"
            --plm-max-batch-sequences "$PLM_MAX_BATCH_SEQUENCES"
        )
        ;;
    0) ;;
    *)
        echo "INCLUDE_LOW_RESOURCE_ESM2 must be 0 or 1." >&2
        return 2
        ;;
esac
if [[ "${#PLM_PRESET_VALUES[@]}" -gt 0 ]]; then
    INCLUDE_PLM=1
fi
USER_ARGS=("${LOW_RESOURCE_ESM2_ARGS[@]}" "${PASSTHROUGH_ARGS[@]}")

read -r -a SPLIT_STRATEGY_VALUES <<< "$SPLIT_STRATEGIES"
read -r -a GROUPED_SPLIT_STRATEGY_VALUES <<< "$GROUPED_SPLIT_STRATEGIES"
read -r -a SPLIT_SEED_VALUES <<< "$SPLIT_SEEDS"
if [[ "${#SPLIT_STRATEGY_VALUES[@]}" -eq 0 ]]; then
    echo "SPLIT_STRATEGIES must contain at least one strategy." >&2
    return 2
fi
if [[ "${#SPLIT_SEED_VALUES[@]}" -eq 0 ]]; then
    echo "SPLIT_SEEDS must contain at least one integer seed." >&2
    return 2
fi
if [[ "${#GROUPED_SPLIT_STRATEGY_VALUES[@]}" -eq 0 ]]; then
    echo "GROUPED_SPLIT_STRATEGIES must contain at least one strategy." >&2
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

SUITE_ROOT="$OUT_DIR/$RUN_NAME"
IDENTITY_GRID_NAME="${IDENTITY_GRID_NAME:-identity}"
if [[ -z "${SEQUENCE_CLUSTER_GRID_NAME:-}" ]]; then
    if [[ -n "$SEQUENCE_CLUSTERS" ]]; then
        SEQUENCE_CLUSTER_GRID_NAME="supplied_sequence_clusters"
    else
        SEQUENCE_CLUSTER_GRID_NAME="mmseqs2_id${SEQUENCE_CLUSTER_MIN_SEQ_ID}_cov${SEQUENCE_CLUSTER_COVERAGE}_mode${SEQUENCE_CLUSTER_COV_MODE}"
    fi
fi

COMMON_GRID_ARGS=(
    --pairs "$PAIRS"
    --fasta "$FASTA"
    --profile "$BENCHMARK_PROFILE"
    --sampling-seed "$SAMPLING_SEED"
    --train-size "$TRAIN_SIZE"
    --val-size "$VAL_SIZE"
    --split-seeds "${SPLIT_SEED_VALUES[@]}"
    --model-seeds "${MODEL_SEED_VALUES[@]}"
    --max-iter "$MAX_ITER"
    --k "$K"
)

if [[ -n "$PROTEIN_METADATA" ]]; then
    COMMON_GRID_ARGS+=(--protein-metadata "$PROTEIN_METADATA")
fi
if [[ -v MAX_PAIRS ]]; then
    if [[ -n "$MAX_PAIRS" ]]; then
        COMMON_GRID_ARGS+=(--max-pairs "$MAX_PAIRS")
    else
        COMMON_GRID_ARGS+=(--full-cohort)
    fi
fi
if [[ -n "${N_SPLIT_TRIALS:-}" ]]; then
    COMMON_GRID_ARGS+=(--n-split-trials "$N_SPLIT_TRIALS")
fi

case "$AGGREGATE_RESULTS" in
    1|0) ;;
    *)
        echo "AGGREGATE_RESULTS must be 0 or 1." >&2
        return 2
        ;;
esac
case "$INCLUDE_TORCH_MLP" in
    1) COMMON_GRID_ARGS+=(--include-torch-mlp) ;;
    0) COMMON_GRID_ARGS+=(--no-include-torch-mlp) ;;
    *)
        echo "INCLUDE_TORCH_MLP must be 0 or 1." >&2
        return 2
        ;;
esac
case "$INCLUDE_SGD" in
    1) COMMON_GRID_ARGS+=(--include-sgd) ;;
    0) COMMON_GRID_ARGS+=(--no-include-sgd) ;;
    *)
        echo "INCLUDE_SGD must be 0 or 1." >&2
        return 2
        ;;
esac
case "$INCLUDE_PLM" in
    1)
        COMMON_GRID_ARGS+=(--include-plm)
        if [[ "${#PLM_PRESET_VALUES[@]}" -gt 0 ]]; then
            COMMON_GRID_ARGS+=(--plm-presets "${PLM_PRESET_VALUES[@]}")
        else
            if [[ -z "$PLM_REVISION" ]]; then
                echo "INCLUDE_PLM=1 requires an immutable PLM_REVISION." >&2
                return 2
            fi
            COMMON_GRID_ARGS+=(
                --plm-adapter "$PLM_ADAPTER"
                --plm-model "$PLM_MODEL"
                --plm-revision "$PLM_REVISION"
            )
        fi
        if [[ -n "$EMBEDDING_CACHE_DIR" ]]; then
            COMMON_GRID_ARGS+=(--embedding-cache-dir "$EMBEDDING_CACHE_DIR")
        fi
        ;;
    0) COMMON_GRID_ARGS+=(--no-include-plm) ;;
    *)
        echo "INCLUDE_PLM must be 0 or 1." >&2
        return 2
        ;;
esac

echo "Benchmark profile: $BENCHMARK_PROFILE"
echo "Split fractions: train=$TRAIN_SIZE, val=$VAL_SIZE, test=remainder"
echo "Split seeds: ${SPLIT_SEED_VALUES[*]}"
echo "Model seeds: ${MODEL_SEED_VALUES[*]}"
if [[ "$INCLUDE_LOW_RESOURCE_ESM2" == "1" ]]; then
    echo "Low-resource ESM-2: enabled (device=$PLM_DEVICE, token budget=$PLM_MAX_BATCH_TOKENS)"
fi
if [[ "${#PLM_PRESET_VALUES[@]}" -gt 0 ]]; then
    echo "PLM presets: ${PLM_PRESET_VALUES[*]}"
fi

GROUPING_GRID_ARGS=()
case "$INCLUDE_SEQUENCE_CLUSTER_SPLITS" in
    1)
        for split_strategy in "${GROUPED_SPLIT_STRATEGY_VALUES[@]}"; do
            if [[ "$split_strategy" != "c2" && "$split_strategy" != "c3" ]]; then
                echo "Grouped grids support only c2 and c3, not '$split_strategy'." >&2
                return 2
            fi
        done
        if [[ -n "$SEQUENCE_CLUSTERS" ]]; then
            GROUPING_GRID_ARGS+=(--sequence-clusters "$SEQUENCE_CLUSTERS")
            echo "Grouped C2/C3 source: $SEQUENCE_CLUSTERS"
        else
            if ! command -v mmseqs >/dev/null 2>&1; then
                echo "MMseqs2 grouped splits require 'mmseqs' on PATH. Set INCLUDE_SEQUENCE_CLUSTER_SPLITS=0 (or use --no-include-sequence-cluster-splits) for the dependency-free identity grid." >&2
                return 2
            fi
            GROUPING_GRID_ARGS+=(
                --sequence-cluster-method mmseqs2
                --sequence-cluster-min-seq-id "$SEQUENCE_CLUSTER_MIN_SEQ_ID"
                --sequence-cluster-coverage "$SEQUENCE_CLUSTER_COVERAGE"
                --sequence-cluster-cov-mode "$SEQUENCE_CLUSTER_COV_MODE"
                --sequence-cluster-evalue "$SEQUENCE_CLUSTER_EVALUE"
                --sequence-cluster-sensitivity "$SEQUENCE_CLUSTER_SENSITIVITY"
                --sequence-cluster-cluster-mode "$SEQUENCE_CLUSTER_CLUSTER_MODE"
                --sequence-cluster-threads "$SEQUENCE_CLUSTER_THREADS"
            )
            if [[ -n "$SEQUENCE_CLUSTER_CACHE_DIR" ]]; then
                GROUPING_GRID_ARGS+=(
                    --sequence-cluster-cache-dir "$SEQUENCE_CLUSTER_CACHE_DIR"
                )
            fi
            echo "Grouped C2/C3 source: MMseqs2 sequence clusters"
            echo "MMseqs2 parameters: id=$SEQUENCE_CLUSTER_MIN_SEQ_ID, coverage=$SEQUENCE_CLUSTER_COVERAGE, cov-mode=$SEQUENCE_CLUSTER_COV_MODE, evalue=$SEQUENCE_CLUSTER_EVALUE, sensitivity=$SEQUENCE_CLUSTER_SENSITIVITY, cluster-mode=$SEQUENCE_CLUSTER_CLUSTER_MODE, threads=$SEQUENCE_CLUSTER_THREADS"
        fi
        ;;
    0) ;;
    *)
        echo "INCLUDE_SEQUENCE_CLUSTER_SPLITS must be 0 or 1." >&2
        return 2
        ;;
esac

echo "Benchmark suite: $SUITE_ROOT"
echo "Identity grid: ${SPLIT_STRATEGY_VALUES[*]}"
ppi-grid \
    --out-dir "$SUITE_ROOT" \
    --run-name "$IDENTITY_GRID_NAME" \
    --split-strategies "${SPLIT_STRATEGY_VALUES[@]}" \
    --no-aggregate-results \
    "${COMMON_GRID_ARGS[@]}" \
    -- "${USER_ARGS[@]}"

if [[ "$INCLUDE_SEQUENCE_CLUSTER_SPLITS" == "1" ]]; then
    echo "Grouped grid: ${GROUPED_SPLIT_STRATEGY_VALUES[*]}"
    ppi-grid \
        --out-dir "$SUITE_ROOT" \
        --run-name "$SEQUENCE_CLUSTER_GRID_NAME" \
        --split-strategies "${GROUPED_SPLIT_STRATEGY_VALUES[@]}" \
        --no-aggregate-results \
        "${GROUPING_GRID_ARGS[@]}" \
        "${COMMON_GRID_ARGS[@]}" \
        -- "${USER_ARGS[@]}"
fi

if [[ "$AGGREGATE_RESULTS" == "1" ]]; then
    ppi-aggregate --benchmark-dir "$SUITE_ROOT"
fi
