#!/usr/bin/env bash
set -euo pipefail

# Generate and benchmark synthetic PPI data:
# bash scripts/run_toy_ppi_example.sh --no-metrics-plots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PAIRS="${PAIRS:-processed/toy_pairs.csv}"
FASTA="${FASTA:-processed/toy_sequences.fasta}"
PROTEIN_METADATA="${PROTEIN_METADATA:-processed/toy_protein_metadata.csv}"
OUT_DIR="${OUT_DIR:-results/toy_all_models}"
NUM_RERUNS="${NUM_RERUNS:-5}"
MAX_ITER="${MAX_ITER:-10000}"
K="${K:-3}"
GENERATE_TOY_DATA="${GENERATE_TOY_DATA:-1}"

if [[ "$GENERATE_TOY_DATA" == "1" ]]; then
    mkdir -p "$(dirname "$PAIRS")" "$(dirname "$FASTA")" \
        "$(dirname "$PROTEIN_METADATA")"
    ppi-make-toy-data \
        --pairs-out "$PAIRS" \
        --fasta-out "$FASTA" \
        --protein-meta-out "$PROTEIN_METADATA"
fi

source "$SCRIPT_DIR/_run_ppi_benchmark_grid.sh" "$@"
