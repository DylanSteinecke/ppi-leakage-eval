#!/usr/bin/env bash
set -euo pipefail

# Prepare and benchmark the local yeast BioGRID data:
# BENCHMARK_PROFILE=laptop bash scripts/run_yeast_biogrid_ppi_example.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

YEAST_DATASET_NAME="${YEAST_DATASET_NAME:-biogrid_yeast_physical}"
PAIRS="${PAIRS:-processed/$YEAST_DATASET_NAME/pairs.csv}"
FASTA="${FASTA:-processed/$YEAST_DATASET_NAME/proteins.fasta}"
PROTEIN_METADATA="${PROTEIN_METADATA:-processed/$YEAST_DATASET_NAME/protein_metadata.csv}"
OUT_DIR="${OUT_DIR:-results/${YEAST_DATASET_NAME}_example}"
MODEL_SEEDS="${MODEL_SEEDS:-0}"
MAX_ITER="${MAX_ITER:-100}"
K="${K:-2}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-laptop}"
SAMPLING_SEED="${SAMPLING_SEED:-17}"

BIOGRID_ARCHIVE="${BIOGRID_ARCHIVE:-input/BIOGRID-ORGANISM-LATEST.tab3.zip}"
BIOGRID_ARCHIVE_MEMBER="${BIOGRID_ARCHIVE_MEMBER:-BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-5.0.259.tab3.txt}"
YEAST_FASTA="${YEAST_FASTA:-input/UP000002311_559292.fasta}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
PREPARE_YEAST_DATA="${PREPARE_YEAST_DATA:-1}"

if [[ "$PREPARE_YEAST_DATA" == "1" ]]; then
    ppi-prepare biogrid \
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

source "$SCRIPT_DIR/_run_ppi_benchmark_grid.sh" "$@"
