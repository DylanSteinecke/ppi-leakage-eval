"""
Generate a synthetic PPI dataset for pipeline smoke tests.

This file creates a small, controlled dataset with multiple disconnected
protein components. Future additions should include richer sequence simulators,
harder negative sampling, configurable family structures, and optional missing
sequence examples for input-QC testing.
"""

import argparse
import bisect
import logging
import random
from pathlib import Path

import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
MOTIFS = ["ACDEFG", "KLMNPQ", "RSTVWY", "GHIKLM", "NPQRST"]
EXPECTED_LABEL_VALUES = {0, 1}
LOGGER = logging.getLogger(__name__)


#######
# CLI #
#######
def positive_int(value: str) -> int:
    """
    Parse a positive integer argparse value.
    """
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")

    return parsed_value


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for synthetic PPI data generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=positive_int, default=200)
    parser.add_argument("--n-pairs", type=positive_int, default=1000)
    parser.add_argument("--n-components", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs-out", default="processed/toy_pairs.csv")
    parser.add_argument("--fasta-out", default="processed/toy_sequences.fasta")
    args = parser.parse_args()

    return args


#######################
# Sequence generation #
#######################
def random_seq(length: int = 120) -> str:
    """
    Generate a random protein-like sequence.
    """
    sequence = "".join(random.choice(AA) for _ in range(length))

    return sequence


def insert_motif(seq: str, motif: str) -> str:
    """
    Insert a motif into a random position in a sequence.
    """
    i = random.randint(0, len(seq) - len(motif))
    sequence = seq[:i] + motif + seq[i + len(motif):]

    return sequence


########################
# Component generation #
########################
def get_component_sizes(n_proteins: int, n_components: int) -> list[int]:
    """
    Return balanced protein counts for disconnected components.
    """
    base_size = n_proteins // n_components
    remainder = n_proteins % n_components
    component_sizes = [
        base_size + int(component_index < remainder)
        for component_index in range(n_components)
    ]

    return component_sizes


def pair_capacity(component: list[str]) -> int:
    """
    Return the number of unique undirected pairs in one component.
    """
    n_proteins = len(component)
    capacity = n_proteins * (n_proteins - 1) // 2

    return capacity


def get_total_pair_capacity(components: list[list[str]]) -> int:
    """
    Return the total number of within-component pairs that can be sampled.
    """
    total_capacity = sum(pair_capacity(component) for component in components)

    return total_capacity


def validate_generation_args(args: argparse.Namespace) -> None:
    """
    Fail fast if the requested toy graph cannot be generated.
    """
    if args.n_components > args.n_proteins // 2:
        raise ValueError(
            "--n-components must be no greater than half of --n-proteins "
            "so each component can contain at least two proteins.")
    if args.n_pairs < args.n_components:
        raise ValueError(
            "--n-pairs must be at least --n-components so every component "
            "can have at least one observed protein pair.")

    component_sizes = get_component_sizes(
        n_proteins=args.n_proteins,
        n_components=args.n_components,
    )
    components = [
        [f"P{protein_index:04d}" for protein_index in range(component_size)]
        for component_size in component_sizes
    ]
    total_capacity = get_total_pair_capacity(components)
    if args.n_pairs > total_capacity:
        raise ValueError(
            f"--n-pairs={args.n_pairs} exceeds the within-component "
            f"capacity of {total_capacity} unique pairs.")


def make_components(n_proteins: int, n_components: int) -> list[list[str]]:
    """
    Create balanced protein IDs grouped by connected component.
    """
    component_sizes = get_component_sizes(n_proteins, n_components)
    components = []
    protein_index = 0
    for component_size in component_sizes:
        component = []
        for _ in range(component_size):
            protein_id = f"P{protein_index:04d}"
            component.append(protein_id)
            protein_index = protein_index + 1

        components.append(component)

    return components


def make_sequences_and_families(
        components: list[list[str]],
    ) -> tuple[dict[str, str], dict[str, int]]:
    """
    Generate sequences and motif-family labels for each protein.
    """
    seqs = {}
    families = {}
    for component_index, component in enumerate(components):
        for component_position, protein_id in enumerate(component):
            family = (component_position + component_index) % len(MOTIFS)
            sequence = insert_motif(random_seq(), MOTIFS[family])
            seqs[protein_id] = sequence
            families[protein_id] = family

    return seqs, families


###################
# Pair generation #
###################
def pair_from_component_index(
        component: list[str], pair_index: int,
    ) -> tuple[str, str]:
    """
    Return one pair from the lexicographic pair index within a component.
    """
    n_proteins = len(component)
    low = 0
    high = n_proteins - 1
    while low < high:
        mid = (low + high) // 2
        n_pairs_through_mid = (
            (mid + 1) * (2 * n_proteins - mid - 2) // 2
        )
        if n_pairs_through_mid <= pair_index:
            low = mid + 1
        else:
            high = mid

    left_index = low
    n_pairs_before_left = (
        left_index * (2 * n_proteins - left_index - 1) // 2
    )
    right_index = left_index + 1 + pair_index - n_pairs_before_left
    protein_a = component[left_index]
    protein_b = component[right_index]

    return protein_a, protein_b


def get_component_pair_offsets(
        components: list[list[str]],
    ) -> tuple[list[int], list[int]]:
    """
    Return cumulative pair offsets for mapping sampled pair indices.
    """
    capacities = [pair_capacity(component) for component in components]
    offsets = []
    running_total = 0
    for capacity in capacities:
        running_total = running_total + capacity
        offsets.append(running_total)

    return offsets, capacities


def sample_pair_indices(
        capacities: list[int], offsets: list[int], n_pairs: int,
    ) -> list[int]:
    """
    Sample within-component pair indices while keeping every component present.
    """
    reserved_indices = []
    previous_offset = 0
    for capacity in capacities:
        if capacity > 0:
            reserved_indices.append(previous_offset)
        previous_offset = previous_offset + capacity

    reserved_index_set = set(reserved_indices)
    total_capacity = offsets[-1]
    available_indices = [
        pair_index
        for pair_index in range(total_capacity)
        if pair_index not in reserved_index_set
    ]
    n_sampled_indices = n_pairs - len(reserved_indices)
    sampled_indices = random.sample(available_indices, n_sampled_indices)
    selected_indices = reserved_indices + sampled_indices
    random.shuffle(selected_indices)

    return selected_indices


def make_pair_rows(
        components: list[list[str]], families: dict[str, int], n_pairs: int,
    ) -> list[dict[str, int | str]]:
    """
    Generate labeled protein-pair rows within disconnected components.
    """
    offsets, capacities = get_component_pair_offsets(components)
    selected_indices = sample_pair_indices(
        capacities=capacities,
        offsets=offsets,
        n_pairs=n_pairs,
    )

    rows = []
    for pair_index in selected_indices:
        component_index = bisect.bisect_right(offsets, pair_index)
        if component_index == 0:
            component_offset = 0
        else:
            component_offset = offsets[component_index - 1]

        local_pair_index = pair_index - component_offset
        protein_a, protein_b = pair_from_component_index(
            components[component_index],
            local_pair_index,
        )

        # Toy rule: proteins from the same motif family interact.
        label = int(families[protein_a] == families[protein_b])
        rows.append({
            "protein_a": protein_a,
            "protein_b": protein_b,
            "label": label,
        })

    observed_labels = {row["label"] for row in rows}
    if observed_labels != EXPECTED_LABEL_VALUES:
        raise ValueError(
            "Toy data generation produced only one label. Increase "
            "--n-proteins, --n-pairs, or --n-components.")

    return rows


##################
# Output writing #
##################
def write_fasta(seqs: dict[str, str], fasta_path: Path) -> None:
    """
    Write protein sequences to FASTA.
    """
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as fout:
        for protein_id, sequence in seqs.items():
            fout.write(f">{protein_id}\n{sequence}\n")


def main() -> None:
    """
    Generate toy protein-pair labels and sequences.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = argument_parser()
    validate_generation_args(args)
    random.seed(args.seed)

    components = make_components(args.n_proteins, args.n_components)
    seqs, families = make_sequences_and_families(components)
    rows = make_pair_rows(components, families, args.n_pairs)

    pairs_path = Path(args.pairs_out)
    fasta_path = Path(args.fasta_out)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(pairs_path, index=False)
    write_fasta(seqs, fasta_path)

    LOGGER.info(
        f"Wrote {len(rows)} pairs across {args.n_components} components to "
        f"{pairs_path}"
    )
    LOGGER.info(f"Wrote {len(seqs)} protein sequences to {fasta_path}")


if __name__ == "__main__":
    main()

    # python scripts/make_toy_data.py
