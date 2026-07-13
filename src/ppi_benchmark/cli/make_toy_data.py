"""
Generate a synthetic PPI dataset for pipeline smoke tests.

This file creates a controlled dataset with realistic-ish graph structure,
sequence signal, hard negatives, label noise, and multiple disconnected
protein components. Future additions should include empirically calibrated
degree distributions, organism-specific sequence priors, and benchmark split
manifests.
"""

import argparse
import heapq
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
FAMILY_MOTIFS = (
    "ACDEFG",
    "KLMNPQ",
    "RSTVWY",
    "GHIKLM",
    "NPQRST",
    "WYACDE",
)
DOMAIN_MOTIFS = {
    "KINASE": "VAIKVL",
    "SH2": "FLVRRS",
    "SH3": "WYPQPV",
    "PDZ": "GLGFVI",
    "COIL": "LQKLEA",
    "LRR": "LxxLxL".replace("x", "A"),
    "PH": "KXWKRL".replace("X", "A"),
    "PRO": "PPAPPR",
    "RING": "CCHCCH",
    "ZINC": "CXXCXX".replace("X", "G"),
}
FAMILY_DOMAINS = (
    ("KINASE", "COIL", "PH"),
    ("SH2", "SH3", "PRO"),
    ("PDZ", "LRR", "ARM"),
    ("RING", "ZINC", "COIL"),
    ("PH", "ARM", "LRR"),
    ("PRO", "SH3", "ZINC"),
)
EXTRA_DOMAIN_MOTIFS = {
    "ARM": "HEATRL",
}
DOMAIN_MOTIFS.update(EXTRA_DOMAIN_MOTIFS)
COMPATIBLE_DOMAIN_PAIRS = {
    tuple(sorted(pair))
    for pair in (
        ("KINASE", "SH2"),
        ("SH3", "PRO"),
        ("PDZ", "ARM"),
        ("RING", "ZINC"),
        ("PH", "COIL"),
        ("LRR", "PH"),
        ("KINASE", "COIL"),
        ("PDZ", "PRO"),
    )
}
BACKGROUND_MOTIFS = ("GGSGGS", "DDEEKK", "STSTST")
COMPATIBLE_RELATION = "compatible"
HARD_NEGATIVE_RELATION = "hard_negative"
BACKGROUND_RELATION = "background_negative"
COMPONENT_PROFILE_CHOICES = ("balanced", "giant", "many_small")
EXPECTED_LABEL_VALUES = {0, 1}
LOGGER = logging.getLogger(__name__)


################
# Data classes #
################
@dataclass(frozen=True)
class ProteinRecord:
    """
    Synthetic protein metadata used to generate sequences and pairs.
    """

    protein_id: str
    component_id: int
    family_id: int
    domains: tuple[str, ...]
    is_hub: bool
    sequence: str


@dataclass(frozen=True)
class PairCandidate:
    """
    Candidate within-component protein pair before label assignment.
    """

    protein_a: str
    protein_b: str
    component_id: int
    relation: str
    latent_score: float
    sample_weight: float


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


def positive_float(value: str) -> float:
    """
    Parse a positive float argparse value.
    """
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")

    return parsed_value


def probability(value: str) -> float:
    """
    Parse a probability argparse value.
    """
    parsed_value = float(value)
    if (
            not math.isfinite(parsed_value)
            or parsed_value < 0.0
            or parsed_value > 1.0):
        raise argparse.ArgumentTypeError("value must be between 0 and 1")

    return parsed_value


def strict_probability(value: str) -> float:
    """
    Parse a probability that must leave room for both labels.
    """
    parsed_value = probability(value)
    if (parsed_value <= 0.0) or (parsed_value >= 1.0):
        raise argparse.ArgumentTypeError("value must be greater than 0 and "
                                         "less than 1")

    return parsed_value


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for synthetic PPI data generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=positive_int, default=200)
    parser.add_argument("--n-pairs", type=positive_int, default=1000)
    parser.add_argument("--n-components", type=positive_int, default=10)
    parser.add_argument(
        "--component-profile",
        choices=COMPONENT_PROFILE_CHOICES,
        default="giant",
        help="Shape of the disconnected component size distribution")
    parser.add_argument(
        "--positive-rate",
        type=strict_probability,
        default=0.15,
        help="Approximate fraction of observed positive labels")
    parser.add_argument(
        "--label-noise",
        type=probability,
        default=0.05,
        help="Probability of flipping each generated label")
    parser.add_argument(
        "--hard-negative-rate",
        type=probability,
        default=0.30,
        help="Approximate fraction of negatives that are hard negatives")
    parser.add_argument(
        "--hub-fraction",
        type=probability,
        default=0.05,
        help="Fraction of proteins sampled as higher-degree hubs")
    parser.add_argument(
        "--hub-pair-weight",
        type=positive_float,
        default=4.0,
        help="Sampling weight multiplier for pairs containing a hub")
    parser.add_argument("--min-seq-len", type=positive_int, default=80)
    parser.add_argument("--max-seq-len", type=positive_int, default=350)
    parser.add_argument(
        "--max-candidate-pairs",
        type=positive_int,
        default=2_000_000,
        help="Fail fast before enumerating too many within-component pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs-out", default="processed/toy_pairs.csv")
    parser.add_argument("--fasta-out", default="processed/toy_sequences.fasta")
    parser.add_argument(
        "--protein-meta-out",
        default="processed/toy_protein_metadata.csv",
        help="CSV file for synthetic protein metadata")
    args = parser.parse_args()

    return args


#######################
# Sequence generation #
#######################
def random_seq(length: int) -> str:
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


def make_sequence(
        family_id: int, domains: tuple[str, ...], is_hub: bool,
        min_seq_len: int, max_seq_len: int,
    ) -> str:
    """
    Generate one synthetic protein sequence with conserved motifs.
    """
    sequence_length = random.randint(min_seq_len, max_seq_len)
    sequence = random_seq(sequence_length)
    sequence = insert_motif(sequence, FAMILY_MOTIFS[family_id])

    for domain in domains:
        sequence = insert_motif(sequence, DOMAIN_MOTIFS[domain])

    if random.random() < 0.35:
        sequence = insert_motif(sequence, random.choice(BACKGROUND_MOTIFS))
    if is_hub:
        sequence = insert_motif(sequence, "GGGSGG")

    return sequence


########################
# Component generation #
########################
def get_profile_weights(
        n_components: int, component_profile: str,
    ) -> list[float]:
    """
    Return component size weights for a graph profile.
    """
    if component_profile == "balanced":
        weights = [1.0 for _ in range(n_components)]
    elif component_profile == "giant":
        weights = [4.0]
        weights.extend(1.0 / (index + 1) for index in range(1, n_components))
    elif component_profile == "many_small":
        weights = [2.2, 1.6, 1.2]
        weights.extend(0.35 for _ in range(max(0, n_components - 3)))
        weights = weights[:n_components]
    else:
        raise ValueError(f"Unknown component profile: {component_profile}")

    return weights


def get_component_sizes(
        n_proteins: int, n_components: int, component_profile: str,
    ) -> list[int]:
    """
    Return protein counts for disconnected components.
    """
    min_component_size = 2
    remaining_proteins = n_proteins - n_components * min_component_size
    weights = get_profile_weights(n_components, component_profile)
    weight_sum = sum(weights)
    raw_sizes = [
        weight * remaining_proteins / weight_sum
        for weight in weights
    ]
    component_sizes = [
        min_component_size + int(raw_size)
        for raw_size in raw_sizes
    ]
    n_leftover = n_proteins - sum(component_sizes)
    fractional_order = sorted(
        range(n_components),
        key=lambda index: raw_sizes[index] - int(raw_sizes[index]),
        reverse=True,
    )
    for index in fractional_order[:n_leftover]:
        component_sizes[index] = component_sizes[index] + 1

    return component_sizes


def min_connected_pair_count(component_sizes: list[int]) -> int:
    """
    Return the minimum observed pairs needed to connect each component.
    """
    min_pair_count = sum(component_size - 1
                         for component_size in component_sizes)

    return min_pair_count


def make_components(
        n_proteins: int, n_components: int, component_profile: str,
    ) -> list[list[str]]:
    """
    Create protein IDs grouped by disconnected component.
    """
    component_sizes = get_component_sizes(
        n_proteins=n_proteins,
        n_components=n_components,
        component_profile=component_profile,
    )
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


def validate_generation_args(args: argparse.Namespace) -> None:
    """
    Fail fast if the requested toy graph cannot be generated.
    """
    longest_motif = max(
        len(motif)
        for motif in (
            tuple(FAMILY_MOTIFS)
            + tuple(DOMAIN_MOTIFS.values())
            + tuple(BACKGROUND_MOTIFS)
        )
    )
    if args.max_seq_len < args.min_seq_len:
        raise ValueError("--max-seq-len must be at least --min-seq-len.")
    if args.min_seq_len < longest_motif:
        raise ValueError(
            f"--min-seq-len must be at least {longest_motif}.")
    if args.n_components > args.n_proteins // 2:
        raise ValueError(
            "--n-components must be no greater than half of --n-proteins "
            "so each component can contain at least two proteins.")

    component_sizes = get_component_sizes(
        n_proteins=args.n_proteins,
        n_components=args.n_components,
        component_profile=args.component_profile,
    )
    min_pair_count = min_connected_pair_count(component_sizes)
    total_capacity = sum(
        component_size * (component_size - 1) // 2
        for component_size in component_sizes
    )
    if args.n_pairs < min_pair_count:
        raise ValueError(
            f"--n-pairs must be at least {min_pair_count} to keep each "
            "requested component connected in the observed pair table.")
    if args.n_pairs > total_capacity:
        raise ValueError(
            f"--n-pairs={args.n_pairs} exceeds the within-component "
            f"capacity of {total_capacity} unique pairs.")
    if total_capacity > args.max_candidate_pairs:
        raise ValueError(
            f"The requested toy graph has {total_capacity} candidate pairs. "
            "Increase --max-candidate-pairs if you intend to enumerate them.")


####################
# Protein metadata #
####################
def choose_hub_ids(
        components: list[list[str]], hub_fraction: float,
    ) -> set[str]:
    """
    Choose proteins that receive higher sampling weights.
    """
    proteins = [protein for component in components for protein in component]
    n_hubs = round(len(proteins) * hub_fraction)
    if (hub_fraction > 0.0) and (n_hubs == 0):
        n_hubs = 1

    hub_ids = set()
    shuffled_components = components.copy()
    random.shuffle(shuffled_components)
    for component in shuffled_components:
        if len(hub_ids) >= n_hubs:
            break
        hub_ids.add(random.choice(component))

    remaining_proteins = [protein for protein in proteins
                          if protein not in hub_ids]
    if len(hub_ids) < n_hubs:
        n_extra_hubs = n_hubs - len(hub_ids)
        hub_ids.update(random.sample(remaining_proteins, n_extra_hubs))

    return hub_ids


def choose_domains(family_id: int) -> tuple[str, ...]:
    """
    Choose one to three domains for a synthetic protein.
    """
    preferred_domains = list(FAMILY_DOMAINS[family_id])
    random.shuffle(preferred_domains)
    domains = [preferred_domains[0]]
    if random.random() < 0.65:
        domains.append(preferred_domains[1])
    if random.random() < 0.25:
        domains.append(random.choice(tuple(DOMAIN_MOTIFS)))

    unique_domains = tuple(dict.fromkeys(domains))

    return unique_domains


def make_protein_records(
        components: list[list[str]], args: argparse.Namespace,
    ) -> dict[str, ProteinRecord]:
    """
    Generate synthetic metadata and sequences for each protein.
    """
    hub_ids = choose_hub_ids(components, args.hub_fraction)
    protein_records = {}
    for component_id, component in enumerate(components):
        for component_position, protein_id in enumerate(component):
            family_id = (
                component_position + component_id
            ) % len(FAMILY_MOTIFS)
            domains = choose_domains(family_id)
            sequence = make_sequence(
                family_id=family_id,
                domains=domains,
                is_hub=protein_id in hub_ids,
                min_seq_len=args.min_seq_len,
                max_seq_len=args.max_seq_len,
            )
            protein_records[protein_id] = ProteinRecord(
                protein_id=protein_id,
                component_id=component_id,
                family_id=family_id,
                domains=domains,
                is_hub=protein_id in hub_ids,
                sequence=sequence,
            )

    return protein_records


def protein_metadata_rows(
        protein_records: dict[str, ProteinRecord],
    ) -> list[dict[str, int | str | bool]]:
    """
    Return protein metadata rows for optional inspection.
    """
    rows = []
    for record in protein_records.values():
        rows.append({
            "protein_id": record.protein_id,
            "component_id": record.component_id,
            "family_id": record.family_id,
            "domains": ";".join(record.domains),
            "is_hub": record.is_hub,
            "sequence_length": len(record.sequence),
        })

    return rows


###################
# Pair generation #
###################
def pair_key(protein_a: str, protein_b: str) -> tuple[str, str]:
    """
    Return a stable undirected pair key.
    """
    key = tuple(sorted((protein_a, protein_b)))

    return key


def domains_are_compatible(
        domains_a: tuple[str, ...], domains_b: tuple[str, ...],
    ) -> bool:
    """
    Return whether any cross-protein domain pair is compatible.
    """
    is_compatible = False
    for domain_a in domains_a:
        for domain_b in domains_b:
            domain_pair = tuple(sorted((domain_a, domain_b)))
            if domain_pair in COMPATIBLE_DOMAIN_PAIRS:
                is_compatible = True
                break
        if is_compatible:
            break

    return is_compatible


def classify_pair_relation(
        record_a: ProteinRecord, record_b: ProteinRecord,
    ) -> str:
    """
    Classify a pair as compatible, hard negative, or background negative.
    """
    shared_domains = set(record_a.domains) & set(record_b.domains)
    is_compatible = domains_are_compatible(record_a.domains, record_b.domains)
    same_family = record_a.family_id == record_b.family_id
    if is_compatible:
        relation = COMPATIBLE_RELATION
    elif same_family or shared_domains:
        relation = HARD_NEGATIVE_RELATION
    else:
        relation = BACKGROUND_RELATION

    return relation


def score_pair(
        record_a: ProteinRecord, record_b: ProteinRecord, relation: str,
    ) -> float:
    """
    Return a latent interaction score before label noise.
    """
    shared_domains = set(record_a.domains) & set(record_b.domains)
    score = random.random() * 0.05
    if relation == COMPATIBLE_RELATION:
        score = score + 3.0
    if relation == HARD_NEGATIVE_RELATION:
        score = score + 1.5
    if record_a.family_id == record_b.family_id:
        score = score + 0.6
    if shared_domains:
        score = score + 0.5
    if record_a.is_hub or record_b.is_hub:
        score = score + 0.2

    return score


def get_pair_sample_weight(
        record_a: ProteinRecord, record_b: ProteinRecord, relation: str,
        hub_pair_weight: float,
    ) -> float:
    """
    Return the sampling weight for a candidate pair.
    """
    sample_weight = 1.0
    if record_a.is_hub or record_b.is_hub:
        sample_weight = sample_weight * hub_pair_weight
    if relation == COMPATIBLE_RELATION:
        sample_weight = sample_weight * 2.5
    elif relation == HARD_NEGATIVE_RELATION:
        sample_weight = sample_weight * 2.0

    return sample_weight


def make_pair_candidate(
        protein_a: str, protein_b: str,
        protein_records: dict[str, ProteinRecord],
        hub_pair_weight: float,
    ) -> PairCandidate:
    """
    Create one within-component pair candidate.
    """
    record_a = protein_records[protein_a]
    record_b = protein_records[protein_b]
    relation = classify_pair_relation(record_a, record_b)
    latent_score = score_pair(record_a, record_b, relation)
    sample_weight = get_pair_sample_weight(
        record_a=record_a,
        record_b=record_b,
        relation=relation,
        hub_pair_weight=hub_pair_weight,
    )
    candidate = PairCandidate(
        protein_a=protein_a,
        protein_b=protein_b,
        component_id=record_a.component_id,
        relation=relation,
        latent_score=latent_score,
        sample_weight=sample_weight,
    )

    return candidate


def make_pair_candidates(
        components: list[list[str]],
        protein_records: dict[str, ProteinRecord],
        hub_pair_weight: float,
    ) -> dict[tuple[str, str], PairCandidate]:
    """
    Enumerate within-component candidate pairs.
    """
    pair_candidates = {}
    for component in components:
        for left_index, protein_a in enumerate(component[:-1]):
            for protein_b in component[left_index + 1:]:
                key = pair_key(protein_a, protein_b)
                pair_candidates[key] = make_pair_candidate(
                    protein_a=protein_a,
                    protein_b=protein_b,
                    protein_records=protein_records,
                    hub_pair_weight=hub_pair_weight,
                )

    return pair_candidates


def component_anchor(
        component: list[str], protein_records: dict[str, ProteinRecord],
    ) -> str:
    """
    Return the hub protein used to connect an observed component.
    """
    hub_proteins = [
        protein_id for protein_id in component
        if protein_records[protein_id].is_hub
    ]
    if hub_proteins:
        anchor = sorted(hub_proteins)[0]
    else:
        anchor = component[0]

    return anchor


def get_connectivity_keys(
        components: list[list[str]],
        protein_records: dict[str, ProteinRecord],
    ) -> set[tuple[str, str]]:
    """
    Return required pairs that keep each designed component connected.
    """
    connectivity_keys = set()
    for component in components:
        anchor = component_anchor(component, protein_records)
        for protein_id in component:
            if protein_id != anchor:
                connectivity_keys.add(pair_key(anchor, protein_id))

    return connectivity_keys


def weighted_sample_without_replacement(
        candidates: list[PairCandidate], n_candidates: int,
    ) -> list[PairCandidate]:
    """
    Sample candidates without replacement using positive weights.
    """
    n_candidates = min(n_candidates, len(candidates))
    if n_candidates <= 0:
        return []
    if n_candidates == len(candidates):
        return candidates.copy()

    selected_candidates = heapq.nsmallest(
        n_candidates,
        candidates,
        key=lambda candidate: (
            -math.log1p(-random.random()) / candidate.sample_weight
        ),
    )

    return selected_candidates


def sample_relation_candidates(
        candidate_lookup: dict[tuple[str, str], PairCandidate],
        selected_keys: set[tuple[str, str]], relation: str,
        n_candidates: int,
    ) -> list[PairCandidate]:
    """
    Sample available candidates from one relation class.
    """
    relation_candidates = [
        candidate
        for key, candidate in candidate_lookup.items()
        if (key not in selected_keys) and (candidate.relation == relation)
    ]
    selected_candidates = weighted_sample_without_replacement(
        candidates=relation_candidates,
        n_candidates=n_candidates,
    )

    return selected_candidates


def sample_remaining_candidates(
        candidate_lookup: dict[tuple[str, str], PairCandidate],
        selected_keys: set[tuple[str, str]], n_candidates: int,
    ) -> list[PairCandidate]:
    """
    Sample remaining candidates from all relation classes.
    """
    remaining_candidates = [
        candidate
        for key, candidate in candidate_lookup.items()
        if key not in selected_keys
    ]
    selected_candidates = weighted_sample_without_replacement(
        candidates=remaining_candidates,
        n_candidates=n_candidates,
    )

    return selected_candidates


def add_selected_candidates(
        selected_candidates: list[PairCandidate],
        selected_keys: set[tuple[str, str]],
        new_candidates: list[PairCandidate],
    ) -> None:
    """
    Add candidates to the selected list and selected-key set.
    """
    for candidate in new_candidates:
        selected_candidates.append(candidate)
        selected_keys.add(pair_key(candidate.protein_a, candidate.protein_b))


def select_pair_candidates(
        candidate_lookup: dict[tuple[str, str], PairCandidate],
        connectivity_keys: set[tuple[str, str]], args: argparse.Namespace,
    ) -> list[PairCandidate]:
    """
    Select observed pair candidates with positives and hard negatives.
    """
    selected_candidates = []
    selected_keys = set()
    connectivity_candidates = [
        candidate_lookup[key] for key in connectivity_keys
    ]
    add_selected_candidates(
        selected_candidates=selected_candidates,
        selected_keys=selected_keys,
        new_candidates=connectivity_candidates,
    )

    target_n_positive = round(args.n_pairs * args.positive_rate)
    target_n_negative = args.n_pairs - target_n_positive
    target_n_hard_negative = round(
        target_n_negative * args.hard_negative_rate)
    current_n_compatible = sum(
        candidate.relation == COMPATIBLE_RELATION
        for candidate in selected_candidates
    )
    current_n_hard_negative = sum(
        candidate.relation == HARD_NEGATIVE_RELATION
        for candidate in selected_candidates
    )

    remaining_budget = args.n_pairs - len(selected_candidates)
    n_compatible_needed = max(0, target_n_positive - current_n_compatible)
    n_compatible_needed = min(n_compatible_needed, remaining_budget)
    compatible_candidates = sample_relation_candidates(
        candidate_lookup=candidate_lookup,
        selected_keys=selected_keys,
        relation=COMPATIBLE_RELATION,
        n_candidates=n_compatible_needed,
    )
    add_selected_candidates(
        selected_candidates=selected_candidates,
        selected_keys=selected_keys,
        new_candidates=compatible_candidates,
    )
    if len(compatible_candidates) < n_compatible_needed:
        LOGGER.warning(
            f"Requested {n_compatible_needed} extra compatible pairs but "
            f"only sampled {len(compatible_candidates)}.")

    remaining_budget = args.n_pairs - len(selected_candidates)
    n_hard_needed = max(0, target_n_hard_negative
                        - current_n_hard_negative)
    n_hard_needed = min(n_hard_needed, remaining_budget)
    hard_candidates = sample_relation_candidates(
        candidate_lookup=candidate_lookup,
        selected_keys=selected_keys,
        relation=HARD_NEGATIVE_RELATION,
        n_candidates=n_hard_needed,
    )
    add_selected_candidates(
        selected_candidates=selected_candidates,
        selected_keys=selected_keys,
        new_candidates=hard_candidates,
    )
    if len(hard_candidates) < n_hard_needed:
        LOGGER.warning(
            f"Requested {n_hard_needed} extra hard negatives but only "
            f"sampled {len(hard_candidates)}.")

    n_remaining = max(0, args.n_pairs - len(selected_candidates))
    remaining_candidates = sample_remaining_candidates(
        candidate_lookup=candidate_lookup,
        selected_keys=selected_keys,
        n_candidates=n_remaining,
    )
    add_selected_candidates(
        selected_candidates=selected_candidates,
        selected_keys=selected_keys,
        new_candidates=remaining_candidates,
    )
    random.shuffle(selected_candidates)

    return selected_candidates


####################
# Label assignment #
####################
def label_edge_type(
        candidate: PairCandidate, label: int, label_was_flipped: bool,
    ) -> str:
    """
    Return a label provenance category for one observed pair.
    """
    if label_was_flipped:
        if label == 1:
            edge_type = "noisy_positive"
        else:
            edge_type = "noisy_negative"
    elif label == 1:
        if candidate.relation == COMPATIBLE_RELATION:
            edge_type = "compatible_positive"
        else:
            edge_type = "weak_positive"
    elif candidate.relation == HARD_NEGATIVE_RELATION:
        edge_type = "hard_negative"
    else:
        edge_type = "background_negative"

    return edge_type


def set_row_label(
        row: dict[str, int | float | str | bool], label: int,
        edge_type: str,
    ) -> None:
    """
    Update the observed label and edge type for a row.
    """
    row["label"] = label
    row["edge_type"] = edge_type


def ensure_global_label_diversity(
        rows: list[dict[str, int | float | str | bool]],
    ) -> None:
    """
    Force both labels to be present if noise removed one class.
    """
    observed_labels = {int(row["label"]) for row in rows}
    if observed_labels == EXPECTED_LABEL_VALUES:
        return

    sorted_rows = sorted(rows, key=lambda row: float(row["latent_score"]))
    if 0 not in observed_labels:
        set_row_label(
            row=sorted_rows[0],
            label=0,
            edge_type="background_negative",
        )
    if 1 not in observed_labels:
        set_row_label(
            row=sorted_rows[-1],
            label=1,
            edge_type="compatible_positive",
        )


def ensure_component_label_diversity(
        rows: list[dict[str, int | float | str | bool]],
    ) -> None:
    """
    Give multi-pair components both labels when possible.
    """
    rows_by_component = {}
    for row in rows:
        rows_by_component.setdefault(row["component_id"], []).append(row)

    for component_rows in rows_by_component.values():
        if len(component_rows) < 2:
            continue

        observed_labels = {int(row["label"]) for row in component_rows}
        if observed_labels == EXPECTED_LABEL_VALUES:
            continue

        sorted_rows = sorted(
            component_rows,
            key=lambda row: float(row["latent_score"]),
        )
        if 0 not in observed_labels:
            set_row_label(
                row=sorted_rows[0],
                label=0,
                edge_type="background_negative",
            )
        if 1 not in observed_labels:
            set_row_label(
                row=sorted_rows[-1],
                label=1,
                edge_type="compatible_positive",
            )


def make_pair_rows(
        selected_candidates: list[PairCandidate], args: argparse.Namespace,
    ) -> list[dict[str, int | float | str | bool]]:
    """
    Assign labels and metadata to selected pair candidates.
    """
    target_n_positive = round(len(selected_candidates) * args.positive_rate)
    ranked_candidates = sorted(
        selected_candidates,
        key=lambda candidate: candidate.latent_score,
        reverse=True,
    )
    positive_keys = {
        pair_key(candidate.protein_a, candidate.protein_b)
        for candidate in ranked_candidates[:target_n_positive]
    }

    rows = []
    for candidate in selected_candidates:
        key = pair_key(candidate.protein_a, candidate.protein_b)
        label = int(key in positive_keys)
        label_was_flipped = False
        if random.random() < args.label_noise:
            label = 1 - label
            label_was_flipped = True

        edge_type = label_edge_type(candidate, label, label_was_flipped)
        rows.append({
            "protein_a": candidate.protein_a,
            "protein_b": candidate.protein_b,
            "label": label,
            "component_id": candidate.component_id,
            "pair_relation": candidate.relation,
            "edge_type": edge_type,
            "latent_score": round(candidate.latent_score, 6),
            "label_was_flipped": label_was_flipped,
        })

    ensure_global_label_diversity(rows)
    ensure_component_label_diversity(rows)
    random.shuffle(rows)

    return rows


##################
# Output writing #
##################
def write_fasta(
        protein_records: dict[str, ProteinRecord], fasta_path: Path,
    ) -> None:
    """
    Write protein sequences to FASTA.
    """
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as fout:
        for record in protein_records.values():
            fout.write(f">{record.protein_id}\n{record.sequence}\n")


def write_csv(rows: list[dict], output_path: Path) -> None:
    """
    Write dictionary rows to a CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


###################
# Summary logging #
###################
def label_summary(rows: list[dict[str, int | float | str | bool]]) -> str:
    """
    Return a compact label-count summary.
    """
    label_counts = pd.Series([row["label"] for row in rows]).value_counts()
    label_counts = label_counts.sort_index().to_dict()
    summary = str(label_counts)

    return summary


def relation_summary(rows: list[dict[str, int | float | str | bool]]) -> str:
    """
    Return a compact pair-relation summary.
    """
    relation_counts = pd.Series(
        [row["pair_relation"] for row in rows]
    ).value_counts()
    relation_counts = relation_counts.sort_index().to_dict()
    summary = str(relation_counts)

    return summary


def log_generation_summary(
        rows: list[dict[str, int | float | str | bool]],
        components: list[list[str]], protein_records: dict[str, ProteinRecord],
        args: argparse.Namespace, pairs_path: Path,
    ) -> None:
    """
    Log the generated toy dataset structure.
    """
    component_sizes = [len(component) for component in components]
    n_hubs = sum(record.is_hub for record in protein_records.values())
    positive_rate = sum(int(row["label"]) for row in rows) / len(rows)
    LOGGER.info(
        f"Wrote {len(rows)} pairs across {len(components)} components to "
        f"{pairs_path}"
    )
    LOGGER.info(
        f"Component profile={args.component_profile}; sizes="
        f"{component_sizes}"
    )
    LOGGER.info(
        f"Hubs={n_hubs}; observed positive rate={positive_rate:.3f}; "
        f"labels={label_summary(rows)}"
    )
    LOGGER.info(f"Pair relations={relation_summary(rows)}")


def main() -> None:
    """
    Generate toy protein-pair labels, sequences, and metadata.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = argument_parser()
    validate_generation_args(args)
    random.seed(args.seed)

    components = make_components(
        n_proteins=args.n_proteins,
        n_components=args.n_components,
        component_profile=args.component_profile,
    )
    protein_records = make_protein_records(components, args)
    candidate_lookup = make_pair_candidates(
        components=components,
        protein_records=protein_records,
        hub_pair_weight=args.hub_pair_weight,
    )
    connectivity_keys = get_connectivity_keys(components, protein_records)
    selected_candidates = select_pair_candidates(
        candidate_lookup=candidate_lookup,
        connectivity_keys=connectivity_keys,
        args=args,
    )
    rows = make_pair_rows(selected_candidates, args)

    pairs_path = Path(args.pairs_out)
    fasta_path = Path(args.fasta_out)
    protein_meta_path = Path(args.protein_meta_out)
    write_csv(rows, pairs_path)
    write_fasta(protein_records, fasta_path)
    write_csv(protein_metadata_rows(protein_records), protein_meta_path)

    log_generation_summary(
        rows=rows,
        components=components,
        protein_records=protein_records,
        args=args,
        pairs_path=pairs_path,
    )
    LOGGER.info(f"Wrote {len(protein_records)} protein sequences to "
                f"{fasta_path}")
    LOGGER.info(f"Wrote protein metadata to {protein_meta_path}")


if __name__ == "__main__":
    main()
