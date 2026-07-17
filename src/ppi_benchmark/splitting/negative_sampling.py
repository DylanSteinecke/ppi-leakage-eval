"""Explicit negative-construction contracts for PPI datasets."""

import math
import random
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterator, Literal

import pandas as pd

from ..datasets.pairs import prepare_pair_columns, unordered_pair_key


TAXON_PAIR_MATCHED_POLICY = "taxon_pair_matched"
GLOBAL_POLICY = "global"
SOURCE_PROVIDED_POLICY = "source_provided"
NEGATIVE_SAMPLING_POLICY_CHOICES = (
    TAXON_PAIR_MATCHED_POLICY,
    GLOBAL_POLICY,
)
NegativeSamplingPolicy = Literal["taxon_pair_matched", "global"]


def validate_negative_ratio(negative_ratio: float) -> float:
    """Return a finite, positive negative-to-positive ratio."""
    if isinstance(negative_ratio, bool) or not isinstance(negative_ratio, Real):
        raise ValueError("negative_ratio must be finite and greater than 0.")
    negative_ratio = float(negative_ratio)
    if not math.isfinite(negative_ratio) or negative_ratio <= 0.0:
        raise ValueError("negative_ratio must be finite and greater than 0.")

    return negative_ratio


@dataclass(frozen=True)
class PPINegativeSamplingSpec:
    """Configuration for generated, unlabeled PPI negative pairs."""

    policy: NegativeSamplingPolicy = TAXON_PAIR_MATCHED_POLICY
    negative_ratio: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.policy not in NEGATIVE_SAMPLING_POLICY_CHOICES:
            raise ValueError(
                f"Unknown negative sampling policy {self.policy!r}. "
                f"Expected one of {list(NEGATIVE_SAMPLING_POLICY_CHOICES)}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("negative sampling seed must be an integer.")
        object.__setattr__(
            self,
            "negative_ratio",
            validate_negative_ratio(self.negative_ratio),
        )


@dataclass(frozen=True)
class PPINegativeSamplingResult:
    """Generated negative pairs and their audit metadata."""

    pairs: pd.DataFrame
    metadata: dict[str, Any]

    def __iter__(self) -> Iterator[Any]:
        """Preserve tuple unpacking used by existing callers."""
        yield self.pairs
        yield self.metadata


def generated_negative_construction(
    spec: PPINegativeSamplingSpec,
    sampling_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return canonical provenance for generated negative pairs."""
    n_positive = int(sampling_metadata["n_positive_pairs_for_sampling"])
    n_negative = int(sampling_metadata["n_sampled_negatives"])
    taxon_composition = sampling_metadata.get("taxon_composition", {})
    return {
        "label_meaning": "sampled_unobserved_pair",
        "policy": spec.policy,
        "negative_ratio_requested": float(spec.negative_ratio),
        "negative_ratio_realized": n_negative / n_positive,
        "seed": int(spec.seed),
        "timing": "before_split",
        "partition_aware": False,
        "candidate_universe": {
            "prediction_unit": "unordered_non_self_protein_pair",
            "protein_scope": (
                "endpoints_of_observed_positive_pairs_with_fasta_sequences"
            ),
            "excluded_pairs": "observed_positive_pairs_in_input_snapshot",
            "n_proteins": int(sampling_metadata["n_proteins_for_sampling"]),
            "n_observed_positive_pairs": n_positive,
            "n_available_unobserved_pairs": int(
                sampling_metadata["n_available_negative_pairs"]
            ),
        },
        "n_sampled_negatives": n_negative,
        "taxon_composition": {
            "positive_pairs": taxon_composition.get("positive_pairs", {}),
            "negative_pairs": taxon_composition.get(
                "sampled_negative_pairs",
                {},
            ),
        },
    }


def target_negative_count(n_positive: int, negative_ratio: float) -> int:
    """Return the deterministic requested negative count."""
    negative_ratio = validate_negative_ratio(negative_ratio)
    if n_positive <= 0:
        raise ValueError("Cannot sample negatives without positive pairs.")
    return int(max(1, round(n_positive * negative_ratio)))


def require_taxonomy_for_negative_sampling(
    spec: PPINegativeSamplingSpec,
    protein_taxa: dict[str, str],
) -> None:
    """Fail early when matched sampling has no taxonomy metadata."""
    if spec.policy == TAXON_PAIR_MATCHED_POLICY and not protein_taxa:
        raise ValueError(
            "taxon_pair_matched negative sampling requires taxonomy "
            "metadata. Provide --protein-metadata, --taxon-id, FASTA OX= "
            "fields, or use --negative-sampling-policy global."
        )


def _pair_keys_from_prepared(
    pairs: pd.DataFrame,
) -> set[tuple[str, str]]:
    """Return unordered non-self keys from normalized pair columns."""
    return {
        unordered_pair_key(protein_a, protein_b)
        for protein_a, protein_b in zip(
            pairs["protein_a"],
            pairs["protein_b"],
        )
        if protein_a != protein_b
    }


def observed_pair_keys(pairs: pd.DataFrame) -> set[tuple[str, str]]:
    """Return non-self unordered pair keys from a pair table."""
    return _pair_keys_from_prepared(prepare_pair_columns(pairs))


def sample_complement_indexes(
    n_candidates: int,
    forbidden_indexes: Iterable[int],
    n_samples: int,
    rng: random.Random,
) -> list[int]:
    """Sample indexes without materializing the allowed complement."""
    forbidden = sorted(set(forbidden_indexes))
    n_available = n_candidates - len(forbidden)
    if n_samples > n_available:
        raise ValueError(
            f"Requested {n_samples} samples from {n_available} candidates."
        )

    sampled_ranks = rng.sample(range(n_available), n_samples)
    adjusted_forbidden = [
        forbidden_index - rank
        for rank, forbidden_index in enumerate(forbidden)
    ]
    return [
        rank + bisect_right(adjusted_forbidden, rank)
        for rank in sampled_ranks
    ]


class IndexedPairSpace:
    """Indexed combinations or Cartesian products of protein IDs."""

    def __init__(
        self,
        protein_a_ids: list[str],
        protein_b_ids: list[str] | None = None,
    ) -> None:
        self.protein_a_ids = protein_a_ids
        self.protein_b_ids = protein_b_ids
        self.protein_a_indexes = {
            protein_id: index
            for index, protein_id in enumerate(protein_a_ids)
        }
        if protein_b_ids is None:
            n_proteins = len(protein_a_ids)
            self.offsets = [
                index * (2 * n_proteins - index - 1) // 2
                for index in range(n_proteins)
            ]
            self.protein_b_indexes = None
            self.n_candidates = n_proteins * (n_proteins - 1) // 2
        else:
            self.offsets = None
            self.protein_b_indexes = {
                protein_id: index
                for index, protein_id in enumerate(protein_b_ids)
            }
            self.n_candidates = len(protein_a_ids) * len(protein_b_ids)

    def pair_index(self, protein_a: str, protein_b: str) -> int:
        """Encode one candidate pair as an integer index."""
        if self.protein_b_ids is None:
            item_a_index = self.protein_a_indexes[protein_a]
            item_b_index = self.protein_a_indexes[protein_b]
            if item_a_index > item_b_index:
                item_a_index, item_b_index = item_b_index, item_a_index
            assert self.offsets is not None
            return (
                self.offsets[item_a_index]
                + item_b_index
                - item_a_index
                - 1
            )

        if protein_a in self.protein_a_indexes:
            left_id, right_id = protein_a, protein_b
        else:
            left_id, right_id = protein_b, protein_a
        assert self.protein_b_indexes is not None
        return (
            self.protein_a_indexes[left_id] * len(self.protein_b_ids)
            + self.protein_b_indexes[right_id]
        )

    def pair_from_index(self, pair_index: int) -> tuple[str, str]:
        """Decode one integer index as a canonical protein pair."""
        if self.protein_b_ids is None:
            assert self.offsets is not None
            item_a_index = bisect_right(self.offsets, pair_index) - 1
            item_b_index = (
                item_a_index + 1 + pair_index - self.offsets[item_a_index]
            )
            return (
                self.protein_a_ids[item_a_index],
                self.protein_a_ids[item_b_index],
            )

        item_a_index, item_b_index = divmod(
            pair_index,
            len(self.protein_b_ids),
        )
        return unordered_pair_key(
            self.protein_a_ids[item_a_index],
            self.protein_b_ids[item_b_index],
        )


def sample_pair_space(
    pair_space: IndexedPairSpace,
    forbidden_pairs: set[tuple[str, str]],
    n_samples: int,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], int]:
    """Sample an indexed pair space while excluding observed pairs."""
    forbidden_indexes = [
        pair_space.pair_index(*pair)
        for pair in forbidden_pairs
    ]
    sampled_indexes = sample_complement_indexes(
        n_candidates=pair_space.n_candidates,
        forbidden_indexes=forbidden_indexes,
        n_samples=n_samples,
        rng=rng,
    )
    return (
        [pair_space.pair_from_index(index) for index in sampled_indexes],
        pair_space.n_candidates - len(forbidden_pairs),
    )


def allocate_stratified_samples(
    weights: dict[tuple[str, str], int],
    capacities: dict[tuple[str, str], int],
    target_count: int,
) -> dict[tuple[str, str], int]:
    """Allocate an exact sample target proportionally across bounded strata."""
    allocations = {key: 0 for key in weights}
    remaining = target_count
    while remaining:
        active = [
            key for key in sorted(weights)
            if allocations[key] < capacities[key]
        ]
        if not active:
            raise ValueError(
                f"Cannot allocate {target_count} samples across the available "
                "strata."
            )

        total_weight = sum(weights[key] for key in active)
        quotas = {
            key: remaining * weights[key] / total_weight
            for key in active
        }
        grants = {
            key: min(
                capacities[key] - allocations[key],
                int(quotas[key]),
            )
            for key in active
        }
        n_granted = sum(grants.values())
        if n_granted == 0:
            ranked_keys = sorted(active, key=lambda key: (-quotas[key], key))
            for key in ranked_keys[:remaining]:
                grants[key] = 1
            n_granted = sum(grants.values())

        for key, count in grants.items():
            allocations[key] += count
        remaining -= n_granted
    return allocations


def taxon_pair_key(taxon_a: str, taxon_b: str) -> tuple[str, str]:
    """Return a canonical unordered taxonomy-pair key."""
    return tuple(sorted((taxon_a, taxon_b)))


def taxon_pair_name(taxon_pair: tuple[str, str]) -> str:
    """Return a compact JSON key for one taxonomy-pair stratum."""
    return "|".join(taxon_pair)


def sample_taxon_stratified_negative_keys(
    protein_ids: list[str],
    positive_keys: set[tuple[str, str]],
    protein_taxa: dict[str, str],
    target_count: int,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    """Sample within taxonomy-pair strata represented by positive pairs."""
    missing_taxa = sorted(
        protein_id for protein_id in protein_ids
        if protein_id not in protein_taxa
    )
    if missing_taxa:
        raise ValueError(
            "taxon_pair_matched negative sampling requires a taxon_id for "
            f"every eligible protein. Missing {len(missing_taxa)}; examples: "
            f"{missing_taxa[:10]}. Provide complete taxonomy metadata or use "
            "--negative-sampling-policy global."
        )

    proteins_by_taxon: dict[str, list[str]] = {}
    for protein_id in protein_ids:
        proteins_by_taxon.setdefault(
            protein_taxa[protein_id],
            [],
        ).append(protein_id)

    positives_by_stratum: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = {}
    for pair in positive_keys:
        protein_a, protein_b = pair
        stratum = taxon_pair_key(
            protein_taxa[protein_a],
            protein_taxa[protein_b],
        )
        positives_by_stratum.setdefault(stratum, set()).add(pair)

    capacities: dict[tuple[str, str], int] = {}
    weights: dict[tuple[str, str], int] = {}
    for stratum, stratum_positives in positives_by_stratum.items():
        taxon_a, taxon_b = stratum
        if taxon_a == taxon_b:
            n_proteins = len(proteins_by_taxon[taxon_a])
            n_candidates = n_proteins * (n_proteins - 1) // 2
        else:
            n_candidates = (
                len(proteins_by_taxon[taxon_a])
                * len(proteins_by_taxon[taxon_b])
            )
        capacities[stratum] = n_candidates - len(stratum_positives)
        weights[stratum] = len(stratum_positives)

    n_available = sum(capacities.values())
    if n_available < target_count:
        raise ValueError(
            "Not enough possible negative pairs in the observed taxon-pair "
            f"strata. Requested {target_count}, available {n_available}."
        )
    allocations = allocate_stratified_samples(
        weights=weights,
        capacities=capacities,
        target_count=target_count,
    )

    sampled_keys = []
    for stratum in sorted(positives_by_stratum):
        taxon_a, taxon_b = stratum
        protein_a_ids = proteins_by_taxon[taxon_a]
        protein_b_ids = (
            None if taxon_a == taxon_b else proteins_by_taxon[taxon_b]
        )
        stratum_samples, _ = sample_pair_space(
            pair_space=IndexedPairSpace(protein_a_ids, protein_b_ids),
            forbidden_pairs=positives_by_stratum[stratum],
            n_samples=allocations[stratum],
            rng=rng,
        )
        sampled_keys.extend(stratum_samples)

    metadata = {
        "positive_pairs_by_taxon_pair": {
            taxon_pair_name(key): weights[key]
            for key in sorted(weights)
        },
        "sampled_negatives_by_taxon_pair": {
            taxon_pair_name(key): allocations[key]
            for key in sorted(allocations)
        },
        "available_negatives_by_taxon_pair": {
            taxon_pair_name(key): capacities[key]
            for key in sorted(capacities)
        },
    }
    return sampled_keys, metadata


def taxon_composition(
    pair_keys: Iterable[tuple[str, str]],
    protein_taxa: dict[str, str],
) -> dict[str, Any]:
    """Summarize same-, cross-, and unknown-taxon pair composition."""
    same_taxon = 0
    cross_taxon = 0
    missing_taxon = 0
    counts_by_taxon_pair: dict[str, int] = {}
    for protein_a, protein_b in pair_keys:
        taxon_a = protein_taxa.get(protein_a)
        taxon_b = protein_taxa.get(protein_b)
        if taxon_a is None or taxon_b is None:
            missing_taxon += 1
            continue
        if taxon_a == taxon_b:
            same_taxon += 1
        else:
            cross_taxon += 1
        stratum = taxon_pair_name(taxon_pair_key(taxon_a, taxon_b))
        counts_by_taxon_pair[stratum] = counts_by_taxon_pair.get(stratum, 0) + 1

    return {
        "n_same_taxon": same_taxon,
        "n_cross_taxon": cross_taxon,
        "n_missing_taxon": missing_taxon,
        "counts_by_taxon_pair": dict(sorted(counts_by_taxon_pair.items())),
    }


def finalize_negative_construction(
    construction: dict[str, Any] | None,
    pairs: pd.DataFrame,
    protein_taxa: dict[str, str],
    input_sha256: dict[str, str],
) -> dict[str, Any] | None:
    """Bind negative provenance to finalized pairs and source inputs."""
    if construction is None:
        return None

    positive_mask = pairs["label"] == 1
    n_positive = int(positive_mask.sum())
    n_negative = int((~positive_mask).sum())
    composition = construction.get("taxon_composition")
    if not composition:
        positive_keys = _pair_keys_from_prepared(pairs.loc[positive_mask])
        negative_keys = _pair_keys_from_prepared(pairs.loc[~positive_mask])
        composition = {
            "positive_pairs": taxon_composition(
                positive_keys,
                protein_taxa,
            ),
            "negative_pairs": taxon_composition(
                negative_keys,
                protein_taxa,
            ),
        }

    return {
        **construction,
        "negative_ratio_realized": n_negative / n_positive,
        "n_positive_pairs": n_positive,
        "n_negative_pairs": n_negative,
        "taxon_composition": composition,
        "evidence_snapshot": {"input_file_sha256": input_sha256},
    }


def sample_negative_pairs(
    positive_pairs: pd.DataFrame,
    spec: PPINegativeSamplingSpec,
    allowed_protein_ids: Iterable[str] | None = None,
    protein_taxa: dict[str, str] | None = None,
) -> PPINegativeSamplingResult:
    """Sample unobserved pairs using one explicit construction policy."""
    prepared = prepare_pair_columns(positive_pairs)
    prepared = prepared.loc[prepared["protein_a"] != prepared["protein_b"]]
    if allowed_protein_ids is not None:
        allowed = set(allowed_protein_ids)
        prepared = prepared.loc[
            prepared["protein_a"].isin(allowed)
            & prepared["protein_b"].isin(allowed)
        ]
    if prepared.empty:
        raise ValueError(
            "Cannot sample negatives without positive pairs whose proteins "
            "have FASTA sequences."
        )

    protein_ids = sorted(set(prepared["protein_a"]) | set(prepared["protein_b"]))
    positive_keys = _pair_keys_from_prepared(prepared)
    target_n_negatives = target_negative_count(
        len(positive_keys),
        spec.negative_ratio,
    )
    rng = random.Random(spec.seed)
    taxon_metadata: dict[str, Any] = {}
    if spec.policy == TAXON_PAIR_MATCHED_POLICY:
        sampled_keys, taxon_metadata = sample_taxon_stratified_negative_keys(
            protein_ids=protein_ids,
            positive_keys=positive_keys,
            protein_taxa=protein_taxa or {},
            target_count=target_n_negatives,
            rng=rng,
        )
        n_available_negatives = sum(
            taxon_metadata["available_negatives_by_taxon_pair"].values()
        )
    elif spec.policy == GLOBAL_POLICY:
        n_possible_pairs = len(protein_ids) * (len(protein_ids) - 1) // 2
        n_available_negatives = n_possible_pairs - len(positive_keys)
        if n_available_negatives < target_n_negatives:
            raise ValueError(
                "Not enough possible negative pairs to satisfy "
                f"negative_ratio={spec.negative_ratio}. Requested "
                f"{target_n_negatives}, available {n_available_negatives}."
            )
        sampled_keys, n_available_negatives = sample_pair_space(
            pair_space=IndexedPairSpace(protein_ids),
            forbidden_pairs=positive_keys,
            n_samples=target_n_negatives,
            rng=rng,
        )
    else:  # pragma: no cover - guarded by PPINegativeSamplingSpec
        raise ValueError(f"Unsupported negative sampling policy: {spec.policy}")

    sampled_pairs = pd.DataFrame(
        sampled_keys,
        columns=["protein_a", "protein_b"],
    )
    sampled_pairs["label"] = 0
    protein_taxa = protein_taxa or {}
    metadata = {
        "target_n_negatives": int(target_n_negatives),
        "n_sampled_negatives": int(len(sampled_pairs)),
        "n_positive_pairs_for_sampling": int(len(positive_keys)),
        "n_proteins_for_sampling": int(len(protein_ids)),
        "n_available_negative_pairs": int(n_available_negatives),
        "negative_sampling_policy": spec.policy,
        "negative_sampling_seed": int(spec.seed),
        "species_aware_sampling": spec.policy == TAXON_PAIR_MATCHED_POLICY,
        "taxon_composition": {
            "positive_pairs": taxon_composition(positive_keys, protein_taxa),
            "sampled_negative_pairs": taxon_composition(
                sampled_keys,
                protein_taxa,
            ),
        },
        **taxon_metadata,
    }
    metadata["negative_construction"] = generated_negative_construction(
        spec,
        metadata,
    )
    return PPINegativeSamplingResult(sampled_pairs, metadata)


def source_provided_negative_construction() -> dict[str, Any]:
    """Return canonical provenance for source-supplied negative labels."""
    return {
        "label_meaning": "source_provided_negative",
        "policy": SOURCE_PROVIDED_POLICY,
        "negative_ratio_requested": None,
        "negative_ratio_realized": None,
        "seed": None,
        "timing": "before_split",
        "partition_aware": False,
        "candidate_universe": None,
    }


def validate_negative_construction(
    construction: dict[str, Any],
) -> dict[str, Any]:
    """Validate structured negative provenance before propagation."""
    required = {
        "label_meaning",
        "policy",
        "negative_ratio_requested",
        "negative_ratio_realized",
        "seed",
        "timing",
        "partition_aware",
    }
    missing = sorted(required - construction.keys())
    if missing:
        raise ValueError(
            "dataset_metadata.json negative_construction is missing required "
            f"fields: {missing}"
        )

    policy = construction["policy"]
    supported_policies = {
        *NEGATIVE_SAMPLING_POLICY_CHOICES,
        SOURCE_PROVIDED_POLICY,
    }
    if not isinstance(policy, str) or policy not in supported_policies:
        raise ValueError(
            "dataset_metadata.json has unknown negative construction policy "
            f"{policy!r}."
        )
    if not isinstance(construction["timing"], str) or not construction["timing"]:
        raise ValueError("negative construction timing must be a non-empty string.")
    if not isinstance(construction["partition_aware"], bool):
        raise ValueError("negative construction partition_aware must be boolean.")

    requested_ratio = construction["negative_ratio_requested"]
    realized_ratio = construction["negative_ratio_realized"]
    if policy in NEGATIVE_SAMPLING_POLICY_CHOICES:
        if construction["label_meaning"] != "sampled_unobserved_pair":
            raise ValueError(
                "Generated negative construction must use label meaning "
                "'sampled_unobserved_pair'."
            )
        validate_negative_ratio(requested_ratio)
        validate_negative_ratio(realized_ratio)
        seed = construction["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("generated negative construction seed must be integer.")
    else:
        if construction["label_meaning"] != "source_provided_negative":
            raise ValueError(
                "Source-provided negative construction must use label meaning "
                "'source_provided_negative'."
            )
        if requested_ratio is not None or construction["seed"] is not None:
            raise ValueError(
                "Source-provided negative construction cannot declare a "
                "sampling ratio or seed."
            )
        if realized_ratio is not None:
            validate_negative_ratio(realized_ratio)

    return dict(construction)


def normalize_negative_construction(
    dataset_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return current or legacy negative-construction metadata."""
    construction = dataset_metadata.get("negative_construction")
    if construction is not None:
        if not isinstance(construction, dict):
            raise ValueError(
                "dataset_metadata.json negative_construction must be an "
                "object."
            )
        return validate_negative_construction(construction)

    sampled = dataset_metadata.get("sampled_negatives")
    if sampled is True:
        policy = (
            TAXON_PAIR_MATCHED_POLICY
            if dataset_metadata.get("species_aware_sampling")
            else GLOBAL_POLICY
        )
        n_positive = dataset_metadata.get("n_positive_pairs_for_sampling")
        n_negative = dataset_metadata.get("n_sampled_negatives")
        realized_ratio = (
            None
            if not n_positive or n_negative is None
            else int(n_negative) / int(n_positive)
        )
        return {
            "label_meaning": "sampled_unobserved_pair",
            "policy": policy,
            "negative_ratio_requested": dataset_metadata.get(
                "negative_ratio"
            ),
            "negative_ratio_realized": realized_ratio,
            "seed": dataset_metadata.get("seed"),
            "timing": "before_split",
            "partition_aware": False,
        }
    if sampled is False and dataset_metadata.get("n_negative_input", 0):
        construction = source_provided_negative_construction()
        n_positive = int(dataset_metadata.get("n_positive_output", 0))
        n_negative = int(dataset_metadata.get("n_negative_output", 0))
        if n_positive:
            construction["negative_ratio_realized"] = n_negative / n_positive
        return construction

    return None
