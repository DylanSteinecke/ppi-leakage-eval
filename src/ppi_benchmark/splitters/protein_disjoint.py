"""
C1/C2/C3 protein-disjoint splitting for PPI benchmark pairs.
"""

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


SplitMode = Literal["c1", "c2", "c3"]
REQUIRED_COLUMNS = {"protein_a", "protein_b", "label"}


@dataclass(frozen=True)
class SplitResult:
    """
    Selected pair tables and JSON-serializable split audit information.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    dropped: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class ThreeWaySplitResult:
    """
    Selected train/validation/test tables and split audit information.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class _PairArrays:
    """
    Integer-coded pair data reused across candidate trials.
    """

    protein_a: np.ndarray
    protein_b: np.ndarray
    group_a: np.ndarray
    group_b: np.ndarray
    labels: np.ndarray
    label_values: tuple[Any, ...]
    n_proteins: int
    n_groups: int


@dataclass(frozen=True)
class _Candidate:
    """
    One valid candidate assignment and its ordered optimization score.
    """

    train_mask: np.ndarray
    test_mask: np.ndarray
    dropped_mask: np.ndarray
    heldout_groups: frozenset[int]
    score: tuple[int, float, float, float]


@dataclass(frozen=True)
class _ThreeWayCandidate:
    """
    One valid three-way candidate and its ordered optimization score.
    """

    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    dropped_mask: np.ndarray
    group_partitions: np.ndarray | None
    score: tuple[int, float, float, float]


@dataclass(frozen=True)
class _EdgeUnits:
    """
    Compact unordered-edge unit codes and their row counts.
    """

    row_codes: np.ndarray
    sizes: np.ndarray


def _native_value(value: Any) -> Any:
    """
    Convert numpy scalar values to JSON-native values.
    """
    item_method = getattr(value, "item", None)
    if item_method is not None:
        try:
            return item_method()
        except (TypeError, ValueError):
            pass

    return value


def _validate_inputs(
        pairs: pd.DataFrame, mode: str, test_size: float,
        protein_to_group: Mapping[Hashable, Hashable] | None,
        n_trials: int,
    ) -> None:
    """
    Validate the public splitter inputs before candidate generation.
    """
    missing_columns = REQUIRED_COLUMNS - set(pairs.columns)
    if missing_columns:
        raise ValueError(f"pairs is missing columns: {missing_columns}")
    if pairs.empty:
        raise ValueError("pairs must contain at least one row.")
    if mode not in {"c1", "c2", "c3"}:
        raise ValueError("mode must be one of: c1, c2, c3.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    protein_columns = pairs[["protein_a", "protein_b"]]
    if protein_columns.isna().any(axis=None):
        raise ValueError("protein_a and protein_b must not contain missing IDs.")
    if pairs["label"].isna().any():
        raise ValueError("label must not contain missing values.")

    if protein_to_group is not None and mode in {"c2", "c3"}:
        proteins = set(protein_columns.to_numpy().ravel())
        missing_proteins = proteins - set(protein_to_group)
        if missing_proteins:
            examples = sorted(missing_proteins, key=str)[:10]
            raise ValueError(
                "protein_to_group must map every protein used by pairs. "
                f"Missing {len(missing_proteins)} protein(s): {examples}")
        for protein in proteins:
            group = protein_to_group[protein]
            group_is_missing = (
                group is None
                or group is pd.NA
                or (
                    isinstance(group, (float, np.floating))
                    and math.isnan(group)
                )
                or (isinstance(group, str) and not group.strip())
            )
            if group_is_missing:
                raise ValueError(
                    f"protein_to_group[{protein!r}] must not be missing.")
            try:
                hash(group)
            except TypeError as exc:
                raise ValueError(
                    "protein_to_group values must be hashable.") from exc


def _factorize(values: list[Hashable]) -> tuple[np.ndarray, int]:
    """
    Factorize hashable values without relying on cross-type sorting.
    """
    codes = np.empty(len(values), dtype=np.int64)
    value_to_code: dict[Hashable, int] = {}
    for index, value in enumerate(values):
        if value not in value_to_code:
            value_to_code[value] = len(value_to_code)
        codes[index] = value_to_code[value]

    return codes, len(value_to_code)


def _encode_pairs(
        pairs: pd.DataFrame,
        protein_to_group: Mapping[Hashable, Hashable] | None,
    ) -> _PairArrays:
    """
    Encode protein, group, and label values as compact integer arrays.
    """
    protein_values = list(pairs["protein_a"]) + list(pairs["protein_b"])
    protein_codes, n_proteins = _factorize(protein_values)
    n_pairs = len(pairs)
    protein_a = protein_codes[:n_pairs]
    protein_b = protein_codes[n_pairs:]

    protein_ids = list(dict.fromkeys(protein_values))
    if protein_to_group is None:
        protein_group_codes = np.arange(n_proteins, dtype=np.int64)
        n_groups = n_proteins
    else:
        group_values = [protein_to_group[protein] for protein in protein_ids]
        protein_group_codes, n_groups = _factorize(group_values)

    label_codes, label_values = pd.factorize(pairs["label"], sort=False)

    return _PairArrays(
        protein_a=protein_a,
        protein_b=protein_b,
        group_a=protein_group_codes[protein_a],
        group_b=protein_group_codes[protein_b],
        labels=label_codes.astype(np.int64, copy=False),
        label_values=tuple(_native_value(value) for value in label_values),
        n_proteins=n_proteins,
        n_groups=n_groups,
    )


def _label_balance_error(
        arrays: _PairArrays, train_mask: np.ndarray, test_mask: np.ndarray,
    ) -> float:
    """
    Return total train/test class-rate deviation from the input rates.
    """
    n_labels = len(arrays.label_values)
    input_rates = np.bincount(
        arrays.labels,
        minlength=n_labels,
    ) / len(arrays.labels)
    error = 0.0
    for mask in (train_mask, test_mask):
        split_rates = np.bincount(
            arrays.labels[mask],
            minlength=n_labels,
        ) / int(mask.sum())
        error += float(np.abs(split_rates - input_rates).sum())

    return error


def _degree_values(arrays: _PairArrays, mask: np.ndarray) -> np.ndarray:
    """
    Return nonzero protein degrees for selected edges.
    """
    degrees = np.bincount(
        np.concatenate((arrays.protein_a[mask], arrays.protein_b[mask])),
        minlength=arrays.n_proteins,
    )

    return degrees[degrees > 0].astype(float, copy=False)


def _degree_distribution_distance(
        arrays: _PairArrays, train_mask: np.ndarray, test_mask: np.ndarray,
    ) -> float:
    """
    Compare scale-normalized train/test degree-distribution quantiles.
    """
    train_degrees = _degree_values(arrays, train_mask)
    test_degrees = _degree_values(arrays, test_mask)
    if not len(train_degrees) or not len(test_degrees):
        return math.inf

    quantiles = np.linspace(0.0, 1.0, 21)
    train_quantiles = np.quantile(
        train_degrees / train_degrees.mean(),
        quantiles,
    )
    test_quantiles = np.quantile(
        test_degrees / test_degrees.mean(),
        quantiles,
    )

    return float(np.abs(train_quantiles - test_quantiles).mean())


def _candidate_score(
        arrays: _PairArrays, train_mask: np.ndarray, test_mask: np.ndarray,
        test_size: float,
    ) -> tuple[int, float, float, float]:
    """
    Return the requested lexicographic candidate score.
    """
    n_retained = int(train_mask.sum() + test_mask.sum())
    actual_test_size = int(test_mask.sum()) / n_retained

    return (
        n_retained,
        -abs(actual_test_size - test_size),
        -_label_balance_error(arrays, train_mask, test_mask),
        -_degree_distribution_distance(arrays, train_mask, test_mask),
    )


def _three_way_candidate_score(
        arrays: _PairArrays, train_mask: np.ndarray, val_mask: np.ndarray,
        test_mask: np.ndarray, val_size: float, test_size: float,
    ) -> tuple[int, float, float, float]:
    """
    Return a lexicographic score for a retained three-way partition.
    """
    masks = (train_mask, val_mask, test_mask)
    n_retained = sum(int(mask.sum()) for mask in masks)
    actual_val_size = int(val_mask.sum()) / n_retained
    actual_test_size = int(test_mask.sum()) / n_retained
    size_error = (
        abs(actual_val_size - val_size)
        + abs(actual_test_size - test_size)
    )

    n_labels = len(arrays.label_values)
    input_rates = np.bincount(
        arrays.labels,
        minlength=n_labels,
    ) / len(arrays.labels)
    label_balance_error = 0.0
    for mask in masks:
        split_rates = np.bincount(
            arrays.labels[mask],
            minlength=n_labels,
        ) / int(mask.sum())
        label_balance_error += float(
            np.abs(split_rates - input_rates).sum())

    degree_distance = (
        _degree_distribution_distance(arrays, train_mask, val_mask)
        + _degree_distribution_distance(arrays, train_mask, test_mask)
    )

    return (
        n_retained,
        -size_error,
        -label_balance_error,
        -degree_distance,
    )


def _masks_contain_all_labels(
        arrays: _PairArrays, masks: tuple[np.ndarray, ...],
    ) -> bool:
    """
    Return whether every mask contains every input label.
    """
    n_labels = len(arrays.label_values)
    return all(
        bool((np.bincount(
            arrays.labels[mask],
            minlength=n_labels,
        ) > 0).all())
        for mask in masks
    )


def _edge_units(arrays: _PairArrays) -> _EdgeUnits:
    """
    Group duplicate unordered protein pairs into indivisible C1 edge units.
    """
    lower_proteins = np.minimum(arrays.protein_a, arrays.protein_b)
    upper_proteins = np.maximum(arrays.protein_a, arrays.protein_b)
    pair_keys = lower_proteins * arrays.n_proteins + upper_proteins
    _, row_codes, sizes = np.unique(
        pair_keys,
        return_inverse=True,
        return_counts=True,
    )

    return _EdgeUnits(
        row_codes=row_codes.astype(np.int64, copy=False),
        sizes=sizes.astype(np.int64, copy=False),
    )


def _make_c1_candidate(
        arrays: _PairArrays, edge_units: _EdgeUnits, test_size: float,
        rng: np.random.Generator,
    ) -> _Candidate | None:
    """
    Make an edge-disjoint candidate with every test protein seen in train.
    """
    target_test_edges = len(arrays.labels) * test_size
    shuffled_units = rng.permutation(len(edge_units.sizes))
    cumulative_sizes = np.cumsum(edge_units.sizes[shuffled_units])
    n_selected_units = int(np.argmin(
        np.abs(cumulative_sizes - target_test_edges),
    )) + 1
    selected_units = np.zeros(len(edge_units.sizes), dtype=bool)
    selected_units[shuffled_units[:n_selected_units]] = True

    if not selected_units.any() or selected_units.all():
        return None

    test_mask = selected_units[edge_units.row_codes]

    # Move whole edge units back to train until every test endpoint is seen.
    while test_mask.any():
        train_mask = ~test_mask
        train_proteins = np.zeros(arrays.n_proteins, dtype=bool)
        train_proteins[arrays.protein_a[train_mask]] = True
        train_proteins[arrays.protein_b[train_mask]] = True
        missing_test_proteins = np.zeros(arrays.n_proteins, dtype=bool)
        missing_test_proteins[arrays.protein_a[test_mask]] = True
        missing_test_proteins[arrays.protein_b[test_mask]] = True
        missing_test_proteins &= ~train_proteins
        if not missing_test_proteins.any():
            break

        missing_endpoint_rows = test_mask & (
            missing_test_proteins[arrays.protein_a]
            | missing_test_proteins[arrays.protein_b]
        )
        movable_units = np.unique(edge_units.row_codes[missing_endpoint_rows])
        if not len(movable_units):
            return None
        selected_units[rng.choice(movable_units)] = False
        test_mask = selected_units[edge_units.row_codes]

    if not test_mask.any() or test_mask.all():
        return None

    train_mask = ~test_mask
    dropped_mask = np.zeros(len(arrays.labels), dtype=bool)

    return _Candidate(
        train_mask=train_mask,
        test_mask=test_mask,
        dropped_mask=dropped_mask,
        heldout_groups=frozenset(),
        score=_candidate_score(arrays, train_mask, test_mask, test_size),
    )


def _heldout_group_fraction(mode: SplitMode, test_size: float) -> float:
    """
    Estimate the group fraction giving the requested retained test fraction.
    """
    if mode == "c2":
        return test_size / (2.0 - test_size)

    test_weight = math.sqrt(test_size)
    train_weight = math.sqrt(1.0 - test_size)

    return test_weight / (test_weight + train_weight)


def _make_group_candidate(
        arrays: _PairArrays, mode: Literal["c2", "c3"], test_size: float,
        rng: np.random.Generator,
    ) -> _Candidate | None:
    """
    Make one C2 or C3 candidate from an atomic group assignment.
    """
    n_heldout = round(
        arrays.n_groups * _heldout_group_fraction(mode, test_size))
    n_heldout = min(max(1, n_heldout), arrays.n_groups - 1)
    heldout_groups = frozenset(
        int(group_code)
        for group_code in rng.choice(
            arrays.n_groups,
            size=n_heldout,
            replace=False,
        )
    )
    group_is_heldout = np.zeros(arrays.n_groups, dtype=bool)
    group_is_heldout[list(heldout_groups)] = True
    protein_a_heldout = group_is_heldout[arrays.group_a]
    protein_b_heldout = group_is_heldout[arrays.group_b]
    train_mask = ~protein_a_heldout & ~protein_b_heldout

    if mode == "c2":
        test_mask = protein_a_heldout ^ protein_b_heldout
        represented_train_groups = np.zeros(arrays.n_groups, dtype=bool)
        represented_train_groups[arrays.group_a[train_mask]] = True
        represented_train_groups[arrays.group_b[train_mask]] = True
        test_train_groups = np.where(
            protein_a_heldout[test_mask],
            arrays.group_b[test_mask],
            arrays.group_a[test_mask],
        )
        if not represented_train_groups[test_train_groups].all():
            return None
    else:
        test_mask = protein_a_heldout & protein_b_heldout

    dropped_mask = ~(train_mask | test_mask)
    if not train_mask.any() or not test_mask.any():
        return None

    return _Candidate(
        train_mask=train_mask,
        test_mask=test_mask,
        dropped_mask=dropped_mask,
        heldout_groups=heldout_groups,
        score=_candidate_score(arrays, train_mask, test_mask, test_size),
    )


def _make_three_way_c1_candidate(
        arrays: _PairArrays, edge_units: _EdgeUnits,
        val_size: float, test_size: float,
        rng: np.random.Generator,
    ) -> _ThreeWayCandidate | None:
    """
    Split a C1 heldout edge set into validation and test edge units.
    """
    combined_heldout_size = val_size + test_size
    two_way_candidate = _make_c1_candidate(
        arrays=arrays,
        edge_units=edge_units,
        test_size=combined_heldout_size,
        rng=rng,
    )
    if two_way_candidate is None:
        return None

    heldout_units = np.unique(
        edge_units.row_codes[two_way_candidate.test_mask])
    if len(heldout_units) < 2:
        return None
    shuffled_units = rng.permutation(heldout_units)
    shuffled_sizes = edge_units.sizes[shuffled_units]
    target_val_edges = (
        int(two_way_candidate.test_mask.sum())
        * val_size
        / combined_heldout_size
    )
    cumulative_sizes = np.cumsum(shuffled_sizes)
    n_val_units = int(np.argmin(
        np.abs(cumulative_sizes - target_val_edges),
    )) + 1
    n_val_units = min(max(1, n_val_units), len(heldout_units) - 1)

    val_units = np.zeros(len(edge_units.sizes), dtype=bool)
    val_units[shuffled_units[:n_val_units]] = True
    val_mask = (
        two_way_candidate.test_mask
        & val_units[edge_units.row_codes]
    )
    test_mask = two_way_candidate.test_mask & ~val_mask
    train_mask = two_way_candidate.train_mask
    dropped_mask = two_way_candidate.dropped_mask
    masks = (train_mask, val_mask, test_mask)
    if (
            not all(mask.any() for mask in masks)
            or not _masks_contain_all_labels(arrays, masks)
            ):
        return None

    return _ThreeWayCandidate(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        dropped_mask=dropped_mask,
        group_partitions=None,
        score=_three_way_candidate_score(
            arrays,
            train_mask,
            val_mask,
            test_mask,
            val_size,
            test_size,
        ),
    )


def _three_way_group_weights(
        mode: Literal["c2", "c3"], train_size: float,
        val_size: float, test_size: float,
    ) -> np.ndarray:
    """
    Estimate group weights yielding requested retained edge fractions.
    """
    if mode == "c2":
        weights = np.array([
            1.0,
            val_size / (2.0 * train_size),
            test_size / (2.0 * train_size),
        ])
    else:
        weights = np.sqrt(np.array([train_size, val_size, test_size]))

    return weights / weights.sum()


def _three_way_group_counts(
        n_groups: int, weights: np.ndarray,
    ) -> np.ndarray:
    """
    Convert group weights to positive integer train/val/test counts.
    """
    raw_counts = weights * n_groups
    counts = np.maximum(1, np.floor(raw_counts).astype(int))
    while int(counts.sum()) > n_groups:
        removable = np.where(counts > 1)[0]
        if not len(removable):
            raise ValueError("Three-way splitting requires at least 3 groups.")
        excess = counts[removable] - raw_counts[removable]
        counts[removable[int(np.argmax(excess))]] -= 1
    while int(counts.sum()) < n_groups:
        deficit = raw_counts - counts
        counts[int(np.argmax(deficit))] += 1

    return counts


def _make_three_way_group_candidate(
        arrays: _PairArrays, mode: Literal["c2", "c3"],
        val_size: float, test_size: float,
        rng: np.random.Generator,
    ) -> _ThreeWayCandidate | None:
    """
    Make one C2/C3 candidate from atomic train/val/test group assignments.
    """
    train_size = 1.0 - val_size - test_size
    group_counts = _three_way_group_counts(
        arrays.n_groups,
        _three_way_group_weights(
            mode,
            train_size,
            val_size,
            test_size,
        ),
    )
    shuffled_groups = rng.permutation(arrays.n_groups)
    group_partitions = np.empty(arrays.n_groups, dtype=np.int8)
    train_end = int(group_counts[0])
    val_end = train_end + int(group_counts[1])
    group_partitions[shuffled_groups[:train_end]] = 0
    group_partitions[shuffled_groups[train_end:val_end]] = 1
    group_partitions[shuffled_groups[val_end:]] = 2

    partition_a = group_partitions[arrays.group_a]
    partition_b = group_partitions[arrays.group_b]
    train_mask = (partition_a == 0) & (partition_b == 0)
    if mode == "c2":
        val_mask = (
            ((partition_a == 0) & (partition_b == 1))
            | ((partition_a == 1) & (partition_b == 0))
        )
        test_mask = (
            ((partition_a == 0) & (partition_b == 2))
            | ((partition_a == 2) & (partition_b == 0))
        )
        represented_train_groups = np.zeros(arrays.n_groups, dtype=bool)
        represented_train_groups[arrays.group_a[train_mask]] = True
        represented_train_groups[arrays.group_b[train_mask]] = True
        heldout_mask = val_mask | test_mask
        heldout_train_groups = np.where(
            partition_a[heldout_mask] == 0,
            arrays.group_a[heldout_mask],
            arrays.group_b[heldout_mask],
        )
        if not represented_train_groups[heldout_train_groups].all():
            return None
    else:
        val_mask = (partition_a == 1) & (partition_b == 1)
        test_mask = (partition_a == 2) & (partition_b == 2)

    dropped_mask = ~(train_mask | val_mask | test_mask)
    masks = (train_mask, val_mask, test_mask)
    if (
            not all(mask.any() for mask in masks)
            or not _masks_contain_all_labels(arrays, masks)
            ):
        return None

    return _ThreeWayCandidate(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        dropped_mask=dropped_mask,
        group_partitions=group_partitions,
        score=_three_way_candidate_score(
            arrays,
            train_mask,
            val_mask,
            test_mask,
            val_size,
            test_size,
        ),
    )


def _class_summary(
        arrays: _PairArrays, mask: np.ndarray,
    ) -> dict[str, dict[str, int | float]]:
    """
    Return class counts and rates for one edge mask.
    """
    counts = np.bincount(
        arrays.labels[mask],
        minlength=len(arrays.label_values),
    )
    n_rows = int(mask.sum())

    return {
        str(label): {
            "count": int(count),
            "rate": float(count / n_rows) if n_rows else 0.0,
        }
        for label, count in zip(arrays.label_values, counts)
    }


def _degree_summary(
        arrays: _PairArrays, mask: np.ndarray,
    ) -> dict[str, int | float]:
    """
    Return compact degree statistics for proteins present in one edge set.
    """
    degrees = _degree_values(arrays, mask)
    if not len(degrees):
        return {
            "n_proteins": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }

    return {
        "n_proteins": int(len(degrees)),
        "min": float(degrees.min()),
        "max": float(degrees.max()),
        "mean": float(degrees.mean()),
        "median": float(np.median(degrees)),
        "std": float(degrees.std()),
        "q25": float(np.quantile(degrees, 0.25)),
        "q75": float(np.quantile(degrees, 0.75)),
    }


def _protein_set(
        arrays: _PairArrays, mask: np.ndarray,
    ) -> set[int]:
    """
    Return encoded proteins represented by selected edges.
    """
    return set(arrays.protein_a[mask]) | set(arrays.protein_b[mask])


def _group_set(
        arrays: _PairArrays, mask: np.ndarray,
    ) -> set[int]:
    """
    Return encoded protein groups represented by selected edges.
    """
    return set(arrays.group_a[mask]) | set(arrays.group_b[mask])


def _invariant_checks(
        arrays: _PairArrays, candidate: _Candidate, mode: SplitMode,
    ) -> dict[str, bool]:
    """
    Evaluate mode-specific and partition invariants for the final candidate.
    """
    train_mask = candidate.train_mask
    test_mask = candidate.test_mask
    dropped_mask = candidate.dropped_mask
    checks = {
        "train_test_nonempty": bool(train_mask.any() and test_mask.any()),
        "masks_do_not_overlap": bool(
            not (train_mask & test_mask).any()
            and not (train_mask & dropped_mask).any()
            and not (test_mask & dropped_mask).any()
        ),
        "all_edges_accounted_for": bool(
            (train_mask | test_mask | dropped_mask).all()),
    }

    if mode == "c1":
        train_proteins = _protein_set(arrays, train_mask)
        test_proteins = _protein_set(arrays, test_mask)
        train_pairs = {
            tuple(sorted((int(protein_a), int(protein_b))))
            for protein_a, protein_b in zip(
                arrays.protein_a[train_mask],
                arrays.protein_b[train_mask],
            )
        }
        test_pairs = {
            tuple(sorted((int(protein_a), int(protein_b))))
            for protein_a, protein_b in zip(
                arrays.protein_a[test_mask],
                arrays.protein_b[test_mask],
            )
        }
        checks.update({
            "unordered_edges_are_disjoint": train_pairs.isdisjoint(test_pairs),
            "all_test_proteins_appear_in_train": test_proteins <= train_proteins,
            "no_edges_dropped": not dropped_mask.any(),
        })
    else:
        heldout = np.zeros(arrays.n_groups, dtype=bool)
        heldout[list(candidate.heldout_groups)] = True
        a_heldout = heldout[arrays.group_a]
        b_heldout = heldout[arrays.group_b]
        checks["train_edges_have_two_train_groups"] = bool(
            (~a_heldout[train_mask] & ~b_heldout[train_mask]).all())
        if mode == "c2":
            checks.update({
                "test_edges_have_one_train_and_one_heldout_group": bool(
                    (a_heldout[test_mask] ^ b_heldout[test_mask]).all()),
                "dropped_edges_have_two_heldout_groups": bool(
                    (a_heldout[dropped_mask] & b_heldout[dropped_mask]).all()),
            })
        else:
            checks.update({
                "test_edges_have_two_heldout_groups": bool(
                    (a_heldout[test_mask] & b_heldout[test_mask]).all()),
                "dropped_edges_cross_group_partitions": bool(
                    (a_heldout[dropped_mask] ^ b_heldout[dropped_mask]).all()),
            })

    return checks


def _unordered_edge_set(
        arrays: _PairArrays, mask: np.ndarray,
    ) -> set[tuple[int, int]]:
    """
    Return encoded unordered protein pairs selected by a mask.
    """
    return {
        tuple(sorted((int(protein_a), int(protein_b))))
        for protein_a, protein_b in zip(
            arrays.protein_a[mask],
            arrays.protein_b[mask],
        )
    }


def _three_way_invariant_checks(
        arrays: _PairArrays, candidate: _ThreeWayCandidate,
        mode: SplitMode,
    ) -> dict[str, bool]:
    """
    Evaluate mode-specific invariants for a train/validation/test candidate.
    """
    train_mask = candidate.train_mask
    val_mask = candidate.val_mask
    test_mask = candidate.test_mask
    dropped_mask = candidate.dropped_mask
    masks = (train_mask, val_mask, test_mask, dropped_mask)
    checks = {
        "train_val_test_nonempty": bool(
            train_mask.any() and val_mask.any() and test_mask.any()),
        "masks_do_not_overlap": bool(
            all(
                not (left_mask & right_mask).any()
                for left_index, left_mask in enumerate(masks[:-1])
                for right_mask in masks[left_index + 1:]
            )
        ),
        "all_edges_accounted_for": bool(
            (train_mask | val_mask | test_mask | dropped_mask).all()),
        "all_splits_contain_all_labels": _masks_contain_all_labels(
            arrays,
            (train_mask, val_mask, test_mask),
        ),
    }

    if mode == "c1":
        train_proteins = _protein_set(arrays, train_mask)
        val_proteins = _protein_set(arrays, val_mask)
        test_proteins = _protein_set(arrays, test_mask)
        train_pairs = _unordered_edge_set(arrays, train_mask)
        val_pairs = _unordered_edge_set(arrays, val_mask)
        test_pairs = _unordered_edge_set(arrays, test_mask)
        checks.update({
            "unordered_edges_are_pairwise_disjoint": bool(
                train_pairs.isdisjoint(val_pairs)
                and train_pairs.isdisjoint(test_pairs)
                and val_pairs.isdisjoint(test_pairs)
            ),
            "all_val_proteins_appear_in_train": (
                val_proteins <= train_proteins),
            "all_test_proteins_appear_in_train": (
                test_proteins <= train_proteins),
            "no_edges_dropped": not dropped_mask.any(),
        })
        return checks

    assert candidate.group_partitions is not None
    partition_a = candidate.group_partitions[arrays.group_a]
    partition_b = candidate.group_partitions[arrays.group_b]
    checks["train_edges_have_two_train_groups"] = bool(
        ((partition_a[train_mask] == 0)
         & (partition_b[train_mask] == 0)).all())
    if mode == "c2":
        checks.update({
            "val_edges_have_one_train_and_one_val_group": bool(
                (((partition_a[val_mask] == 0)
                  & (partition_b[val_mask] == 1))
                 | ((partition_a[val_mask] == 1)
                    & (partition_b[val_mask] == 0))).all()
            ),
            "test_edges_have_one_train_and_one_test_group": bool(
                (((partition_a[test_mask] == 0)
                  & (partition_b[test_mask] == 2))
                 | ((partition_a[test_mask] == 2)
                    & (partition_b[test_mask] == 0))).all()
            ),
            "dropped_edges_are_outside_c2_definitions": bool(
                (~(
                    ((partition_a[dropped_mask] == 0)
                     & (partition_b[dropped_mask] == 0))
                    | ((partition_a[dropped_mask] == 0)
                       & (partition_b[dropped_mask] == 1))
                    | ((partition_a[dropped_mask] == 1)
                       & (partition_b[dropped_mask] == 0))
                    | ((partition_a[dropped_mask] == 0)
                       & (partition_b[dropped_mask] == 2))
                    | ((partition_a[dropped_mask] == 2)
                       & (partition_b[dropped_mask] == 0))
                )).all()
            ),
        })
    else:
        checks.update({
            "val_edges_have_two_val_groups": bool(
                ((partition_a[val_mask] == 1)
                 & (partition_b[val_mask] == 1)).all()),
            "test_edges_have_two_test_groups": bool(
                ((partition_a[test_mask] == 2)
                 & (partition_b[test_mask] == 2)).all()),
            "dropped_edges_cross_group_partitions": bool(
                (partition_a[dropped_mask]
                 != partition_b[dropped_mask]).all()),
        })

    return checks


def _three_way_audit(
        arrays: _PairArrays, candidate: _ThreeWayCandidate,
        mode: SplitMode, val_size: float, test_size: float,
        seed: int, n_trials: int,
    ) -> dict[str, Any]:
    """
    Build a JSON-serializable report for a three-way split candidate.
    """
    split_masks = {
        "train": candidate.train_mask,
        "val": candidate.val_mask,
        "test": candidate.test_mask,
    }
    retained_mask = (
        candidate.train_mask | candidate.val_mask | candidate.test_mask)
    n_input = len(arrays.labels)
    n_retained = int(retained_mask.sum())
    split_proteins = {
        split_name: _protein_set(arrays, mask)
        for split_name, mask in split_masks.items()
    }
    split_groups = {
        split_name: _group_set(arrays, mask)
        for split_name, mask in split_masks.items()
    }
    checks = _three_way_invariant_checks(arrays, candidate, mode)

    audit = {
        "mode": mode,
        "seed": seed,
        "n_trials": n_trials,
        "requested_train_size": 1.0 - val_size - test_size,
        "requested_val_size": val_size,
        "requested_test_size": test_size,
        "actual_train_size": float(
            candidate.train_mask.sum() / n_retained),
        "actual_val_size": float(candidate.val_mask.sum() / n_retained),
        "actual_test_size": float(candidate.test_mask.sum() / n_retained),
        "n_input_edges": n_input,
        "n_train_edges": int(candidate.train_mask.sum()),
        "n_val_edges": int(candidate.val_mask.sum()),
        "n_test_edges": int(candidate.test_mask.sum()),
        "n_retained_edges": n_retained,
        "n_dropped_edges": int(candidate.dropped_mask.sum()),
        "retained_edge_fraction": float(n_retained / n_input),
        "dropped_edge_fraction": float(
            candidate.dropped_mask.sum() / n_input),
        "n_proteins_input": arrays.n_proteins,
        "n_groups_input": arrays.n_groups,
        "class_distribution": {
            "input": _class_summary(
                arrays,
                np.ones(n_input, dtype=bool),
            ),
            **{
                split_name: _class_summary(arrays, mask)
                for split_name, mask in split_masks.items()
            },
        },
        "degree_statistics": {
            "input": _degree_summary(
                arrays,
                np.ones(n_input, dtype=bool),
            ),
            **{
                split_name: _degree_summary(arrays, mask)
                for split_name, mask in split_masks.items()
            },
            "train_val_distance": _degree_distribution_distance(
                arrays,
                candidate.train_mask,
                candidate.val_mask,
            ),
            "train_test_distance": _degree_distribution_distance(
                arrays,
                candidate.train_mask,
                candidate.test_mask,
            ),
        },
        "selection_score": {
            "retained_edges": candidate.score[0],
            "val_test_size_error": -candidate.score[1],
            "label_balance_error": -candidate.score[2],
            "degree_distribution_distance": -candidate.score[3],
        },
        "invariant_checks": checks,
        "all_invariants_passed": all(checks.values()),
    }
    for split_name in split_masks:
        audit[f"n_proteins_{split_name}"] = len(
            split_proteins[split_name])
        audit[f"n_groups_{split_name}"] = len(split_groups[split_name])
    for left_name, right_name in (
            ("train", "val"), ("train", "test"), ("val", "test")):
        audit[f"n_shared_proteins_{left_name}_{right_name}"] = len(
            split_proteins[left_name] & split_proteins[right_name])
        audit[f"n_shared_groups_{left_name}_{right_name}"] = len(
            split_groups[left_name] & split_groups[right_name])
    if candidate.group_partitions is not None:
        partition_counts = np.bincount(
            candidate.group_partitions,
            minlength=3,
        )
        audit["n_assigned_train_groups"] = int(partition_counts[0])
        audit["n_assigned_val_groups"] = int(partition_counts[1])
        audit["n_assigned_test_groups"] = int(partition_counts[2])

    return audit


def _make_audit(
        arrays: _PairArrays, candidate: _Candidate, mode: SplitMode,
        test_size: float, seed: int, n_trials: int,
    ) -> dict[str, Any]:
    """
    Build a compact JSON-serializable report for the selected candidate.
    """
    train_mask = candidate.train_mask
    test_mask = candidate.test_mask
    dropped_mask = candidate.dropped_mask
    retained_mask = train_mask | test_mask
    n_input = len(arrays.labels)
    n_retained = int(retained_mask.sum())
    train_proteins = _protein_set(arrays, train_mask)
    test_proteins = _protein_set(arrays, test_mask)
    train_groups = _group_set(arrays, train_mask)
    test_groups = _group_set(arrays, test_mask)
    checks = _invariant_checks(arrays, candidate, mode)

    return {
        "mode": mode,
        "seed": seed,
        "n_trials": n_trials,
        "requested_test_size": test_size,
        "actual_test_size": float(test_mask.sum() / n_retained),
        "n_input_edges": n_input,
        "n_train_edges": int(train_mask.sum()),
        "n_test_edges": int(test_mask.sum()),
        "n_retained_edges": n_retained,
        "n_dropped_edges": int(dropped_mask.sum()),
        "retained_edge_fraction": float(n_retained / n_input),
        "dropped_edge_fraction": float(dropped_mask.sum() / n_input),
        "n_proteins_input": arrays.n_proteins,
        "n_proteins_train": len(train_proteins),
        "n_proteins_test": len(test_proteins),
        "n_shared_proteins_train_test": len(train_proteins & test_proteins),
        "n_groups_input": arrays.n_groups,
        "n_groups_train": len(train_groups),
        "n_groups_test": len(test_groups),
        "n_shared_groups_train_test": len(train_groups & test_groups),
        "n_assigned_heldout_groups": len(candidate.heldout_groups),
        "class_distribution": {
            "input": _class_summary(
                arrays,
                np.ones(n_input, dtype=bool),
            ),
            "train": _class_summary(arrays, train_mask),
            "test": _class_summary(arrays, test_mask),
        },
        "degree_statistics": {
            "input": _degree_summary(
                arrays,
                np.ones(n_input, dtype=bool),
            ),
            "train": _degree_summary(arrays, train_mask),
            "test": _degree_summary(arrays, test_mask),
            "train_test_distance": _degree_distribution_distance(
                arrays,
                train_mask,
                test_mask,
            ),
        },
        "selection_score": {
            "retained_edges": candidate.score[0],
            "test_size_error": -candidate.score[1],
            "label_balance_error": -candidate.score[2],
            "degree_distribution_distance": -candidate.score[3],
        },
        "invariant_checks": checks,
        "all_invariants_passed": all(checks.values()),
    }


def split_pairs(
        pairs: pd.DataFrame,
        mode: SplitMode,
        test_size: float = 0.2,
        protein_to_group: dict[Hashable, Hashable] | None = None,
        seed: int = 0,
        n_trials: int = 100,
    ) -> SplitResult:
    """
    Split PPI pairs under the requested C1, C2, or C3 generalization regime.

    C1 holds out edges while requiring every test protein in train. C2 test
    edges join one train group to one held-out group. C3 test edges join two
    held-out groups. C2/C3 discard edges outside their respective definitions.
    """
    _validate_inputs(
        pairs=pairs,
        mode=mode,
        test_size=test_size,
        protein_to_group=protein_to_group,
        n_trials=n_trials,
    )
    group_mapping = protein_to_group if mode in {"c2", "c3"} else None
    arrays = _encode_pairs(pairs, group_mapping)
    if mode in {"c2", "c3"} and arrays.n_groups < 2:
        raise ValueError(f"{mode.upper()} requires at least two protein groups.")

    edge_units = _edge_units(arrays) if mode == "c1" else None
    best_candidate = None
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        if mode == "c1":
            assert edge_units is not None
            candidate = _make_c1_candidate(
                arrays,
                edge_units,
                test_size,
                rng,
            )
        else:
            candidate = _make_group_candidate(
                arrays,
                mode,
                test_size,
                rng,
            )
        if candidate is not None and (
                best_candidate is None
                or candidate.score > best_candidate.score
            ):
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError(
            f"Could not produce a valid {mode.upper()} split after "
            f"{n_trials} trial(s). Try more data, a different seed, a "
            "different test_size, or more trials.")

    audit = _make_audit(
        arrays=arrays,
        candidate=best_candidate,
        mode=mode,
        test_size=test_size,
        seed=seed,
        n_trials=n_trials,
    )
    if not audit["all_invariants_passed"]:
        raise RuntimeError(
            f"Internal {mode.upper()} split invariant validation failed.")

    return SplitResult(
        train=pairs.loc[best_candidate.train_mask].copy(),
        test=pairs.loc[best_candidate.test_mask].copy(),
        dropped=pairs.loc[best_candidate.dropped_mask].copy(),
        audit=audit,
    )


def split_pairs_three_way(
        pairs: pd.DataFrame, mode: SplitMode,
        val_size: float = 0.1, test_size: float = 0.1,
        protein_to_group: dict[Hashable, Hashable] | None = None,
        seed: int = 0, n_trials: int = 100,
    ) -> ThreeWaySplitResult:
    """
    Split PPI pairs into train/validation/test under a C1, C2, or C3 regime.

    C1 uses pairwise-disjoint edge units and requires every validation/test
    endpoint in train. C2 assigns separate validation- and test-novel protein
    groups, with retained heldout edges joining those groups to train groups.
    C3 assigns mutually disjoint protein groups to all three splits.
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be greater than 0 and less than 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.")
    _validate_inputs(
        pairs=pairs,
        mode=mode,
        test_size=val_size + test_size,
        protein_to_group=protein_to_group,
        n_trials=n_trials,
    )
    group_mapping = protein_to_group if mode in {"c2", "c3"} else None
    arrays = _encode_pairs(pairs, group_mapping)
    if mode in {"c2", "c3"} and arrays.n_groups < 3:
        raise ValueError(
            f"{mode.upper()} three-way splitting requires at least "
            "three protein groups.")

    edge_units = _edge_units(arrays) if mode == "c1" else None
    best_candidate = None
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        if mode == "c1":
            assert edge_units is not None
            candidate = _make_three_way_c1_candidate(
                arrays=arrays,
                edge_units=edge_units,
                val_size=val_size,
                test_size=test_size,
                rng=rng,
            )
        else:
            candidate = _make_three_way_group_candidate(
                arrays=arrays,
                mode=mode,
                val_size=val_size,
                test_size=test_size,
                rng=rng,
            )
        if candidate is not None and (
                best_candidate is None
                or candidate.score > best_candidate.score
            ):
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError(
            f"Could not produce a valid three-way {mode.upper()} split after "
            f"{n_trials} trial(s). Try more data, a different seed, "
            "different split sizes, or more trials.")

    audit = _three_way_audit(
        arrays=arrays,
        candidate=best_candidate,
        mode=mode,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        n_trials=n_trials,
    )
    if not audit["all_invariants_passed"]:
        raise RuntimeError(
            f"Internal three-way {mode.upper()} split invariant validation "
            "failed.")

    return ThreeWaySplitResult(
        train=pairs.loc[best_candidate.train_mask].copy(),
        val=pairs.loc[best_candidate.val_mask].copy(),
        test=pairs.loc[best_candidate.test_mask].copy(),
        dropped=pairs.loc[best_candidate.dropped_mask].copy(),
        audit=audit,
    )
