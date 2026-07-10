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
