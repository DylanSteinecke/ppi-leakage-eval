"""
Deterministic, task-neutral cohort sampling.

The sampler works with example identities and optional categorical strata. It
does not know whether an example is a PPI pair, a PTM site, or another task's
row. Task-specific wrappers are responsible for choosing identities, strata,
and any downstream validity checks.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SAMPLING_ALGORITHM = "stable_hash_stratified_sainte_lague"
SAMPLING_ALGORITHM_VERSION = 1
SAMPLING_DIRNAME = "sampling"
SELECTED_EXAMPLES_FILENAME = "selected_examples.csv"
UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class SamplingSpec:
    """Configuration for selecting one cohort of examples."""

    max_examples: int | None = None
    fraction: float | None = None
    seed: int = 0
    minimum_per_stratum: int = 0


@dataclass(frozen=True)
class SamplingResult:
    """Selected input positions and audit metadata."""

    selected_positions: np.ndarray
    sampling_ranks: np.ndarray
    metadata: dict[str, Any]


def _native_value(value: Any) -> Any:
    """Return a JSON-compatible scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _validate_spec(spec: SamplingSpec) -> None:
    """Validate sampling settings independently of a cohort."""
    if spec.max_examples is not None and spec.fraction is not None:
        raise ValueError("Set only one of max_examples and fraction.")
    if spec.max_examples is not None and spec.max_examples < 1:
        raise ValueError("max_examples must be at least 1.")
    if spec.fraction is not None and not 0.0 < spec.fraction < 1.0:
        raise ValueError("fraction must be greater than 0 and less than 1.")
    if spec.minimum_per_stratum < 0:
        raise ValueError("minimum_per_stratum must be nonnegative.")


def _normalize_example_ids(example_ids: Sequence[Any]) -> pd.Series:
    """Return validated, index-free example identities."""
    normalized_ids = pd.Series(example_ids, copy=False).reset_index(drop=True)
    if normalized_ids.isna().any():
        raise ValueError("example_ids must not contain missing values.")
    if normalized_ids.duplicated().any():
        raise ValueError("example_ids must be unique.")
    return normalized_ids


def _normalize_strata(
        strata: pd.DataFrame | pd.Series | None, n_examples: int,
    ) -> pd.DataFrame:
    """Return a validated stratum dataframe with at least one column."""
    if strata is None:
        normalized = pd.DataFrame({"stratum": ["all"] * n_examples})
    elif isinstance(strata, pd.Series):
        column_name = strata.name if strata.name is not None else "stratum"
        normalized = strata.rename(column_name).to_frame().reset_index(
            drop=True)
    else:
        normalized = strata.reset_index(drop=True).copy()

    if len(normalized) != n_examples:
        raise ValueError("strata and example_ids must have the same length.")
    if normalized.shape[1] == 0:
        raise ValueError("strata must contain at least one column.")
    if not normalized.columns.is_unique:
        raise ValueError("strata column names must be unique.")
    if normalized.isna().any(axis=None):
        raise ValueError("strata must not contain missing values.")
    return normalized


def _target_count(n_examples: int, spec: SamplingSpec) -> int:
    """Resolve the requested number of selected examples."""
    if spec.max_examples is not None:
        return min(spec.max_examples, n_examples)
    if spec.fraction is not None:
        return min(
            max(1, int(math.floor(n_examples * spec.fraction))),
            n_examples,
        )
    return n_examples


def _stable_hash_priorities(example_ids: pd.Series, seed: int) -> np.ndarray:
    """Return deterministic uint64 priorities for example identities."""
    base_hashes = pd.util.hash_pandas_object(
        example_ids,
        index=False,
        hash_key="ppi-sampling-v01",
        categorize=False,
    ).to_numpy(dtype=np.uint64, copy=False)
    seed_value = np.uint64(seed & UINT64_MASK)
    with np.errstate(over="ignore"):
        values = base_hashes + seed_value + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30)))
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27)))
        values *= np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    return values


def _stratum_key(values: tuple[Any, ...]) -> str:
    """Return a stable string used only to break allocation ties."""
    native_values = [_native_value(value) for value in values]
    return json.dumps(native_values, sort_keys=True, default=str)


def _group_positions(
        strata: pd.DataFrame,
    ) -> tuple[list[tuple[Any, ...]], list[np.ndarray]]:
    """Return deterministically ordered stratum keys and row positions."""
    grouped = strata.groupby(
        list(strata.columns),
        sort=False,
        dropna=False,
        observed=True,
    ).indices
    items = []
    for raw_key, raw_positions in grouped.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        items.append((key, np.asarray(raw_positions, dtype=np.int64)))
    items.sort(key=lambda item: _stratum_key(item[0]))
    return [item[0] for item in items], [item[1] for item in items]


def _allocation_schedule(
        group_sizes: np.ndarray, target_count: int,
        minimum_per_stratum: int,
    ) -> tuple[np.ndarray, list[int]]:
    """Allocate a nested, approximately proportional sample across groups."""
    minimums = np.minimum(group_sizes, minimum_per_stratum).astype(np.int64)
    required_count = int(minimums.sum())
    if target_count < required_count:
        raise ValueError(
            f"Sampling {target_count} examples cannot satisfy "
            f"minimum_per_stratum={minimum_per_stratum} across "
            f"{len(group_sizes)} strata; at least {required_count} examples "
            "are required."
        )

    allocations = np.zeros(len(group_sizes), dtype=np.int64)
    schedule: list[int] = []
    for _ in range(minimum_per_stratum):
        for group_index, group_size in enumerate(group_sizes):
            if allocations[group_index] < min(group_size, minimum_per_stratum):
                allocations[group_index] += 1
                schedule.append(group_index)

    heap: list[tuple[float, int]] = []
    for group_index, group_size in enumerate(group_sizes):
        if allocations[group_index] < group_size:
            priority = group_size / (2 * allocations[group_index] + 1)
            heapq.heappush(heap, (-priority, group_index))

    while len(schedule) < target_count:
        if not heap:
            raise RuntimeError("Sampling allocation exhausted unexpectedly.")
        _, group_index = heapq.heappop(heap)
        allocations[group_index] += 1
        schedule.append(group_index)
        if allocations[group_index] < group_sizes[group_index]:
            priority = (
                group_sizes[group_index]
                / (2 * allocations[group_index] + 1)
            )
            heapq.heappush(heap, (-priority, group_index))

    return allocations, schedule


def _selection_sha256(example_ids: pd.Series) -> str:
    """Hash the ordered selected example identities for auditability."""
    digest = hashlib.sha256()
    for example_id in example_ids:
        encoded = json.dumps(
            _native_value(example_id),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def select_examples(
        example_ids: Sequence[Any], spec: SamplingSpec,
        strata: pd.DataFrame | pd.Series | None = None,
    ) -> SamplingResult:
    """Select a deterministic whole-cohort sample without replacement."""
    _validate_spec(spec)
    normalized_ids = _normalize_example_ids(example_ids)
    n_examples = len(normalized_ids)
    if n_examples == 0:
        raise ValueError("Cannot sample an empty cohort.")
    normalized_strata = _normalize_strata(strata, n_examples)
    target_count = _target_count(n_examples, spec)
    group_keys, positions_by_group = _group_positions(normalized_strata)
    group_sizes = np.asarray(
        [len(positions) for positions in positions_by_group],
        dtype=np.int64,
    )

    sampling_requested = (
        spec.max_examples is not None or spec.fraction is not None)
    applied = target_count < n_examples
    if applied:
        allocations, schedule = _allocation_schedule(
            group_sizes=group_sizes,
            target_count=target_count,
            minimum_per_stratum=spec.minimum_per_stratum,
        )
        priorities = _stable_hash_priorities(normalized_ids, spec.seed)
        selected_by_group: list[np.ndarray] = []
        for positions, allocation in zip(positions_by_group, allocations):
            group_priorities = priorities[positions]
            allocation = int(allocation)
            if allocation == 0:
                chosen = np.asarray([], dtype=np.int64)
            elif allocation == len(positions):
                chosen = positions[np.argsort(group_priorities, kind="stable")]
            else:
                candidate_indices = np.argpartition(
                    group_priorities,
                    allocation - 1,
                )[:allocation]
                ordered_candidates = candidate_indices[
                    np.argsort(
                        group_priorities[candidate_indices],
                        kind="stable",
                    )
                ]
                chosen = positions[ordered_candidates]
            selected_by_group.append(chosen)

        next_in_group = np.zeros(len(group_keys), dtype=np.int64)
        ranked_positions = np.empty(target_count, dtype=np.int64)
        for rank_index, group_index in enumerate(schedule):
            within_group_index = next_in_group[group_index]
            ranked_positions[rank_index] = selected_by_group[group_index][
                within_group_index]
            next_in_group[group_index] += 1
        sampling_rank_by_position = {
            int(position): rank
            for rank, position in enumerate(ranked_positions, start=1)
        }
        selected_positions = np.sort(ranked_positions)
        sampling_ranks = np.asarray(
            [sampling_rank_by_position[int(pos)] for pos in selected_positions],
            dtype=np.int64,
        )
    else:
        allocations = group_sizes.copy()
        selected_positions = np.arange(n_examples, dtype=np.int64)
        sampling_ranks = np.arange(1, n_examples + 1, dtype=np.int64)

    selected_ids = normalized_ids.iloc[selected_positions]
    stratum_counts = []
    for key, eligible_count, selected_count in zip(
            group_keys, group_sizes, allocations):
        stratum_counts.append({
            "stratum": {
                str(column): _native_value(value)
                for column, value in zip(normalized_strata.columns, key)
            },
            "n_eligible": int(eligible_count),
            "n_selected": int(selected_count),
        })

    metadata = {
        "requested": sampling_requested,
        "applied": applied,
        "algorithm": SAMPLING_ALGORITHM,
        "algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "seed": int(spec.seed),
        "max_examples": spec.max_examples,
        "fraction": spec.fraction,
        "minimum_per_stratum": int(spec.minimum_per_stratum),
        "stratum_columns": [str(column) for column in normalized_strata],
        "n_eligible": int(n_examples),
        "n_selected": int(len(selected_positions)),
        "n_excluded": int(n_examples - len(selected_positions)),
        "selection_sha256": (
            _selection_sha256(selected_ids) if sampling_requested else None
        ),
        "stratum_counts": stratum_counts,
    }
    return SamplingResult(
        selected_positions=selected_positions,
        sampling_ranks=sampling_ranks,
        metadata=metadata,
    )


def write_selection_manifest(
        selection_manifest: pd.DataFrame | None, output_path: Path,
        append_results: bool,
    ) -> None:
    """Write or validate the selected-example audit artifact."""
    if selection_manifest is None:
        if append_results and output_path.exists():
            raise ValueError(
                f"Cannot append results to {output_path.parent.parent}: the "
                "existing run uses cohort sampling but this invocation does "
                "not. Use a new --run-dir or matching sampling arguments."
            )
        if not append_results and output_path.exists():
            output_path.unlink()
        return

    if append_results:
        if not output_path.exists():
            raise ValueError(
                f"Cannot append results to {output_path.parent.parent}: the "
                "existing run has no sampling manifest. Use a new --run-dir "
                "or matching sampling arguments."
            )
        existing_manifest = pd.read_csv(output_path)
        try:
            pd.testing.assert_frame_equal(
                existing_manifest,
                selection_manifest,
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Cannot append results to {output_path.parent.parent}: "
                "selected_examples.csv does not match this invocation. Use "
                "a new --run-dir or matching sampling arguments."
            ) from exc
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest.to_csv(output_path, index=False)
