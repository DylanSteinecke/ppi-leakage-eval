"""
Reusable split diagnostics for PPI pipeline runs.

The helpers in this module accept plain split-to-set mappings where possible so
future task wrappers can reuse the overlap logic. For a PTM prediction wrapper,
the task-specific layer would extract protein IDs, sites such as
``(protein_id, residue_index)``, residues, PTM types, and local sequence-window
identifiers from each split, then pass those sets into the generic overlap
helpers below.
"""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import pandas as pd

from ppi_inputs import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT, protein_ids_in_pairs


DEFAULT_SPLIT_ORDER = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
MAX_EXAMPLES = 10


def native_value(value: Any) -> Any:
    """
    Convert common pandas/numpy scalar containers to JSON-native values.
    """
    if isinstance(value, tuple):
        return [native_value(item) for item in value]
    if isinstance(value, list):
        return [native_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(native_value(key)): native_value(dict_value)
            for key, dict_value in value.items()
        }
    if isinstance(value, set):
        return [native_value(item) for item in sorted(value, key=sort_key)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    item_method = getattr(value, "item", None)
    if item_method is not None:
        try:
            item_value = item_method()
        except (TypeError, ValueError):
            item_value = value
        else:
            return native_value(item_value)

    return str(value)


def sort_key(value: Any) -> tuple[str, ...]:
    """
    Return a deterministic string sort key for scalar or tuple examples.
    """
    if isinstance(value, tuple):
        key = tuple(str(item) for item in value)
    else:
        key = (str(value),)

    return key


def capped_examples(
        values: set[Hashable], max_examples: int = MAX_EXAMPLES,
    ) -> list[Any]:
    """
    Return deterministic, JSON-friendly examples from a set.
    """
    examples = [
        native_value(value)
        for value in sorted(values, key=sort_key)[:max_examples]
    ]

    return examples


def split_pairs(
        split_to_values: Mapping[str, Any],
    ) -> list[tuple[str, str]]:
    """
    Return ordered split-name pairs, including custom split names if present.
    """
    ordered_names = [
        split_name
        for split_name in DEFAULT_SPLIT_ORDER
        if split_name in split_to_values
    ]
    extra_names = sorted(
        set(split_to_values) - set(ordered_names),
        key=str,
    )
    ordered_names.extend(extra_names)

    pairs = [
        (left_name, right_name)
        for left_index, left_name in enumerate(ordered_names[:-1])
        for right_name in ordered_names[left_index + 1:]
    ]

    return pairs


def set_overlap_diagnostics(
        split_to_sets: Mapping[str, set[Hashable]],
        count_prefix: str,
        example_prefix: str | None = None,
        max_examples: int = MAX_EXAMPLES,
        example_key_style: str = "suffix",
    ) -> dict[str, Any]:
    """
    Return pairwise intersection counts and capped examples for split sets.
    """
    example_prefix = count_prefix if example_prefix is None else example_prefix
    diagnostics = {}
    for left_name, right_name in split_pairs(split_to_sets):
        overlap = split_to_sets[left_name] & split_to_sets[right_name]
        diagnostics[f"n_{count_prefix}_{left_name}_{right_name}"] = int(
            len(overlap))
        if example_key_style == "before_splits":
            example_key = f"{example_prefix}_examples_{left_name}_{right_name}"
        else:
            example_key = f"{example_prefix}_{left_name}_{right_name}_examples"
        diagnostics[example_key] = capped_examples(
            overlap,
            max_examples=max_examples,
        )

    return diagnostics


def duplicate_value_diagnostics(
        split_to_values: Mapping[str, Sequence[Hashable]],
        value_name: str,
    ) -> dict[str, int]:
    """
    Return duplicate counts within each split as rows beyond unique values.
    """
    diagnostics = {}
    for split_name, values in split_to_values.items():
        diagnostics[f"n_duplicate_{value_name}_{split_name}"] = int(
            len(values) - len(set(values)))

    return diagnostics


def unique_value_count_diagnostics(
        split_to_sets: Mapping[str, set[Hashable]],
        value_name: str,
    ) -> dict[str, int]:
    """
    Return unique value counts within each split.
    """
    diagnostics = {
        f"n_unique_{value_name}_{split_name}": int(len(values))
        for split_name, values in split_to_sets.items()
    }

    return diagnostics


def label_counts(split_df: pd.DataFrame | None) -> dict[str, int]:
    """
    Return JSON-friendly label counts for one optional dataframe.
    """
    if split_df is None or split_df.empty:
        counts = {}
    else:
        counts = {
            str(native_value(label)): int(count)
            for label, count in split_df["label"].value_counts(
                sort=False,
            ).sort_index().items()
        }

    return counts


def split_size_and_label_diagnostics(
        split_dfs: Mapping[str, pd.DataFrame | None],
    ) -> dict[str, Any]:
    """
    Return split row counts, realized fractions, and label counts.
    """
    split_sizes = {
        split_name: 0 if split_df is None else int(len(split_df))
        for split_name, split_df in split_dfs.items()
    }
    n_total = sum(split_sizes.values())

    diagnostics: dict[str, Any] = {}
    for split_name in split_dfs:
        split_size = split_sizes[split_name]
        diagnostics[f"n_{split_name}"] = split_size
        diagnostics[f"actual_{split_name}_size"] = (
            0.0 if n_total == 0 else float(split_size / n_total)
        )
        diagnostics[f"label_counts_{split_name}"] = label_counts(
            split_dfs[split_name])

    return diagnostics


def ppi_protein_ids(split_df: pd.DataFrame | None) -> set[Hashable]:
    """
    Return protein IDs in one optional PPI split dataframe.
    """
    if split_df is None or split_df.empty:
        protein_ids = set()
    else:
        protein_ids = protein_ids_in_pairs(split_df)

    return set(protein_ids)


def ordered_ppi_pairs(
        split_df: pd.DataFrame | None,
    ) -> list[tuple[Hashable, Hashable]]:
    """
    Return ordered ``(protein_a, protein_b)`` pairs for one PPI split.
    """
    if split_df is None or split_df.empty:
        pairs = []
    else:
        pairs = list(zip(split_df["protein_a"], split_df["protein_b"]))

    return pairs


def unordered_ppi_pairs(
        split_df: pd.DataFrame | None,
    ) -> list[tuple[Hashable, Hashable]]:
    """
    Return unordered sorted protein-pair tuples for one PPI split.
    """
    pairs = [
        tuple(sorted((protein_a, protein_b), key=str))
        for protein_a, protein_b in ordered_ppi_pairs(split_df)
    ]

    return pairs


def split_dataframe_mapping(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame | None]:
    """
    Return a canonical train/val/test dataframe mapping.
    """
    split_dfs = {
        TRAIN_SPLIT: train_df,
        VAL_SPLIT: val_df,
        TEST_SPLIT: test_df,
    }

    return split_dfs


def compute_generic_split_diagnostics(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
    """
    Return generic row-count and label-count diagnostics for train/val/test.
    """
    diagnostics = split_size_and_label_diagnostics(
        split_dataframe_mapping(train_df, val_df, test_df))

    return diagnostics


def compute_ppi_split_diagnostics(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, args: Any | None = None,
        protein_pairs: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
    """
    Return non-fatal PPI split diagnostics as JSON-serializable values.
    """
    split_dfs = split_dataframe_mapping(train_df, val_df, test_df)
    diagnostics = split_size_and_label_diagnostics(split_dfs)

    if protein_pairs is not None:
        diagnostics["label_counts_total_after_filtering"] = label_counts(
            protein_pairs)
        diagnostics["n_unique_proteins_total_after_filtering"] = int(
            len(protein_ids_in_pairs(protein_pairs)))

    split_to_proteins = {
        split_name: ppi_protein_ids(split_df)
        for split_name, split_df in split_dfs.items()
    }
    diagnostics.update(
        unique_value_count_diagnostics(split_to_proteins, "proteins"))
    diagnostics.update(
        set_overlap_diagnostics(
            split_to_sets=split_to_proteins,
            count_prefix="shared_proteins",
            example_prefix="shared_proteins",
        ))

    split_to_ordered_pairs = {
        split_name: ordered_ppi_pairs(split_df)
        for split_name, split_df in split_dfs.items()
    }
    split_to_unordered_pairs = {
        split_name: unordered_ppi_pairs(split_df)
        for split_name, split_df in split_dfs.items()
    }
    split_to_ordered_pair_sets = {
        split_name: set(pairs)
        for split_name, pairs in split_to_ordered_pairs.items()
    }
    split_to_unordered_pair_sets = {
        split_name: set(pairs)
        for split_name, pairs in split_to_unordered_pairs.items()
    }

    diagnostics.update(
        unique_value_count_diagnostics(
            split_to_ordered_pair_sets,
            "ordered_pairs",
        ))
    diagnostics.update(
        unique_value_count_diagnostics(
            split_to_unordered_pair_sets,
            "unordered_pairs",
        ))
    diagnostics.update(
        set_overlap_diagnostics(
            split_to_sets=split_to_ordered_pair_sets,
            count_prefix="exact_ordered_pair_overlaps",
            example_prefix="exact_ordered_pair_overlap",
            example_key_style="before_splits",
        ))
    diagnostics.update(
        set_overlap_diagnostics(
            split_to_sets=split_to_unordered_pair_sets,
            count_prefix="unordered_pair_overlaps",
            example_prefix="unordered_pair_overlap",
            example_key_style="before_splits",
        ))
    diagnostics.update(
        duplicate_value_diagnostics(
            split_to_ordered_pairs,
            "ordered_pairs",
        ))
    diagnostics.update(
        duplicate_value_diagnostics(
            split_to_unordered_pairs,
            "unordered_pairs",
        ))

    diagnostics["n_connected_components"] = native_value(
        getattr(args, "n_connected_components", None))
    diagnostics["n_pruned_pairs"] = native_value(
        getattr(args, "n_pruned_pairs", 0))
    diagnostics["pruned_pair_fraction"] = native_value(
        getattr(args, "pruned_pair_fraction", 0.0))

    return native_value(diagnostics)
