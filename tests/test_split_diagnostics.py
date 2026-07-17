import json

import pandas as pd

from ppi_benchmark.splitting.diagnostics import (
    compute_generic_split_diagnostics,
    compute_ppi_split_diagnostics,
)
from ppi_benchmark.splitting.dispatch import (
    load_split_column,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from ppi_benchmark.splitting.protein_disjoint import split_pairs


def pair_frame(rows):
    """
    Return a compact PPI pair dataframe from tuples.
    """
    return pd.DataFrame(
        rows,
        columns=["protein_a", "protein_b", "label"],
    )


def test_generic_split_sizes_and_labels_without_val():
    train_df = pair_frame([
        ("A", "B", 1),
        ("C", "D", 0),
    ])
    test_df = pair_frame([
        ("E", "F", 1),
    ])

    diagnostics = compute_generic_split_diagnostics(
        train_df=train_df,
        val_df=None,
        test_df=test_df,
    )

    assert diagnostics["n_train"] == 2
    assert diagnostics["n_val"] == 0
    assert diagnostics["n_test"] == 1
    assert diagnostics["actual_train_size"] == 2 / 3
    assert diagnostics["actual_val_size"] == 0.0
    assert diagnostics["label_counts_train"] == {"0": 1, "1": 1}
    assert diagnostics["label_counts_val"] == {}


def test_generic_split_sizes_and_labels_with_val():
    train_df = pair_frame([
        ("A", "B", 1),
        ("C", "D", 0),
    ])
    val_df = pair_frame([
        ("E", "F", 0),
    ])
    test_df = pair_frame([
        ("G", "H", 1),
    ])

    diagnostics = compute_generic_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert diagnostics["n_train"] == 2
    assert diagnostics["n_val"] == 1
    assert diagnostics["n_test"] == 1
    assert diagnostics["actual_val_size"] == 0.25
    assert diagnostics["label_counts_val"] == {"0": 1}


def test_shared_protein_counts_and_examples():
    train_df = pair_frame([
        ("A", "B", 1),
        ("C", "D", 0),
    ])
    val_df = pair_frame([
        ("B", "E", 1),
    ])
    test_df = pair_frame([
        ("C", "F", 0),
    ])

    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert diagnostics["n_shared_proteins_train_val"] == 1
    assert diagnostics["shared_proteins_train_val_examples"] == ["B"]
    assert diagnostics["n_shared_proteins_train_test"] == 1
    assert diagnostics["shared_proteins_train_test_examples"] == ["C"]
    assert diagnostics["n_shared_proteins_val_test"] == 0
    assert diagnostics["has_shared_proteins_across_splits"] is True
    assert diagnostics["has_pair_leakage_across_splits"] is False


def test_exact_ordered_pair_overlap():
    train_df = pair_frame([
        ("A", "B", 1),
    ])
    val_df = pair_frame([
        ("A", "B", 0),
    ])
    test_df = pair_frame([
        ("X", "Y", 1),
    ])

    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert diagnostics["n_exact_ordered_pair_overlaps_train_val"] == 1
    assert diagnostics["exact_ordered_pair_overlap_examples_train_val"] == [
        ["A", "B"],
    ]


def test_reversed_pair_counts_only_as_unordered_overlap():
    train_df = pair_frame([
        ("A", "B", 1),
    ])
    val_df = pair_frame([
        ("B", "A", 0),
    ])
    test_df = pair_frame([
        ("X", "Y", 1),
    ])

    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert diagnostics["n_exact_ordered_pair_overlaps_train_val"] == 0
    assert diagnostics["n_unordered_pair_overlaps_train_val"] == 1
    assert diagnostics["unordered_pair_overlap_examples_train_val"] == [
        ["A", "B"],
    ]
    assert diagnostics["has_exact_ordered_pair_overlap_across_splits"] is False
    assert diagnostics["has_unordered_pair_overlap_across_splits"] is True
    assert diagnostics["has_pair_leakage_across_splits"] is True


def test_within_split_duplicate_pair_counts():
    train_df = pair_frame([
        ("A", "B", 1),
        ("A", "B", 0),
        ("B", "A", 1),
        ("C", "D", 0),
    ])
    test_df = pair_frame([
        ("X", "Y", 1),
    ])

    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=None,
        test_df=test_df,
    )

    assert diagnostics["n_duplicate_ordered_pairs_train"] == 1
    assert diagnostics["n_duplicate_unordered_pairs_train"] == 2
    json.dumps(diagnostics)


def test_provided_split_diagnostics_use_actual_split_rows():
    pairs = pair_frame([
        ("A", "B", 1),
        ("C", "D", 0),
        ("E", "F", 1),
        ("G", "H", 0),
        ("I", "J", 1),
        ("K", "L", 0),
    ])
    pairs["split"] = [
        TRAIN_SPLIT,
        TRAIN_SPLIT,
        VAL_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT,
        TEST_SPLIT,
    ]

    train_df, val_df, test_df = load_split_column(
        pairs=pairs,
        split_col="split",
        val_size=0.0,
    )
    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert diagnostics["n_train"] == 2
    assert diagnostics["n_val"] == 2
    assert diagnostics["n_test"] == 2
    assert diagnostics["actual_val_size"] == 1 / 3
    assert diagnostics["label_counts_val"] == {"0": 1, "1": 1}


def test_c3_reports_zero_shared_proteins():
    pairs = pair_frame([
        (f"P{left}", f"P{right}", (left + right) % 2)
        for left in range(12)
        for right in range(left + 1, 12)
    ])
    split_result = split_pairs(
        pairs,
        mode="c3",
        test_size=0.3,
        seed=5,
        n_trials=20,
    )
    diagnostics = compute_ppi_split_diagnostics(
        train_df=split_result.train,
        val_df=None,
        test_df=split_result.test,
    )

    assert diagnostics["n_shared_proteins_train_test"] == 0
    assert diagnostics["has_shared_proteins_across_splits"] is False
    assert diagnostics["has_pair_leakage_across_splits"] is False
