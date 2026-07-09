import json
from types import SimpleNamespace

import pandas as pd

from ppi_inputs import (
    load_split_column,
    make_protein_component_split,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from split_diagnostics import (
    compute_generic_split_diagnostics,
    compute_ppi_split_diagnostics,
)


def pair_frame(rows):
    """
    Return a compact PPI pair dataframe from tuples.
    """
    return pd.DataFrame(
        rows,
        columns=["protein_a", "protein_b", "label"],
    )


def multi_component_pairs(n_components=5):
    """
    Return disconnected pair components with both labels in each component.
    """
    rows = []
    for component_index in range(n_components):
        protein_a = f"C{component_index}_A"
        protein_b = f"C{component_index}_B"
        protein_c = f"C{component_index}_C"
        rows.extend([
            (protein_a, protein_b, 0),
            (protein_b, protein_c, 1),
        ])

    return pair_frame(rows)


def split_args(train_size=0.6, val_size=0.2, seed=5):
    """
    Return minimal args for split helper tests.
    """
    return SimpleNamespace(
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        n_pruned_pairs=0,
        pruned_pair_fraction=0.0,
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


def test_protein_disjoint_components_reports_zero_shared_proteins():
    pairs = multi_component_pairs()

    train_df, val_df, test_df = make_protein_component_split(
        pairs,
        split_args(),
    )
    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert diagnostics["n_shared_proteins_train_val"] == 0
    assert diagnostics["n_shared_proteins_train_test"] == 0
    assert diagnostics["n_shared_proteins_val_test"] == 0
