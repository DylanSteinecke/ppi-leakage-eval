from types import SimpleNamespace

import pandas as pd
import pytest

from ppi_features import build_feature_matrices
from ppi_inputs import (
    load_split_column,
    make_protein_component_split,
    make_protein_prune_split,
    make_random_pair_split,
    protein_ids_in_pairs,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    validate_disjoint_splits,
)


def balanced_pairs(n_pairs=24):
    """
    Return a balanced pair table for stratified split tests.
    """
    rows = []
    for index in range(n_pairs):
        rows.append({
            "protein_a": f"P{index:02d}A",
            "protein_b": f"P{index:02d}B",
            "label": index % 2,
        })

    return pd.DataFrame(rows)


def complete_graph_pairs(n_proteins=8):
    """
    Return all undirected pairs among proteins for prune-split tests.
    """
    rows = []
    proteins = [f"P{index}" for index in range(n_proteins)]
    for left_index, protein_a in enumerate(proteins[:-1]):
        for right_index, protein_b in enumerate(proteins[left_index + 1:]):
            rows.append({
                "protein_a": protein_a,
                "protein_b": protein_b,
                "label": (left_index + right_index) % 2,
            })

    return pd.DataFrame(rows)


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
            {
                "protein_a": protein_a,
                "protein_b": protein_b,
                "label": 0,
            },
            {
                "protein_a": protein_b,
                "protein_b": protein_c,
                "label": 1,
            },
        ])

    return pd.DataFrame(rows)


def split_args(train_size=0.5, val_size=0.0, seed=0):
    """
    Return minimal split args.
    """
    args = SimpleNamespace(
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        n_pruned_pairs=0,
        pruned_pair_fraction=0.0,
    )

    return args


def assert_has_both_labels(split_df):
    """
    Assert that one split contains both binary labels.
    """
    assert set(split_df["label"]) == {0, 1}


def test_random_pair_split_without_validation_has_binary_train_test():
    pairs = balanced_pairs()

    train_df, val_df, test_df = make_random_pair_split(
        pairs,
        split_args(train_size=0.5, val_size=0.0, seed=1),
    )

    assert val_df is None
    assert len(train_df) + len(test_df) == len(pairs)
    assert_has_both_labels(train_df)
    assert_has_both_labels(test_df)


def test_random_pair_split_with_validation_has_binary_non_empty_splits():
    pairs = balanced_pairs()

    train_df, val_df, test_df = make_random_pair_split(
        pairs,
        split_args(train_size=0.5, val_size=0.25, seed=2),
    )

    assert val_df is not None
    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty
    assert_has_both_labels(train_df)
    assert_has_both_labels(val_df)
    assert_has_both_labels(test_df)


def test_protein_prune_split_is_disjoint_without_validation():
    pairs = complete_graph_pairs()

    train_df, val_df, test_df = make_protein_prune_split(
        pairs,
        split_args(train_size=0.6, val_size=0.0, seed=3),
    )

    assert val_df is None
    assert protein_ids_in_pairs(train_df).isdisjoint(protein_ids_in_pairs(test_df))


def test_protein_prune_split_is_disjoint_with_validation():
    pairs = complete_graph_pairs()

    train_df, val_df, test_df = make_protein_prune_split(
        pairs,
        split_args(train_size=0.5, val_size=0.25, seed=4),
    )

    validate_disjoint_splits({
        TRAIN_SPLIT: train_df,
        VAL_SPLIT: val_df,
        TEST_SPLIT: test_df,
    })


def test_protein_component_split_is_disjoint_with_validation():
    pairs = multi_component_pairs()

    train_df, val_df, test_df = make_protein_component_split(
        pairs,
        split_args(train_size=0.6, val_size=0.2, seed=5),
    )

    validate_disjoint_splits({
        TRAIN_SPLIT: train_df,
        VAL_SPLIT: val_df,
        TEST_SPLIT: test_df,
    })


def test_load_split_column_accepts_train_val_test_labels():
    pairs = balanced_pairs(12)
    pairs["split"] = [
        "train", "train", "train", "train",
        "val", "val", "val", "val",
        "test", "test", "test", "test",
    ]

    train_df, val_df, test_df = load_split_column(
        pairs,
        split_col="split",
        val_size=0.25,
    )

    assert len(train_df) == 4
    assert val_df is not None
    assert len(val_df) == 4
    assert len(test_df) == 4


def test_load_split_column_rejects_unexpected_labels():
    pairs = balanced_pairs(6)
    pairs["split"] = ["train", "train", "val", "test", "holdout", "test"]

    with pytest.raises(ValueError, match="Found"):
        load_split_column(pairs, split_col="split", val_size=0.0)


def test_load_split_column_requires_val_rows_when_requested():
    pairs = balanced_pairs(6)
    pairs["split"] = ["train", "train", "train", "test", "test", "test"]

    with pytest.raises(ValueError, match="'val' rows"):
        load_split_column(pairs, split_col="split", val_size=0.25)


def test_build_feature_matrices_fits_vectorizer_on_train_only():
    train_df = pd.DataFrame({
        "protein_a": ["train_a"],
        "protein_b": ["train_b"],
        "label": [1],
    })
    val_df = pd.DataFrame({
        "protein_a": ["val_a"],
        "protein_b": ["val_b"],
        "label": [0],
    })
    test_df = pd.DataFrame({
        "protein_a": ["test_a"],
        "protein_b": ["test_b"],
        "label": [0],
    })
    sequences = {
        "train_a": "AAAA",
        "train_b": "AAAA",
        "val_a": "CCCC",
        "val_b": "CCCC",
        "test_a": "GGGG",
        "test_b": "GGGG",
    }
    args = SimpleNamespace(k=2, bm25_k1=1.5, bm25_b=0.75)

    x_train, x_val, x_test = build_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sequences=sequences,
        feature_types=("count",),
        args=args,
    )

    assert x_train.shape[1] == 3
    assert x_val.shape[1] == x_train.shape[1]
    assert x_test.shape[1] == x_train.shape[1]
