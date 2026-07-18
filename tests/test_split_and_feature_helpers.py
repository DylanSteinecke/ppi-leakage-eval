from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from ppi_benchmark import features as ppi_features
from ppi_benchmark.features import (
    build_feature_matrices,
    configured_feature_identity,
    fitted_vectorizer_sha256,
    make_vectorizer,
)
from ppi_benchmark.matrix_provenance import canonicalize_matrix
from ppi_benchmark.splitting.dispatch import (
    load_split_column,
    make_random_pair_split,
    validate_splits,
)
from ppi_benchmark.splitting.preparation import prepare_input_data


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


def split_args(train_size=0.5, val_size=0.0, seed=0):
    """
    Return minimal split args.
    """
    args = SimpleNamespace(
        train_size=train_size,
        val_size=val_size,
        seed=seed,
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


def test_validate_splits_rejects_empty_required_split():
    pairs = balanced_pairs(4)

    with pytest.raises(ValueError, match="train split must not be empty"):
        validate_splits(
            train_df=pairs.iloc[0:0],
            val_df=None,
            test_df=pairs,
        )


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


def test_prepare_input_data_normalizes_ids_without_source_metadata():
    pairs = pd.DataFrame({
        "protein_a": [" A ", 1, "MISSING"],
        "protein_b": ["B", 2, "B"],
        "label": [0, 1, 0],
    })
    sequences = {
        "A": "AAAA",
        "B": "BBBB",
        "1": "CCCC",
        "2": "DDDD",
    }

    prepared, dropped = prepare_input_data(pairs, sequences)

    assert prepared[["protein_a", "protein_b"]].values.tolist() == [
        ["A", "B"],
        ["1", "2"],
    ]
    assert dropped.columns.tolist() == ["drop_reason"]
    assert dropped["drop_reason"].tolist() == [
        "missing_protein_a_sequence",
    ]


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

    x_train, x_val, x_test, fitted_extractor_sha256 = build_feature_matrices(
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
    assert len(fitted_extractor_sha256) == 64
    assert x_train.has_canonical_format
    assert x_train.has_sorted_indices
    assert canonicalize_matrix(x_train) is x_train


def test_logical_feature_identity_excludes_split_fitted_state():
    first_args = SimpleNamespace(
        k=2,
        bm25_k1=1.5,
        bm25_b=0.75,
        split_seed=3,
    )
    second_args = SimpleNamespace(
        k=2,
        bm25_k1=1.5,
        bm25_b=0.75,
        split_seed=17,
    )
    first_vectorizer = make_vectorizer("count", first_args)
    second_vectorizer = make_vectorizer("count", second_args)
    first_vectorizer.fit(["AAAA", "AATA"])
    second_vectorizer.fit(["CCCC", "CGCC"])

    first_identity = configured_feature_identity(("count",), first_args)
    second_identity = configured_feature_identity(("count",), second_args)
    first_fitted_hash = fitted_vectorizer_sha256(
        "count", first_vectorizer)
    second_fitted_hash = fitted_vectorizer_sha256(
        "count", second_vectorizer)

    assert first_identity == second_identity
    assert len(first_identity.feature_spec_sha256) == 64
    assert first_fitted_hash != second_fitted_hash


def test_build_feature_matrices_transforms_each_unique_protein_once(monkeypatch):
    transform_calls = []

    class RecordingVectorizer:
        """
        Record sequence batches while delegating count-vectorizer behavior.
        """

        def __init__(self):
            self.vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=(2, 2),
                lowercase=False,
            )

        def fit(self, sequences):
            self.vectorizer.fit(sequences)
            return self

        def transform(self, sequences):
            transform_calls.append(list(sequences))
            return self.vectorizer.transform(sequences)

        @property
        def vocabulary_(self):
            return self.vectorizer.vocabulary_

    monkeypatch.setattr(
        ppi_features,
        "make_vectorizer",
        lambda feature_type, args: RecordingVectorizer(),
    )
    train_df = pd.DataFrame({
        "protein_a": ["A", "A"],
        "protein_b": ["B", "B"],
        "label": [0, 1],
    })
    val_df = pd.DataFrame({
        "protein_a": ["A"],
        "protein_b": ["C"],
        "label": [0],
    })
    test_df = pd.DataFrame({
        "protein_a": ["B"],
        "protein_b": ["D"],
        "label": [1],
    })
    sequences = {
        "A": "AAAA",
        "B": "BBBB",
        "C": "CCCC",
        "D": "DDDD",
    }
    args = SimpleNamespace(k=2, bm25_k1=1.5, bm25_b=0.75)

    build_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sequences=sequences,
        feature_types=("count",),
        args=args,
    )

    assert transform_calls == [["AAAA", "BBBB", "CCCC", "DDDD"]]
