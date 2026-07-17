"""Validation and canonical preparation of PPI input examples."""

import logging

import pandas as pd

from ..datasets.pairs import normalize_labels


REQUIRED_PAIR_COLUMNS = {"protein_a", "protein_b", "label"}
EXPECTED_LABEL_VALUES = {0, 1}
LOGGER = logging.getLogger(__name__)


def validate_binary_labeling(labels: pd.Series, context: str) -> None:
    """Fail fast unless a label series contains both binary classes."""
    observed_labels = set(labels.unique())
    missing_labels = sorted(EXPECTED_LABEL_VALUES - observed_labels)
    if missing_labels:
        raise ValueError(
            f"{context} must contain both labels 0 and 1. "
            f"Missing: {missing_labels}"
        )


def validate_pair_ids(protein_pairs: pd.DataFrame) -> None:
    """Fail fast if pair rows are missing protein identifiers."""
    pair_ids = protein_pairs[["protein_a", "protein_b"]]
    missing_id_mask = pair_ids.isna().any(axis=1)
    blank_id_mask = (
        pair_ids.astype("string")
        .apply(lambda values: values.str.strip().eq(""))
        .any(axis=1)
    )
    missing_id_mask = missing_id_mask | blank_id_mask
    if missing_id_mask.any():
        n_missing_rows = int(missing_id_mask.sum())
        raise ValueError(
            f"pairs.csv has {n_missing_rows} rows with missing protein IDs."
        )


def drop_pairs_missing_sequences(
    protein_pairs: pd.DataFrame,
    sequences: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows where either protein is absent from the FASTA sequences."""
    sequence_ids = pd.Index(sequences)
    has_protein_a = protein_pairs["protein_a"].isin(sequence_ids)
    has_protein_b = protein_pairs["protein_b"].isin(sequence_ids)
    keep_mask = has_protein_a & has_protein_b
    n_dropped = int((~keep_mask).sum())
    dropped_columns = []
    if "source_row_index" in protein_pairs.columns:
        dropped_columns.append("source_row_index")
    dropped_columns.append("drop_reason")
    if "pair_id" in protein_pairs.columns:
        dropped_columns.append("pair_id")

    if n_dropped == 0:
        filtered_pairs = protein_pairs.copy()
        dropped_pairs = pd.DataFrame(columns=dropped_columns)
    else:
        missing_a = protein_pairs.loc[
            ~has_protein_a, "protein_a"
        ].dropna().unique()
        missing_b = protein_pairs.loc[
            ~has_protein_b, "protein_b"
        ].dropna().unique()
        missing_proteins = pd.Index(missing_a).union(
            pd.Index(missing_b),
            sort=False,
        )
        missing_examples = missing_proteins[:10].tolist()
        LOGGER.warning(
            "Dropping %s of %s pairs because one or both proteins are missing "
            "FASTA sequences. Missing protein count: %s. Examples: %s",
            n_dropped,
            len(protein_pairs),
            len(missing_proteins),
            missing_examples,
        )
        filtered_pairs = protein_pairs.loc[keep_mask].copy()
        dropped_pairs = protein_pairs.loc[~keep_mask].copy()
        dropped_pairs["drop_reason"] = "missing_both_sequences"
        dropped_pairs.loc[
            (~has_protein_a) & has_protein_b,
            "drop_reason",
        ] = "missing_protein_a_sequence"
        dropped_pairs.loc[
            has_protein_a & (~has_protein_b),
            "drop_reason",
        ] = "missing_protein_b_sequence"
        dropped_pairs = dropped_pairs[dropped_columns]

    if filtered_pairs.empty:
        raise ValueError(
            "All pairs were dropped because one or both proteins are missing "
            "FASTA sequences."
        )

    return filtered_pairs, dropped_pairs


def prepare_input_data(
    protein_pairs: pd.DataFrame,
    sequences: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate inputs and return label-normalized pairs and dropped rows."""
    if protein_pairs.empty:
        raise ValueError("pairs.csv must contain at least one row.")
    if not sequences:
        raise ValueError("FASTA must contain at least one sequence.")

    missing_cols = REQUIRED_PAIR_COLUMNS - set(protein_pairs.columns)
    if missing_cols:
        raise ValueError(f"pairs.csv missing columns: {missing_cols}")

    prepared_pairs = protein_pairs.copy()
    validate_pair_ids(prepared_pairs)
    prepared_pairs["protein_a"] = (
        prepared_pairs["protein_a"].astype("string").str.strip()
    )
    prepared_pairs["protein_b"] = (
        prepared_pairs["protein_b"].astype("string").str.strip()
    )
    prepared_pairs["label"] = normalize_labels(
        labels=prepared_pairs["label"], context="pairs.csv"
    )
    prepared_pairs, dropped_pairs = drop_pairs_missing_sequences(
        prepared_pairs,
        sequences,
    )
    prepared_pairs = prepared_pairs.reset_index(drop=True)
    validate_binary_labeling(
        labels=prepared_pairs["label"], context="pairs.csv"
    )
    return prepared_pairs, dropped_pairs
