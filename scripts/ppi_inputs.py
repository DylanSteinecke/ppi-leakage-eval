"""
Input loading, validation, and split helpers for PPI pipelines.

This module owns the boundary between raw pipeline inputs and trusted in-memory
tables. Future additions should include new input formats, stricter QC checks,
leakage checks, and alternative train/test split strategies.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_PAIR_COLUMNS = {"protein_a", "protein_b", "label"}
EXPECTED_LABEL_VALUES = {0, 1}


def read_fasta(sequences_path: str | Path) -> dict[str, str]:
    """
    Read protein sequences from a FASTA file.
    """
    sequences = {}
    current_id = None
    chunks = []

    with Path(sequences_path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)

                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(chunks)

    return sequences


def normalize_labels(labels: pd.Series, context: str) -> pd.Series:
    """
    Return labels as ints after validating they are binary 0/1 values.
    """
    if labels.isna().any():
        raise ValueError(f"{context} label column contains missing values.")

    try:
        numeric_labels = pd.to_numeric(labels, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} label column must contain only 0/1 values.") from exc

    unexpected_values = sorted(
        value
        for value in pd.unique(numeric_labels)
        if value not in EXPECTED_LABEL_VALUES)
    if unexpected_values:
        raise ValueError(
            f"{context} label column must contain only 0/1 values. "
            f"Found: {unexpected_values}")

    return numeric_labels.astype(int)


def validate_binary_labeling(labels: pd.Series, context: str) -> None:
    """
    Fail fast unless a label series contains both binary classes.
    """
    # Check that both classes are present 
    observed_labels = set(labels.unique())
    missing_labels = sorted(EXPECTED_LABEL_VALUES - observed_labels)
    if missing_labels:
        raise ValueError(
            f"{context} must contain both labels 0 and 1. "
            f"Missing: {missing_labels}")


def prepare_input_data(
        protein_pairs: pd.DataFrame, sequences: dict[str, str]
    ) -> pd.DataFrame:
    """
    Validate inputs early and return a label-normalized pairs dataframe.
    """
    # Check for empty files
    if protein_pairs.empty:
        raise ValueError("pairs.csv must contain at least one row.")
    if not sequences:
        raise ValueError("FASTA must contain at least one sequence.")

    # Check for missing columns
    missing_cols = REQUIRED_PAIR_COLUMNS - set(protein_pairs.columns)
    if missing_cols:
        raise ValueError(f"pairs.csv missing columns: {missing_cols}")

    # Check for missing sequences and normalize labels
    prepared_pairs = protein_pairs.copy()
    prepared_pairs["label"] = normalize_labels(
        labels=prepared_pairs["label"], context="pairs.csv")
    validate_binary_labeling(
        labels=prepared_pairs["label"], context="pairs.csv")

    # Check for proteins without FASTA sequences
    proteins = set(protein_pairs["protein_a"]) |\
               set(protein_pairs["protein_b"])
    missing_sequences = sorted(proteins - set(sequences))
    if missing_sequences:
        examples = missing_sequences[:10]
        raise ValueError(
            f"{len(missing_sequences)} protein pairs are missing sequences"
            f"Examples: {examples}")

    return prepared_pairs


def validate_train_test_splits(
        train_df: pd.DataFrame,  test_df: pd.DataFrame) -> None:
    """
    Validate split labels before feature construction and training.
    """
    validate_binary_labeling(train_df["label"], context="train split")
    validate_binary_labeling(test_df["label"], context="test split")


def load_or_make_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a pre-defined train/test split or create one from the protein pairs.
    """
    # Load train/test split or ...
    if args.split_col and args.split_col in pairs.columns:
        split_values = pairs[args.split_col].astype(str).str.lower()
        train_df = pairs[split_values == "train"].copy()
        test_df = pairs[split_values == "test"].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("split_col must contain 'train' & 'test' rows.")
    # Create a new train/test split
    else:
        train_df, test_df = train_test_split(
            pairs,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=pairs["label"],
        )

    return train_df, test_df
