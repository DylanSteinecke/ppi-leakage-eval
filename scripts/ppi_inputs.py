"""
Input loading, validation, and split helpers for PPI pipelines.

This module owns the boundary between raw pipeline inputs and trusted in-memory
tables. Future additions should include new input formats, stricter QC checks,
leakage checks, and alternative train/test split strategies.
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_PAIR_COLUMNS = {"protein_a", "protein_b", "label"}
EXPECTED_LABEL_VALUES = {0, 1}
EXPECTED_SPLIT_VALUES = {"train", "test"}
PROVIDED_SPLIT_STRATEGY = "provided_column"
RANDOM_SPLIT_STRATEGY = "random"
PROTEIN_COMPONENT_SPLIT_STRATEGY = "protein_disjoint_components"
SPLIT_STRATEGY_CHOICES = (
    RANDOM_SPLIT_STRATEGY,
    PROTEIN_COMPONENT_SPLIT_STRATEGY,
)
LOGGER = logging.getLogger(__name__)


#################
# FASTA parsing #
#################
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


####################
# Data-checking QC #
####################
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

    normalized_labels = numeric_labels.astype(int)

    return normalized_labels


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


def validate_pair_ids(protein_pairs: pd.DataFrame) -> None:
    """
    Fail fast if pair rows are missing protein identifiers.
    """
    missing_id_mask = protein_pairs[["protein_a", "protein_b"]].isna().any(
        axis=1)
    if missing_id_mask.any():
        n_missing_rows = int(missing_id_mask.sum())
        raise ValueError(
            f"pairs.csv has {n_missing_rows} rows with missing protein IDs.")


def drop_pairs_missing_sequences(
        protein_pairs: pd.DataFrame, sequences: dict[str, str],
    ) -> pd.DataFrame:
    """
    Drop rows where either protein is absent from the FASTA sequences.
    """
    sequence_ids = pd.Index(sequences.keys())
    has_protein_a = protein_pairs["protein_a"].isin(sequence_ids)
    has_protein_b = protein_pairs["protein_b"].isin(sequence_ids)
    keep_mask = has_protein_a & has_protein_b
    n_dropped = int((~keep_mask).sum())

    if n_dropped == 0:
        filtered_pairs = protein_pairs
    else:
        missing_a = protein_pairs.loc[
            ~has_protein_a, "protein_a"].dropna().unique()
        missing_b = protein_pairs.loc[
            ~has_protein_b, "protein_b"].dropna().unique()
        missing_proteins = pd.Index(missing_a).union(
            pd.Index(missing_b),
            sort=False,
        )
        missing_examples = missing_proteins[:10].tolist()
        warning_message = (
            f"Dropping {n_dropped} of {len(protein_pairs)} pairs because one "
            f"or both proteins are missing FASTA sequences. Missing protein "
            f"count: {len(missing_proteins)}. Examples: {missing_examples}"
        )
        LOGGER.warning(warning_message)
        filtered_pairs = protein_pairs.loc[keep_mask].copy()

    if filtered_pairs.empty:
        raise ValueError(
            "All pairs were dropped because one or both proteins are missing "
            "FASTA sequences.")

    return filtered_pairs


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

    # Check protein IDs, normalize labels, and drop missing-sequence pairs
    prepared_pairs = protein_pairs.copy()
    validate_pair_ids(prepared_pairs)
    prepared_pairs["label"] = normalize_labels(
        labels=prepared_pairs["label"], context="pairs.csv")
    prepared_pairs = drop_pairs_missing_sequences(prepared_pairs, sequences)
    prepared_pairs = prepared_pairs.reset_index(drop=True)

    # Check that the final usable data still has both labels
    validate_binary_labeling(
        labels=prepared_pairs["label"], context="pairs.csv")
    return prepared_pairs


####################
# Train/test split #
####################
class DisjointSet:
    """
    Union-find data structure for scalable connected components.
    """

    def __init__(self) -> None:
        self.parent = {}
        self.size = {}

    def add(self, item: str) -> None:
        """
        Add an item if it is not already present.
        """
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        """
        Return the representative item for a connected component.
        """
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent

        return root

    def union(self, left_item: str, right_item: str) -> None:
        """
        Merge two connected components.
        """
        left_root = self.find(left_item)
        right_root = self.find(right_item)
        if left_root != right_root:
            if self.size[left_root] < self.size[right_root]:
                left_root, right_root = right_root, left_root

            self.parent[right_root] = left_root
            self.size[left_root] = self.size[left_root] + self.size[right_root]


def protein_ids_in_pairs(pairs: pd.DataFrame) -> set[str]:
    """
    Return all protein IDs represented in a pair dataframe.
    """
    protein_ids = set(pairs["protein_a"]) | set(pairs["protein_b"])

    return protein_ids


def validate_disjoint_proteins(
        train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Fail fast if any protein appears in both train and test pairs.
    """
    overlap = protein_ids_in_pairs(train_df) & protein_ids_in_pairs(test_df)
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"Train/test split has {len(overlap)} shared proteins. "
            f"Examples: {examples}")


def validate_train_test_splits(
        train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Validate split labels before feature construction and training.
    """
    validate_binary_labeling(train_df["label"], context="train split")
    validate_binary_labeling(test_df["label"], context="test split")


def make_random_pair_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make a stratified random pair split.
    """
    train_df, test_df = train_test_split(
        pairs,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=pairs["label"],
    )

    return train_df, test_df


def load_split_column(
        pairs: pd.DataFrame, split_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test labels from a provided split column.
    """
    if split_col not in pairs.columns:
        raise ValueError(f"split_col '{split_col}' is not in pairs.csv.")

    split_values = pairs[split_col].astype(str).str.lower()
    unexpected_values = sorted(set(split_values) - EXPECTED_SPLIT_VALUES)
    if unexpected_values:
        raise ValueError(
            f"split_col must contain only 'train' and 'test'. "
            f"Found: {unexpected_values}")

    train_df = pairs[split_values == "train"].copy()
    test_df = pairs[split_values == "test"].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("split_col must contain 'train' & 'test' rows.")

    return train_df, test_df


def get_pair_component_ids(pairs: pd.DataFrame) -> pd.Series:
    """
    Return connected component IDs for each protein-pair row.
    """
    disjoint_set = DisjointSet()
    for protein_a, protein_b in zip(pairs["protein_a"], pairs["protein_b"]):
        disjoint_set.union(protein_a, protein_b)

    component_ids = pd.Series(
        [disjoint_set.find(protein_a) for protein_a in pairs["protein_a"]],
        index=pairs.index,
    )

    return component_ids


def choose_train_components(
        component_sizes: pd.Series, train_size: float, seed: int,
    ) -> set[str]:
    """
    Choose connected components with train rows close to the target size.
    """
    target_n_train = component_sizes.sum() * train_size
    component_items = list(component_sizes.items())
    rng = random.Random(seed)
    rng.shuffle(component_items)
    component_items = sorted(
        component_items,
        key=lambda component_item: component_item[1],
        reverse=True,
    )

    train_components = set()
    n_train = 0
    for component_id, component_n_rows in component_items:
        current_distance = abs(n_train - target_n_train)
        next_distance = abs(n_train + component_n_rows - target_n_train)
        if next_distance <= current_distance:
            train_components.add(component_id)
            n_train = n_train + component_n_rows

    return train_components


def make_protein_component_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make a protein-disjoint split by assigning connected components.
    """
    component_ids = get_pair_component_ids(pairs)
    component_sizes = component_ids.value_counts(sort=False)
    n_components = len(component_sizes)
    args.n_connected_components = n_components
    if n_components < 2:
        raise ValueError(
            "protein_disjoint_components split requires at least two "
            "connected components.")

    train_components = choose_train_components(
        component_sizes=component_sizes,
        train_size=args.train_size,
        seed=args.seed,
    )
    train_mask = component_ids.isin(train_components)
    train_df = pairs.loc[train_mask].copy()
    test_df = pairs.loc[~train_mask].copy()

    return train_df, test_df


def log_split_summary(
        train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace,
    ) -> None:
    """
    Log target and realized train/test split sizes.
    """
    n_total = len(train_df) + len(test_df)
    actual_train_size = len(train_df) / n_total
    actual_test_size = len(test_df) / n_total
    message = (
        f"Split strategy={args.effective_split_strategy}; "
        f"target train fraction={args.train_size:.3f}; "
        f"actual train fraction={actual_train_size:.3f}; "
        f"actual test fraction={actual_test_size:.3f}; "
        f"train pairs={len(train_df)}; test pairs={len(test_df)}"
    )
    if args.n_connected_components is not None:
        message = (
            f"{message}; connected components={args.n_connected_components}"
        )
    LOGGER.info(message)


def load_or_make_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a pre-defined train/test split or create one from the protein pairs.
    """
    # Load train/test split
    if args.split_col:
        train_df, test_df = load_split_column(pairs, args.split_col)

    # Create a protein-disjoint train/test split
    elif args.effective_split_strategy == PROTEIN_COMPONENT_SPLIT_STRATEGY:
        train_df, test_df = make_protein_component_split(pairs, args)
        validate_disjoint_proteins(train_df, test_df)

    # Create a random train/test split
    else:
        train_df, test_df = make_random_pair_split(pairs, args)

    log_split_summary(train_df, test_df, args)
    return train_df, test_df
