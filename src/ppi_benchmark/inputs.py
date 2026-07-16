"""
Input loading, validation, and split helpers for PPI pipelines.

This module owns the boundary between raw pipeline inputs and trusted in-memory
tables. Future additions should include new input formats, stricter QC checks,
leakage checks, and alternative train/test split strategies.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .datasets.common import normalize_labels
from .split_protocols import (
    C1_SPLIT_STRATEGY as C1_SPLIT_STRATEGY,
    C2_SPLIT_STRATEGY,
    C3_SPLIT_STRATEGY,
    PROTEIN_DISJOINT_SPLIT_STRATEGIES,
    PROVIDED_SPLIT_STRATEGY as PROVIDED_SPLIT_STRATEGY,
    RANDOM_SPLIT_STRATEGY as RANDOM_SPLIT_STRATEGY,
    SPLIT_STRATEGY_CHOICES as SPLIT_STRATEGY_CHOICES,
    SplitStrategySpec,
    get_split_strategy,
)
from .splitters import (
    split_pairs as split_protein_disjoint_pairs,
    split_pairs_three_way as split_protein_disjoint_pairs_three_way,
)


REQUIRED_PAIR_COLUMNS = {"protein_a", "protein_b", "label"}
EXPECTED_LABEL_VALUES = {0, 1}
EXPECTED_SPLIT_VALUES = {"train", "val", "test"}
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
LOGGER = logging.getLogger(__name__)
SEQUENCE_CLUSTER_PROTEIN_COLUMN = "protein_id"
SEQUENCE_CLUSTER_COLUMN = "cluster_id"

####################
# Data-checking QC #
####################
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
            f"pairs.csv has {n_missing_rows} rows with missing protein IDs.")


def drop_pairs_missing_sequences(
        protein_pairs: pd.DataFrame, sequences: dict[str, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where either protein is absent from the FASTA sequences.
    """
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
            "FASTA sequences.")

    return filtered_pairs, dropped_pairs


def prepare_input_data(
        protein_pairs: pd.DataFrame, sequences: dict[str, str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate inputs early and return label-normalized pairs and dropped rows.
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
    prepared_pairs["protein_a"] = (
        prepared_pairs["protein_a"].astype("string").str.strip())
    prepared_pairs["protein_b"] = (
        prepared_pairs["protein_b"].astype("string").str.strip())
    prepared_pairs["label"] = normalize_labels(
        labels=prepared_pairs["label"], context="pairs.csv")
    prepared_pairs, dropped_pairs = drop_pairs_missing_sequences(
        prepared_pairs,
        sequences,
    )
    prepared_pairs = prepared_pairs.reset_index(drop=True)

    # Check that the final usable data still has both labels
    validate_binary_labeling(
        labels=prepared_pairs["label"], context="pairs.csv")
    return prepared_pairs, dropped_pairs


def load_sequence_cluster_mapping(
        mapping_path: str | Path,
        required_proteins: set[str] | None = None,
    ) -> tuple[dict[str, str], pd.DataFrame, dict[str, int]]:
    """Load a canonical protein-to-sequence-cluster mapping CSV."""
    mapping_path = Path(mapping_path)
    mapping_df = pd.read_csv(
        mapping_path,
        dtype={
            SEQUENCE_CLUSTER_PROTEIN_COLUMN: "string",
            SEQUENCE_CLUSTER_COLUMN: "string",
        },
    )
    required_columns = {
        SEQUENCE_CLUSTER_PROTEIN_COLUMN,
        SEQUENCE_CLUSTER_COLUMN,
    }
    missing_columns = required_columns - set(mapping_df.columns)
    if missing_columns:
        raise ValueError(
            f"Sequence-cluster CSV is missing columns: {missing_columns}")
    mapping_df = mapping_df[[
        SEQUENCE_CLUSTER_PROTEIN_COLUMN,
        SEQUENCE_CLUSTER_COLUMN,
    ]].copy()
    for column in required_columns:
        mapping_df[column] = mapping_df[column].astype("string").str.strip()
        missing_mask = mapping_df[column].isna() | mapping_df[column].eq("")
        if missing_mask.any():
            raise ValueError(
                f"Sequence-cluster CSV has {int(missing_mask.sum())} "
                f"missing {column} value(s)."
            )
    duplicate_mask = mapping_df[
        SEQUENCE_CLUSTER_PROTEIN_COLUMN
    ].duplicated(keep=False)
    if duplicate_mask.any():
        examples = mapping_df.loc[
            duplicate_mask,
            SEQUENCE_CLUSTER_PROTEIN_COLUMN,
        ].drop_duplicates().head(10).tolist()
        raise ValueError(
            "Sequence-cluster CSV must contain one row per protein. "
            f"Duplicate examples: {examples}"
        )

    protein_to_group = dict(zip(
        mapping_df[SEQUENCE_CLUSTER_PROTEIN_COLUMN],
        mapping_df[SEQUENCE_CLUSTER_COLUMN],
    ))
    if required_proteins is not None:
        missing_proteins = set(required_proteins) - set(protein_to_group)
        if missing_proteins:
            examples = sorted(missing_proteins)[:10]
            raise ValueError(
                "Sequence-cluster CSV must map every eligible protein for "
                "C2/C3 splitting. Missing "
                f"{len(missing_proteins)} protein(s): {examples}"
            )
    mapping_df = mapping_df.sort_values(
        SEQUENCE_CLUSTER_PROTEIN_COLUMN,
    ).reset_index(drop=True)
    metadata = {
        "n_mapped_proteins": len(mapping_df),
        "n_sequence_clusters": mapping_df[
            SEQUENCE_CLUSTER_COLUMN
        ].nunique(),
    }
    return protein_to_group, mapping_df, metadata


####################
# Train/test split #
####################
def protein_ids_in_pairs(pairs: pd.DataFrame) -> set[str]:
    """
    Return all protein IDs represented in a pair dataframe.
    """
    protein_ids = set(pairs["protein_a"]) | set(pairs["protein_b"])

    return protein_ids


def validate_splits(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ) -> None:
    """
    Validate non-empty split labels before feature construction and training.
    """
    split_dfs = {
        TRAIN_SPLIT: train_df,
        VAL_SPLIT: val_df,
        TEST_SPLIT: test_df,
    }
    for split_name, split_df in split_dfs.items():
        if split_df is None:
            continue
        if split_df.empty:
            raise ValueError(f"{split_name} split must not be empty.")
        validate_binary_labeling(
            split_df["label"],
            context=f"{split_name} split",
        )


def make_random_pair_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Make a stratified random pair split.
    """
    split_seed = getattr(args, "split_seed", args.seed)
    if args.val_size == 0.0:
        train_df, test_df = train_test_split(
            pairs,
            train_size=args.train_size,
            random_state=split_seed,
            stratify=pairs["label"],
        )
        val_df = None
    else:
        train_df, heldout_df = train_test_split(
            pairs,
            train_size=args.train_size,
            random_state=split_seed,
            stratify=pairs["label"],
        )
        relative_val_size = args.val_size / (1.0 - args.train_size)
        val_df, test_df = train_test_split(
            heldout_df,
            train_size=relative_val_size,
            random_state=split_seed + 1,
            stratify=heldout_df["label"],
        )

    return train_df, val_df, test_df


def load_split_column(
        pairs: pd.DataFrame, split_col: str, val_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Load train/validation/test labels from a provided split column.
    """
    split_values = normalized_split_values(pairs, split_col)

    train_df = pairs[split_values == TRAIN_SPLIT].copy()
    val_df = pairs[split_values == VAL_SPLIT].copy()
    test_df = pairs[split_values == TEST_SPLIT].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("split_col must contain 'train' & 'test' rows.")
    if val_size > 0.0 and len(val_df) == 0:
        raise ValueError(
            "split_col must contain 'val' rows when --val-size is greater "
            "than 0.")
    if val_df.empty:
        val_df = None

    return train_df, val_df, test_df


def normalized_split_values(
        pairs: pd.DataFrame, split_col: str,
    ) -> pd.Series:
    """
    Validate and normalize a provided split column.

    This helper can run before cohort sampling so invalid values cannot be
    hidden merely because their rows were not selected.
    """
    if split_col not in pairs.columns:
        raise ValueError(f"split_col '{split_col}' is not in pairs.csv.")

    split_values = pairs[split_col].astype(str).str.strip().str.lower()
    unexpected_values = sorted(set(split_values) - EXPECTED_SPLIT_VALUES)
    if unexpected_values:
        raise ValueError(
            f"split_col must contain only 'train', 'val', and 'test'. "
            f"Found: {unexpected_values}")
    return split_values


def log_split_summary(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, args: argparse.Namespace,
    ) -> None:
    """
    Log target and realized split sizes.
    """
    n_val = 0 if val_df is None else len(val_df)
    n_total = len(train_df) + n_val + len(test_df)
    actual_train_size = len(train_df) / n_total
    actual_val_size = n_val / n_total
    actual_test_size = len(test_df) / n_total
    message = (
        f"Split strategy={args.effective_split_strategy}; "
        f"target train fraction={args.train_size:.3f}; "
        f"target val fraction={args.val_size:.3f}; "
        f"actual train fraction={actual_train_size:.3f}; "
        f"actual val fraction={actual_val_size:.3f}; "
        f"actual test fraction={actual_test_size:.3f}; "
        f"train pairs={len(train_df)}; val pairs={n_val}; "
        f"test pairs={len(test_df)}"
    )
    if args.effective_split_strategy in PROTEIN_DISJOINT_SPLIT_STRATEGIES:
        message = (
            f"{message}; discarded edges={args.n_discarded_edges}; "
            f"discarded fraction={args.discarded_edge_fraction:.3f}"
        )
    LOGGER.info(message)


def load_or_make_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
        protein_to_group: dict[str, str] | None = None,
        *, split_spec: SplitStrategySpec | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Load a pre-defined split or create one from the protein pairs.
    """
    resolved_spec = (
        get_split_strategy("ppi", args.effective_split_strategy)
        if split_spec is None else split_spec
    )
    if resolved_spec.task_id != "ppi":
        raise ValueError(
            "The PPI split implementation cannot execute task "
            f"{resolved_spec.task_id!r}."
        )
    if resolved_spec.strategy_name != args.effective_split_strategy:
        raise ValueError(
            "Resolved split strategy does not match the requested legacy "
            f"strategy: {resolved_spec.strategy_name!r} != "
            f"{args.effective_split_strategy!r}."
        )

    strategy_name = resolved_spec.strategy_name
    args.n_discarded_edges = 0
    args.discarded_edge_fraction = 0.0
    args.split_audit = None
    split_seed = getattr(args, "split_seed", args.seed)

    # Load provided split labels
    if args.split_col:
        train_df, val_df, test_df = load_split_column(
            pairs=pairs,
            split_col=args.split_col,
            val_size=args.val_size,
        )

    # Create a C1/C2/C3 protein-disjoint split
    elif strategy_name in PROTEIN_DISJOINT_SPLIT_STRATEGIES:
        test_size = 1.0 - args.train_size - args.val_size
        if args.val_size > 0.0:
            split_result = split_protein_disjoint_pairs_three_way(
                pairs,
                mode=strategy_name,
                val_size=args.val_size,
                test_size=test_size,
                seed=split_seed,
                n_trials=args.n_split_trials,
                protein_to_group=protein_to_group,
            )
            val_df = split_result.val
        else:
            split_result = split_protein_disjoint_pairs(
                pairs,
                mode=strategy_name,
                test_size=test_size,
                seed=split_seed,
                n_trials=args.n_split_trials,
                protein_to_group=protein_to_group,
            )
            val_df = None
        train_df = split_result.train
        test_df = split_result.test
        args.split_audit = split_result.audit
        args.split_audit["grouping_type"] = (
            "sequence_cluster"
            if protein_to_group is not None
            and strategy_name in {
                C2_SPLIT_STRATEGY,
                C3_SPLIT_STRATEGY,
            }
            else "protein_id"
        )
        args.n_discarded_edges = len(split_result.dropped)
        args.discarded_edge_fraction = args.n_discarded_edges / len(pairs)

    # Create a random split
    else:
        train_df, val_df, test_df = make_random_pair_split(pairs, args)

    log_split_summary(train_df, val_df, test_df, args)
    return train_df, val_df, test_df
