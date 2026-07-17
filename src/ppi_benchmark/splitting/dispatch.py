"""Resolve and dispatch PPI split strategies without changing their semantics."""

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from .preparation import validate_binary_labeling
from .protein_disjoint import (
    split_pairs as split_protein_disjoint_pairs,
    split_pairs_three_way as split_protein_disjoint_pairs_three_way,
)
from .protocols import (
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


EXPECTED_SPLIT_VALUES = {"train", "val", "test"}
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
LOGGER = logging.getLogger(__name__)


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
        if strategy_name in {C2_SPLIT_STRATEGY, C3_SPLIT_STRATEGY}:
            grouping_kind = (
                "sequence_cluster"
                if protein_to_group is not None
                else resolved_spec.default_grouping_kind
            )
        else:
            grouping_kind = None
        args.split_audit["grouping_kind"] = grouping_kind
        args.split_audit["grouping_type"] = (
            "protein_id"
            if grouping_kind == "protein_identity"
            else grouping_kind
        )
        args.n_discarded_edges = len(split_result.dropped)
        args.discarded_edge_fraction = args.n_discarded_edges / len(pairs)

    # Create a random split
    else:
        train_df, val_df, test_df = make_random_pair_split(pairs, args)

    log_split_summary(train_df, val_df, test_df, args)
    return train_df, val_df, test_df
