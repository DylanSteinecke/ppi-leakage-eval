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
EXPECTED_SPLIT_VALUES = {"train", "val", "test"}
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
PROVIDED_SPLIT_STRATEGY = "provided_column"
RANDOM_SPLIT_STRATEGY = "random"
PROTEIN_COMPONENT_SPLIT_STRATEGY = "protein_disjoint_components"
PROTEIN_PRUNE_SPLIT_STRATEGY = "protein_disjoint_prune_edges"
SPLIT_STRATEGY_CHOICES = (
    RANDOM_SPLIT_STRATEGY,
    PROTEIN_COMPONENT_SPLIT_STRATEGY,
    PROTEIN_PRUNE_SPLIT_STRATEGY,
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

    def save_current_record() -> None:
        """
        Save the current FASTA record after checking it has sequence text.
        """
        if current_id is None:
            return
        if not chunks:
            raise ValueError(
                f"FASTA record '{current_id}' has no sequence.")

        sequences[current_id] = "".join(chunks)

    with Path(sequences_path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                save_current_record()

                header_parts = line[1:].split()
                if not header_parts:
                    raise ValueError(
                        "FASTA record header is missing a sequence ID.")

                current_id = header_parts[0]
                if current_id in sequences:
                    raise ValueError(
                        f"Duplicate FASTA sequence ID: {current_id}")
                chunks = []
            else:
                if current_id is None:
                    raise ValueError(
                        "FASTA sequence line found before any header.")
                chunks.append(line)

    save_current_record()

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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where either protein is absent from the FASTA sequences.
    """
    sequence_ids = pd.Index(sequences.keys())
    has_protein_a = protein_pairs["protein_a"].isin(sequence_ids)
    has_protein_b = protein_pairs["protein_b"].isin(sequence_ids)
    keep_mask = has_protein_a & has_protein_b
    n_dropped = int((~keep_mask).sum())
    dropped_columns = ["source_row_index", "drop_reason"]
    if "pair_id" in protein_pairs.columns:
        dropped_columns.append("pair_id")

    if n_dropped == 0:
        filtered_pairs = protein_pairs
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


def validate_disjoint_splits(
        split_dfs: dict[str, pd.DataFrame | None],
    ) -> None:
    """
    Fail fast if any protein appears in more than one non-empty split.
    """
    protein_sets = {
        split_name: protein_ids_in_pairs(split_df)
        for split_name, split_df in split_dfs.items()
        if split_df is not None and not split_df.empty
    }
    split_names = list(protein_sets)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1:]:
            overlap = protein_sets[left_name] & protein_sets[right_name]
            if overlap:
                examples = sorted(overlap)[:10]
                raise ValueError(
                    f"{left_name}/{right_name} split has {len(overlap)} "
                    f"shared proteins. Examples: {examples}")


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
        if split_df is not None and not split_df.empty:
            validate_binary_labeling(
                split_df["label"],
                context=f"{split_name} split",
            )


def validate_train_test_splits(
        train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Backward-compatible train/test split validator.
    """
    validate_splits(train_df=train_df, val_df=None, test_df=test_df)


def make_random_pair_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Make a stratified random pair split.
    """
    if args.val_size == 0.0:
        train_df, test_df = train_test_split(
            pairs,
            train_size=args.train_size,
            random_state=args.seed,
            stratify=pairs["label"],
        )
        val_df = None
    else:
        train_df, heldout_df = train_test_split(
            pairs,
            train_size=args.train_size,
            random_state=args.seed,
            stratify=pairs["label"],
        )
        relative_val_size = args.val_size / (1.0 - args.train_size)
        val_df, test_df = train_test_split(
            heldout_df,
            train_size=relative_val_size,
            random_state=args.seed + 1,
            stratify=heldout_df["label"],
        )

    return train_df, val_df, test_df


def load_split_column(
        pairs: pd.DataFrame, split_col: str, val_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Load train/validation/test labels from a provided split column.
    """
    if split_col not in pairs.columns:
        raise ValueError(f"split_col '{split_col}' is not in pairs.csv.")

    split_values = pairs[split_col].astype(str).str.lower()
    unexpected_values = sorted(set(split_values) - EXPECTED_SPLIT_VALUES)
    if unexpected_values:
        raise ValueError(
            f"split_col must contain only 'train', 'val', and 'test'. "
            f"Found: {unexpected_values}")

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


def choose_components_by_targets(
        component_sizes: pd.Series, split_targets: dict[str, float],
        seed: int,
    ) -> dict[str, set[str]]:
    """
    Assign connected components to splits near requested row fractions.
    """
    total_rows = component_sizes.sum()
    target_counts = {
        split_name: total_rows * split_fraction
        for split_name, split_fraction in split_targets.items()
    }
    component_items = list(component_sizes.items())
    rng = random.Random(seed)
    rng.shuffle(component_items)
    component_items = sorted(
        component_items,
        key=lambda component_item: component_item[1],
        reverse=True,
    )

    split_components = {
        split_name: set()
        for split_name in split_targets
    }
    split_counts = {
        split_name: 0
        for split_name in split_targets
    }
    for component_id, component_n_rows in component_items:
        best_split = min(
            split_targets,
            key=lambda split_name: (
                abs(
                    split_counts[split_name]
                    + component_n_rows
                    - target_counts[split_name]
                )
                - abs(split_counts[split_name] - target_counts[split_name]),
                split_counts[split_name] / max(target_counts[split_name], 1.0),
            ),
        )
        split_components[best_split].add(component_id)
        split_counts[best_split] += component_n_rows

    return split_components


def make_protein_component_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Make a protein-disjoint split by assigning connected components.
    """
    component_ids = get_pair_component_ids(pairs)
    component_sizes = component_ids.value_counts(sort=False)
    n_components = len(component_sizes)
    args.n_connected_components = n_components
    n_required_splits = 3 if args.val_size > 0.0 else 2
    if n_components < n_required_splits:
        n_pairs = len(pairs)
        n_proteins = len(protein_ids_in_pairs(pairs))
        largest_component_size = int(component_sizes.max())
        raise ValueError(
            "protein_disjoint_components split requires at least "
            f"{n_required_splits} connected components for the requested "
            f"splits. Found {n_components} connected component(s) across "
            f"{n_pairs} pairs and {n_proteins} proteins. Largest component "
            f"has {largest_component_size} pairs.")

    if args.val_size == 0.0:
        train_components = choose_train_components(
            component_sizes=component_sizes,
            train_size=args.train_size,
            seed=args.seed,
        )
        train_mask = component_ids.isin(train_components)
        train_df = pairs.loc[train_mask].copy()
        val_df = None
        test_df = pairs.loc[~train_mask].copy()
    else:
        split_targets = {
            TRAIN_SPLIT: args.train_size,
            VAL_SPLIT: args.val_size,
            TEST_SPLIT: 1.0 - args.train_size - args.val_size,
        }
        split_components = choose_components_by_targets(
            component_sizes=component_sizes,
            split_targets=split_targets,
            seed=args.seed,
        )
        train_df = pairs.loc[
            component_ids.isin(split_components[TRAIN_SPLIT])].copy()
        val_df = pairs.loc[
            component_ids.isin(split_components[VAL_SPLIT])].copy()
        test_df = pairs.loc[
            component_ids.isin(split_components[TEST_SPLIT])].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "protein_disjoint_components split produced an empty train, "
                "val, or test split. Try a different --seed, a larger "
                "dataset, or a less extreme --train-size/--val-size.")

    return train_df, val_df, test_df


############################
# Protein-disjoint pruning #
############################
def estimate_train_protein_fraction(train_size: float) -> float:
    """
    Estimate the protein split needed for the target retained pair split.

    Under random protein assignment, within-train pairs scale with p^2 and
    within-test pairs scale with (1 - p)^2. Solving that approximation keeps
    the retained train/test pair ratio closer to the requested pair ratio after
    crossing pairs are pruned.
    """
    train_weight = train_size ** 0.5
    test_weight = (1.0 - train_size) ** 0.5
    train_protein_fraction = train_weight / (train_weight + test_weight)

    return train_protein_fraction


def estimate_split_protein_fractions(
        split_targets: dict[str, float],
    ) -> dict[str, float]:
    """
    Estimate protein fractions for target retained pair fractions.
    """
    split_weights = {
        split_name: split_fraction ** 0.5
        for split_name, split_fraction in split_targets.items()
    }
    total_weight = sum(split_weights.values())
    protein_fractions = {
        split_name: split_weight / total_weight
        for split_name, split_weight in split_weights.items()
    }

    return protein_fractions


def choose_train_proteins(
        pairs: pd.DataFrame, train_size: float, seed: int,
    ) -> set[str]:
    """
    Choose proteins assigned to the training side of a disjoint split.
    """
    proteins = sorted(protein_ids_in_pairs(pairs))
    if len(proteins) < 2:
        raise ValueError(
            "protein_disjoint_prune_edges split requires at least two "
            "proteins.")

    train_protein_fraction = estimate_train_protein_fraction(train_size)
    n_train_proteins = round(len(proteins) * train_protein_fraction)
    n_train_proteins = max(1, min(n_train_proteins, len(proteins) - 1))

    rng = random.Random(seed)
    rng.shuffle(proteins)
    train_proteins = set(proteins[:n_train_proteins])

    return train_proteins


def choose_proteins_by_targets(
        pairs: pd.DataFrame, split_targets: dict[str, float], seed: int,
    ) -> dict[str, set[str]]:
    """
    Assign proteins to train/validation/test groups.
    """
    proteins = sorted(protein_ids_in_pairs(pairs))
    if len(proteins) < len(split_targets):
        raise ValueError(
            "protein_disjoint_prune_edges split requires at least one "
            "protein per requested split.")

    protein_fractions = estimate_split_protein_fractions(split_targets)
    split_names = list(split_targets)
    raw_counts = {
        split_name: len(proteins) * protein_fractions[split_name]
        for split_name in split_names
    }
    split_counts = {
        split_name: max(1, int(raw_counts[split_name]))
        for split_name in split_names
    }
    remaining_count = len(proteins) - sum(split_counts.values())
    fractional_order = sorted(
        split_names,
        key=lambda split_name: raw_counts[split_name] % 1.0,
        reverse=True,
    )
    order_index = 0
    while remaining_count > 0:
        split_counts[fractional_order[order_index % len(fractional_order)]] += 1
        remaining_count -= 1
        order_index += 1
    while remaining_count < 0:
        reducible_splits = [
            split_name
            for split_name in reversed(fractional_order)
            if split_counts[split_name] > 1
        ]
        if not reducible_splits:
            break
        split_counts[reducible_splits[0]] -= 1
        remaining_count += 1

    rng = random.Random(seed)
    rng.shuffle(proteins)
    split_proteins = {}
    start_index = 0
    for split_name in split_names:
        stop_index = start_index + split_counts[split_name]
        split_proteins[split_name] = set(proteins[start_index:stop_index])
        start_index = stop_index

    return split_proteins


def make_protein_prune_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Make a protein-disjoint split by pruning train/test crossing pairs.
    """
    if args.val_size == 0.0:
        train_proteins = choose_train_proteins(
            pairs=pairs,
            train_size=args.train_size,
            seed=args.seed,
        )
        protein_a_is_train = pairs["protein_a"].isin(train_proteins)
        protein_b_is_train = pairs["protein_b"].isin(train_proteins)

        train_mask = protein_a_is_train & protein_b_is_train
        val_mask = pd.Series(False, index=pairs.index)
        test_mask = (~protein_a_is_train) & (~protein_b_is_train)
    else:
        split_targets = {
            TRAIN_SPLIT: args.train_size,
            VAL_SPLIT: args.val_size,
            TEST_SPLIT: 1.0 - args.train_size - args.val_size,
        }
        split_proteins = choose_proteins_by_targets(
            pairs=pairs,
            split_targets=split_targets,
            seed=args.seed,
        )
        protein_a_split = {}
        protein_b_split = {}
        for split_name, split_protein_ids in split_proteins.items():
            protein_a_split[split_name] = pairs["protein_a"].isin(
                split_protein_ids)
            protein_b_split[split_name] = pairs["protein_b"].isin(
                split_protein_ids)

        train_mask = (
            protein_a_split[TRAIN_SPLIT] & protein_b_split[TRAIN_SPLIT])
        val_mask = protein_a_split[VAL_SPLIT] & protein_b_split[VAL_SPLIT]
        test_mask = protein_a_split[TEST_SPLIT] & protein_b_split[TEST_SPLIT]

    pruned_mask = ~(train_mask | val_mask | test_mask)
    args.n_pruned_pairs = int(pruned_mask.sum())
    args.pruned_pair_fraction = args.n_pruned_pairs / len(pairs)

    train_df = pairs.loc[train_mask].copy()
    val_df = pairs.loc[val_mask].copy() if args.val_size > 0.0 else None
    test_df = pairs.loc[test_mask].copy()
    if args.val_size > 0.0:
        has_empty_split = train_df.empty or test_df.empty or (
            val_df is None or val_df.empty)
        empty_message = "train, val, or test"
    else:
        has_empty_split = train_df.empty or test_df.empty
        empty_message = "train or test"
    if has_empty_split:
        raise ValueError(
            "protein_disjoint_prune_edges split produced an empty "
            f"{empty_message} split after pruning crossing pairs. Try a "
            "different --seed, a larger dataset, or a less extreme "
            "--train-size/--val-size.")

    return train_df, val_df, test_df


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
    if args.n_connected_components is not None:
        message = (
            f"{message}; connected components={args.n_connected_components}"
        )
    if args.effective_split_strategy == PROTEIN_PRUNE_SPLIT_STRATEGY:
        message = (
            f"{message}; pruned crossing pairs={args.n_pruned_pairs}; "
            f"pruned fraction={args.pruned_pair_fraction:.3f}"
        )
    LOGGER.info(message)


def load_or_make_split(
        pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Load a pre-defined split or create one from the protein pairs.
    """
    args.n_connected_components = None
    args.n_pruned_pairs = 0
    args.pruned_pair_fraction = 0.0

    # Load provided split labels
    if args.split_col:
        train_df, val_df, test_df = load_split_column(
            pairs=pairs,
            split_col=args.split_col,
            val_size=args.val_size,
        )

    # Create a protein-disjoint split
    elif args.effective_split_strategy == PROTEIN_COMPONENT_SPLIT_STRATEGY:
        train_df, val_df, test_df = make_protein_component_split(pairs, args)
        validate_disjoint_splits({
            TRAIN_SPLIT: train_df,
            VAL_SPLIT: val_df,
            TEST_SPLIT: test_df,
        })

    # Create a protein-disjoint split by pruning crossing pairs
    elif args.effective_split_strategy == PROTEIN_PRUNE_SPLIT_STRATEGY:
        train_df, val_df, test_df = make_protein_prune_split(pairs, args)
        validate_disjoint_splits({
            TRAIN_SPLIT: train_df,
            VAL_SPLIT: val_df,
            TEST_SPLIT: test_df,
        })

    # Create a random split
    else:
        train_df, val_df, test_df = make_random_pair_split(pairs, args)

    log_split_summary(train_df, val_df, test_df, args)
    return train_df, val_df, test_df
