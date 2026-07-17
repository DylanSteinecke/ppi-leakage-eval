"""Normalization primitives for canonical PPI pair tables."""

import pandas as pd


REQUIRED_PAIR_COLUMNS = {"protein_a", "protein_b", "label"}


def unordered_pair_key(protein_a: str, protein_b: str) -> tuple[str, str]:
    """Return the canonical unordered key for one protein pair."""
    if protein_a <= protein_b:
        return protein_a, protein_b
    return protein_b, protein_a


def validate_pair_columns(pairs: pd.DataFrame) -> None:
    """Raise if required canonical pair columns are absent."""
    missing_columns = sorted(REQUIRED_PAIR_COLUMNS - set(pairs.columns))
    if missing_columns:
        raise ValueError(
            "Pair table is missing required columns: "
            f"{missing_columns}"
        )


def normalize_labels(
    labels: pd.Series,
    context: str = "Pair table",
) -> pd.Series:
    """Return labels as integer zero/one values."""
    if labels.isna().any():
        raise ValueError(f"{context} label column contains missing values.")

    try:
        numeric_labels = pd.to_numeric(labels, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} label column must contain only 0/1 values."
        ) from exc

    unexpected_values = sorted(
        value for value in pd.unique(numeric_labels) if value not in {0, 1}
    )
    if unexpected_values:
        raise ValueError(
            f"{context} label column must contain only 0/1 values. "
            f"Found: {unexpected_values}"
        )
    return numeric_labels.astype(int)


def prepare_pair_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return normalized protein and label columns."""
    validate_pair_columns(pairs)
    work = pairs[["protein_a", "protein_b", "label"]].copy()
    missing_id_mask = work["protein_a"].isna() | work["protein_b"].isna()
    work["protein_a"] = work["protein_a"].astype(str).str.strip()
    work["protein_b"] = work["protein_b"].astype(str).str.strip()
    missing_id_mask = (
        missing_id_mask
        | work["protein_a"].eq("")
        | work["protein_b"].eq("")
    )
    if missing_id_mask.any():
        raise ValueError(
            f"Pair table contains {int(missing_id_mask.sum())} rows with "
            "missing protein IDs."
        )

    work["label"] = normalize_labels(work["label"])
    return work
