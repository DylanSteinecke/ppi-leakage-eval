"""
Plotting helpers for PPI pipeline results.

This module owns publication-style visual summaries derived from machine-
readable result tables. Future additions should include configurable metric
sets, comparison plots across experiments, and optional PNG/PDF export.
"""

import math
from pathlib import Path

import pandas as pd


PLOT_METRICS = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
)

FONT_FAMILY = "Arial, Helvetica, sans-serif"
LABEL_WIDTH = 220
PLOT_WIDTH = 260
PANEL_GAP = 34
ROW_HEIGHT = 28
PANEL_HEADER = 42
AXIS_HEIGHT = 38
TOP_MARGIN = 74
RIGHT_MARGIN = 28
BOTTOM_MARGIN = 24
LEFT_MARGIN = 24
GRID_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_COLOR = "#2E7D32"
TRAIN_STROKE = "#155A24"
TEST_COLOR = "#E68619"
TEST_STROKE = "#9A4F00"
COMPARISON_LINE = "#A8A8A8"
ARROW_HEAD_LENGTH = 8.0
ARROW_HEAD_HALF_HEIGHT = 4.0
ARROW_MARKER_GAP = 6.0
FEATURE_ORDER = ("none", "tfidf", "bm25", "count", "binary")
CLASSIFIER_ORDER = (
    "logistic",
    "linear_svm",
    "sgd_logistic",
    "torch_mlp",
    "degree_logistic",
    "degree_hgb",
    "always_positive",
    "always_negative",
)
BASELINE_CLASSIFIERS = (
    "degree_logistic",
    "degree_hgb",
    "always_positive",
    "always_negative",
)
CLASSIFIER_LABELS = {
    "logistic": "Logistic",
    "linear_svm": "Linear SVM",
    "sgd_logistic": "SGD Logistic",
    "torch_mlp": "Torch MLP",
    "degree_logistic": "Degree Logistic",
    "degree_hgb": "Degree HGB (Sensitivity)",
    "always_positive": "Always Positive",
    "always_negative": "Always Negative",
}
SPLIT_STRATEGY_LABELS = {
    "random": "Random Split",
    "c1": "C1 Edge-Disjoint Split",
    "c2": "C2 One-Novel-Partner Split",
    "c3": "C3 Protein-Disjoint Split",
    "provided_column": "Provided Split",
}
SPLIT_STRATEGY_ORDER = ("random", "c1", "c2", "c3", "provided_column")


def legacy_f1_heatmap_output_paths(
        base_plot_path: Path, comparison_split_name: str = "test",
    ) -> dict[str, Path]:
    """
    Return old per-panel output paths so stale files can be cleared.
    """
    base_plot_path = Path(base_plot_path)
    output_dir = base_plot_path.parent
    stem = base_plot_path.with_suffix("").name
    prefix = f"train_{comparison_split_name}_f1_heatmap_"
    if stem.startswith(prefix):
        suffix = stem.removeprefix(prefix)
        path_names = {
            "train": f"train_f1_heatmap_{suffix}.png",
            comparison_split_name: (
                f"{comparison_split_name}_f1_heatmap_{suffix}.png"),
            "difference": (
                f"{comparison_split_name}_minus_train_f1_heatmap_"
                f"{suffix}.png"),
        }
    else:
        path_names = {
            "train": f"{stem}_train.png",
            comparison_split_name: f"{stem}_{comparison_split_name}.png",
            "difference": f"{stem}_{comparison_split_name}_minus_train.png",
        }

    output_paths = {
        variant: output_dir / path_name
        for variant, path_name in path_names.items()
    }

    return output_paths


def explicit_split_name(summary_df: pd.DataFrame) -> str | None:
    """
    Return a user-provided split name from a metrics summary table.
    """
    if "split_name" not in summary_df.columns:
        return None

    split_names = []
    for split_name in summary_df["split_name"].dropna().unique():
        clean_name = str(split_name).strip()
        if clean_name:
            split_names.append(clean_name)

    if split_names:
        label = " / ".join(split_names)
    else:
        label = None

    return label


def split_strategy_label(summary_df: pd.DataFrame) -> str | None:
    """
    Return a readable split-strategy label from a metrics summary table.
    """
    explicit_label = explicit_split_name(summary_df)
    if explicit_label:
        return explicit_label

    if "split_strategy" not in summary_df.columns:
        return None

    split_flags = [
        str(split_flag)
        for split_flag in summary_df["split_strategy"].dropna().unique()
    ]
    if not split_flags:
        return None

    split_labels = [
        SPLIT_STRATEGY_LABELS.get(
            split_flag,
            split_flag.replace("_", " ").title(),
        )
        for split_flag in sorted(split_flags)
    ]
    label = " / ".join(split_labels)

    return label


def title_with_split_strategy(
        base_title: str, summary_df: pd.DataFrame,
    ) -> str:
    """
    Prefix a plot title with the readable split strategy when available.
    """
    split_label = split_strategy_label(summary_df)
    if split_label:
        title = f"{split_label}: {base_title}"
    else:
        title = base_title

    return title


def available_metric_specs(
        summary_df: pd.DataFrame,
    ) -> list[tuple[str, str, str]]:
    """
    Return metrics that are present in the summary table.
    """
    metric_specs = []
    for metric_name, metric_label in PLOT_METRICS:
        mean_column = f"{metric_name}_mean"
        se_column = f"{metric_name}_standard_error"
        if mean_column in summary_df.columns:
            metric_specs.append((mean_column, se_column, metric_label))

    if not metric_specs:
        raise ValueError("No metric means found in metrics summary.")

    return metric_specs


def available_combined_metric_specs(
        combined_df: pd.DataFrame, comparison_split_name: str = "test",
    ) -> list[tuple[str, str, str, str, str]]:
    """
    Return metrics that have train and comparison estimates in a table.
    """
    metric_specs = []
    for metric_name, metric_label in PLOT_METRICS:
        train_mean_column = f"{metric_name}_mean_train"
        train_se_column = f"{metric_name}_standard_error_train"
        comparison_mean_column = (
            f"{metric_name}_mean_{comparison_split_name}")
        comparison_se_column = (
            f"{metric_name}_standard_error_{comparison_split_name}")
        if (
                train_mean_column in combined_df.columns
                and comparison_mean_column in combined_df.columns):
            metric_specs.append((
                train_mean_column,
                train_se_column,
                comparison_mean_column,
                comparison_se_column,
                metric_label,
            ))

    if not metric_specs:
        raise ValueError("No matching train/test metric means found.")

    return metric_specs


def sort_summary_for_plot(
        summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Sort models so stronger AUPRC models appear near the top of the plot.
    """
    sort_column = "auprc_mean"
    if sort_column not in summary_df.columns:
        sort_column = "accuracy_mean"

    sorted_df = summary_df.sort_values(
        sort_column,
        ascending=False,
        na_position="last",
    )

    return sorted_df


def sort_combined_summary_for_plot(
        combined_df: pd.DataFrame, comparison_split_name: str = "test",
    ) -> pd.DataFrame:
    """
    Sort models so stronger comparison-split models appear near the top.
    """
    sort_column = f"auprc_mean_{comparison_split_name}"
    if sort_column not in combined_df.columns:
        sort_column = f"accuracy_mean_{comparison_split_name}"
    if sort_column not in combined_df.columns:
        sort_column = "auprc_mean_train"
    if sort_column not in combined_df.columns:
        sort_column = "accuracy_mean_train"

    sorted_df = combined_df.sort_values(
        sort_column,
        ascending=False,
        na_position="last",
    )

    return sorted_df


def clean_model_labels(summary_df: pd.DataFrame) -> list[str]:
    """
    Return readable model labels for plotting.
    """
    model_labels = (
        summary_df["model_name"]
        .astype(str)
        .str.replace("__", " / ", regex=False)
        .str.replace("_", " ", regex=False)
        .tolist()
    )

    return model_labels


def clean_feature_name(value: object) -> str:
    """
    Return a stable feature-set name for plotting.
    """
    if pd.isna(value):
        feature_name = "none"
    else:
        feature_name = str(value).strip() or "none"

    return feature_name


def feature_sort_key(feature_name: object) -> tuple:
    """
    Return a sort key that orders feature sets by complexity.
    """
    clean_name = clean_feature_name(feature_name)
    if clean_name == "none":
        sort_key = (0, -1, clean_name)
    else:
        feature_parts = clean_name.split("+")
        part_order = tuple(
            FEATURE_ORDER.index(feature_part)
            if feature_part in FEATURE_ORDER
            else len(FEATURE_ORDER)
            for feature_part in feature_parts
        )
        sort_key = (len(feature_parts), part_order, clean_name)

    return sort_key


def clean_feature_label(feature_name: object) -> str:
    """
    Return a compact multi-line feature-set label.
    """
    clean_name = clean_feature_name(feature_name)
    if clean_name == "none":
        label = "none"
    else:
        label = "\n+".join(clean_name.split("+"))

    return label


def clean_classifier_name(value: object) -> str:
    """
    Return a stable classifier name for plotting.
    """
    if pd.isna(value):
        classifier_name = "unknown"
    else:
        classifier_name = str(value).strip() or "unknown"

    return classifier_name


def classifier_sort_key(classifier_name: object) -> tuple:
    """
    Return a sort key for classifier rows.
    """
    clean_name = clean_classifier_name(classifier_name)
    if clean_name in CLASSIFIER_ORDER:
        sort_key = (CLASSIFIER_ORDER.index(clean_name), clean_name)
    else:
        sort_key = (len(CLASSIFIER_ORDER), clean_name)

    return sort_key


def clean_classifier_label(classifier_name: object) -> str:
    """
    Return a readable classifier label.
    """
    clean_name = clean_classifier_name(classifier_name)
    label = CLASSIFIER_LABELS.get(
        clean_name,
        clean_name.replace("_", " ").title(),
    )

    return label


def finite_or_none(value: object) -> float | None:
    """
    Return a finite float or None for missing values.
    """
    if pd.isna(value):
        finite_value = None
    else:
        numeric_value = float(value)
        finite_value = numeric_value if math.isfinite(numeric_value) else None

    return finite_value
