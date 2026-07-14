"""
Plotting helpers for PPI pipeline results.

This module owns publication-style visual summaries derived from machine-
readable result tables. Future additions should include configurable metric
sets, comparison plots across experiments, and optional PNG/PDF export.
"""

import math
from html import escape
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
    "always_positive",
    "always_negative",
)
BASELINE_CLASSIFIERS = ("always_positive", "always_negative")
CLASSIFIER_LABELS = {
    "logistic": "Logistic",
    "linear_svm": "Linear SVM",
    "sgd_logistic": "SGD Logistic",
    "torch_mlp": "Torch MLP",
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


def plot_benchmark_train_val_f1(
        summary: pd.DataFrame | str | Path,
        plot_path: str | Path,
        execution_id_prefix: str | None = None,
    ) -> bool:
    """
    Plot train/validation F1 dumbbells across all available split strategies.

    Absolute F1 values are shown by the two marker positions. The connecting
    arrow exposes the validation-minus-train generalization gap without
    requiring a second figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".png":
        raise ValueError("benchmark train/validation plot must end with '.png'.")

    if isinstance(summary, pd.DataFrame):
        summary_df = summary.copy()
    else:
        summary_df = pd.read_csv(summary)

    required_columns = {
        "summary_split",
        "split_strategy",
        "model_name",
        "f1_mean",
    }
    missing_columns = required_columns - set(summary_df.columns)
    if missing_columns:
        return False

    if execution_id_prefix is not None:
        if "execution_id" not in summary_df.columns:
            return False
        execution_ids = summary_df["execution_id"].fillna("").astype(str)
        summary_df = summary_df[
            execution_ids.str.startswith(execution_id_prefix)
        ]

    summary_df = summary_df[
        summary_df["summary_split"].isin(("train", "val"))
    ].copy()
    if summary_df.empty:
        return False

    plot_df = summary_df.pivot_table(
        index=["split_strategy", "model_name"],
        columns="summary_split",
        values="f1_mean",
        aggfunc="mean",
    ).reset_index()
    if "train" not in plot_df.columns or "val" not in plot_df.columns:
        return False
    plot_df = plot_df.dropna(subset=["train", "val"])
    if plot_df.empty:
        return False

    observed_strategies = set(plot_df["split_strategy"].astype(str))
    strategies = [
        strategy
        for strategy in SPLIT_STRATEGY_ORDER
        if strategy in observed_strategies
    ]
    strategies.extend(sorted(observed_strategies - set(strategies)))

    model_order = (
        plot_df.groupby("model_name", sort=False)["val"]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    model_labels = [
        model_name.replace("__", " / ").replace("_", " ")
        for model_name in model_order
    ]
    y_positions = list(range(len(model_order)))
    panel_height = max(3.0, 0.28 * len(model_order) + 1.1)
    header_height = 0.95
    figure_height = panel_height * len(strategies) + header_height
    figure, axes = plt.subplots(
        len(strategies),
        1,
        figsize=(12.0, figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, strategy in zip(axes, strategies):
        strategy_df = (
            plot_df[plot_df["split_strategy"].astype(str) == strategy]
            .set_index("model_name")
        )
        axis.set_title(
            SPLIT_STRATEGY_LABELS.get(
                strategy,
                strategy.replace("_", " ").title(),
            ),
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks(list(GRID_VALUES))
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("F1")
        axis.set_ylim(len(model_order) - 0.5, -0.5)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_labels, fontsize=8.0)

        for y_position, model_name in zip(y_positions, model_order):
            axis.axhline(
                y_position,
                color="#F3F3F3",
                linewidth=0.7,
                zorder=0,
            )
            if model_name not in strategy_df.index:
                continue
            row = strategy_df.loc[model_name]
            train_f1 = float(row["train"])
            val_f1 = float(row["val"])
            axis.annotate(
                "",
                xy=(val_f1, y_position),
                xytext=(train_f1, y_position),
                arrowprops={
                    "arrowstyle": "->",
                    "color": COMPARISON_LINE,
                    "lw": 1.5,
                    "shrinkA": 5.0,
                    "shrinkB": 5.0,
                    "mutation_scale": 9.0,
                },
                zorder=1,
            )
            axis.scatter(
                train_f1,
                y_position,
                color=TRAIN_COLOR,
                edgecolor=TRAIN_STROKE,
                linewidth=0.7,
                s=35,
                zorder=3,
            )
            axis.scatter(
                val_f1,
                y_position,
                color=TEST_COLOR,
                edgecolor=TEST_STROKE,
                linewidth=0.7,
                s=31,
                zorder=4,
            )

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=TRAIN_COLOR,
            label="Train"),
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=TEST_COLOR,
            label="Validation"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        ncols=2,
        frameon=False,
    )
    figure.suptitle(
        "Train vs validation F1 across split strategies",
        fontsize=17,
        fontweight="bold",
        x=0.01,
        y=0.995,
        ha="left",
    )
    figure.text(
        0.01,
        1.0 - 0.72 / figure_height,
        (
            "Marker positions show absolute F1; arrows point from train to "
            "validation, so their direction and length show the "
            "generalization gap."
        ),
        fontsize=9.5,
        color="#555555",
    )
    plot_top = 1.0 - header_height / figure_height
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, plot_top),
        h_pad=1.5,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return True


def estimate_to_x(value: float, plot_x: float) -> float:
    """
    Convert a 0-1 metric value to an SVG x coordinate.
    """
    clamped_value = min(max(value, 0.0), 1.0)
    x_position = plot_x + clamped_value * PLOT_WIDTH

    return x_position


def svg_text(
        x: float, y: float, text: str, size: int, fill: str = "#222222",
        anchor: str = "start", weight: str = "400",
    ) -> str:
    """
    Create an SVG text element.
    """
    text_element = (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="{FONT_FAMILY}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">'
        f"{escape(text)}</text>"
    )

    return text_element


def svg_line(
        x1: float, y1: float, x2: float, y2: float, stroke: str,
        width: float = 1.0,
    ) -> str:
    """
    Create an SVG line element.
    """
    line_element = (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}" />'
    )

    return line_element


def svg_circle(
        cx: float, cy: float, radius: float, fill: str, stroke: str,
        width: float = 1.0,
    ) -> str:
    """
    Create an SVG circle element.
    """
    circle_element = (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}" />')

    return circle_element


def svg_horizontal_arrow(
        x1: float, y: float, x2: float, stroke: str, width: float = 1.8,
    ) -> str:
    """
    Create a horizontal SVG arrow from x1 to x2.
    """
    delta = x2 - x1
    if abs(delta) < ARROW_MARKER_GAP * 2.0 + ARROW_HEAD_LENGTH:
        arrow = ""
    else:
        direction = 1.0 if delta > 0 else -1.0
        start_x = x1 + direction * ARROW_MARKER_GAP
        tip_x = x2 - direction * ARROW_MARKER_GAP
        base_x = tip_x - direction * ARROW_HEAD_LENGTH
        points = (
            f"{tip_x:.1f},{y:.1f} "
            f"{base_x:.1f},{y - ARROW_HEAD_HALF_HEIGHT:.1f} "
            f"{base_x:.1f},{y + ARROW_HEAD_HALF_HEIGHT:.1f}"
        )
        arrow = "\n".join((
            svg_line(start_x, y, base_x, y, stroke, width=width),
            f'<polygon points="{points}" fill="{stroke}" />',
        ))

    return arrow


def combined_summary_key_columns(
        train_summary_df: pd.DataFrame, test_summary_df: pd.DataFrame,
    ) -> list[str]:
    """
    Return stable columns used to align train and test model summaries.
    """
    preferred_columns = (
        "model_name",
        "features",
        "classifier",
        "k",
        "bm25_k1",
        "bm25_b",
        "max_iter",
        "split_strategy",
        "split_name",
        "split_seed",
        "target_train_size",
        "actual_train_size",
        "actual_test_size",
        "n_discarded_edges",
        "discarded_edge_fraction",
        "n_train",
        "n_test",
        "n_runs",
    )
    train_columns = set(train_summary_df.columns)
    test_columns = set(test_summary_df.columns)
    key_columns = [
        column
        for column in preferred_columns
        if column in train_columns and column in test_columns
    ]
    if "model_name" not in key_columns:
        raise ValueError("Train/test summaries must contain model_name.")

    return key_columns


def merge_train_test_summaries(
        train_summary_df: pd.DataFrame, test_summary_df: pd.DataFrame,
        comparison_split_name: str = "test",
    ) -> pd.DataFrame:
    """
    Align train and comparison summary rows by model configuration.
    """
    key_columns = combined_summary_key_columns(
        train_summary_df=train_summary_df,
        test_summary_df=test_summary_df,
    )
    train_plot_df = train_summary_df.drop(columns=["split"], errors="ignore")
    test_plot_df = test_summary_df.drop(columns=["split"], errors="ignore")
    combined_df = train_plot_df.merge(
        test_plot_df,
        on=key_columns,
        how="outer",
        suffixes=("_train", f"_{comparison_split_name}"),
    )
    if combined_df.empty:
        raise ValueError(
            f"No train/{comparison_split_name} summary rows could be "
            "aligned.")

    return combined_df


def load_combined_summary(
        train_summary_path: str | Path, comparison_summary_path: str | Path,
        comparison_split_name: str,
    ) -> pd.DataFrame:
    """
    Read, validate, and align train/comparison metric summaries.
    """
    train_summary_path = Path(train_summary_path)
    comparison_summary_path = Path(comparison_summary_path)
    train_summary_df = pd.read_csv(train_summary_path)
    comparison_summary_df = pd.read_csv(comparison_summary_path)
    if train_summary_df.empty:
        raise ValueError(f"Train metrics summary is empty: {train_summary_path}")
    if comparison_summary_df.empty:
        comparison_label = comparison_split_name.title()
        raise ValueError(
            f"{comparison_label} metrics summary is empty: "
            f"{comparison_summary_path}")

    return merge_train_test_summaries(
        train_summary_df=train_summary_df,
        test_summary_df=comparison_summary_df,
        comparison_split_name=comparison_split_name,
    )


def add_estimate_marker(
        svg_parts: list[str], plot_x: float, row_y: float,
        mean_value: float | None, se_value: float | None, fill: str,
        stroke: str, radius: float,
    ) -> None:
    """
    Add one metric point and optional standard-error whisker.
    """
    if mean_value is None:
        return

    se_value = 0.0 if se_value is None else se_value
    low_value = max(0.0, mean_value - se_value)
    high_value = min(1.0, mean_value + se_value)
    mean_x = estimate_to_x(mean_value, plot_x)
    low_x = estimate_to_x(low_value, plot_x)
    high_x = estimate_to_x(high_value, plot_x)

    svg_parts.append(svg_line(low_x, row_y, high_x, row_y, stroke,
                              width=1.4))
    svg_parts.append(svg_line(low_x, row_y - 4, low_x, row_y + 4, stroke,
                              width=1.2))
    svg_parts.append(svg_line(high_x, row_y - 4, high_x, row_y + 4, stroke,
                              width=1.2))
    svg_parts.append(svg_circle(mean_x, row_y, radius, fill, stroke))


#################
# Summary plots #
#################
def plot_metrics_summary(
        summary_path: str | Path, plot_path: str | Path, split_name: str,
    ) -> None:
    """
    Plot mean metric estimates and standard errors from metrics_summary.csv.
    """
    # Load and check the summary table
    summary_path = Path(summary_path)
    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".svg":
        raise ValueError("metrics plot output must end with '.svg'.")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"Metrics summary is empty: {summary_path}")

    title = title_with_split_strategy(
        base_title=f"Model performance on the {split_name} split",
        summary_df=summary_df,
    )
    metric_specs = available_metric_specs(summary_df)
    plot_df = sort_summary_for_plot(summary_df)
    model_labels = clean_model_labels(plot_df)

    # Define figure layout with enough room for readable model labels
    n_models = len(plot_df)
    n_metrics = len(metric_specs)
    n_cols = min(3, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    panel_height = PANEL_HEADER + n_models * ROW_HEIGHT + AXIS_HEIGHT
    figure_width = (
        LEFT_MARGIN
        + LABEL_WIDTH
        + n_cols * PLOT_WIDTH
        + (n_cols - 1) * PANEL_GAP
        + RIGHT_MARGIN
    )
    figure_height = (
        TOP_MARGIN
        + n_rows * panel_height
        + BOTTOM_MARGIN
    )

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{figure_width:.0f}" height="{figure_height:.0f}" '
            f'viewBox="0 0 {figure_width:.0f} {figure_height:.0f}">'
        ),
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        svg_text(
            LEFT_MARGIN,
            34,
            title,
            size=22,
            weight="700",
        ),
        svg_text(
            LEFT_MARGIN,
            58,
            "Points show mean estimates; whiskers show standard errors.",
            size=13,
            fill="#555555",
        ),
    ]

    for metric_idx, (mean_column, se_column, metric_label) in enumerate(
            metric_specs):
        row_idx = metric_idx // n_cols
        col_idx = metric_idx % n_cols
        plot_x = LEFT_MARGIN + LABEL_WIDTH + col_idx * (
            PLOT_WIDTH + PANEL_GAP)
        plot_y = TOP_MARGIN + row_idx * panel_height + PANEL_HEADER
        axis_y = plot_y + n_models * ROW_HEIGHT

        svg_parts.append(
            svg_text(plot_x, plot_y - 18, metric_label, 15, weight="700"))

        for grid_value in GRID_VALUES:
            grid_x = estimate_to_x(grid_value, plot_x)
            svg_parts.append(
                svg_line(grid_x, plot_y - 8, grid_x, axis_y, "#E6E6E6"))
            svg_parts.append(
                svg_text(
                    grid_x,
                    axis_y + 22,
                    f"{grid_value:.2g}",
                    size=10,
                    fill="#555555",
                    anchor="middle",
                )
            )

        svg_parts.append(svg_line(plot_x, axis_y, plot_x + PLOT_WIDTH,
                                  axis_y, "#999999"))

        for model_idx, (_, model_row) in enumerate(plot_df.iterrows()):
            row_y = plot_y + model_idx * ROW_HEIGHT + ROW_HEIGHT / 2
            mean_value = finite_or_none(model_row[mean_column])
            se_value = (
                finite_or_none(model_row[se_column])
                if se_column in plot_df.columns
                else 0.0
            )

            if col_idx == 0:
                svg_parts.append(
                    svg_text(
                        LEFT_MARGIN,
                        row_y + 4,
                        model_labels[model_idx],
                        size=13,
                    )
                )

            svg_parts.append(svg_line(plot_x, row_y, plot_x + PLOT_WIDTH,
                                      row_y, "#F4F4F4"))

            if mean_value is None:
                svg_parts.append(
                    svg_text(
                        plot_x + 5,
                        row_y + 4,
                        "NA",
                        size=12,
                        fill="#777777",
                    )
                )
                continue

            se_value = 0.0 if se_value is None else se_value
            low_value = max(0.0, mean_value - se_value)
            high_value = min(1.0, mean_value + se_value)
            mean_x = estimate_to_x(mean_value, plot_x)
            low_x = estimate_to_x(low_value, plot_x)
            high_x = estimate_to_x(high_value, plot_x)

            svg_parts.append(svg_line(low_x, row_y, high_x, row_y,
                                      "#222222", width=1.6))
            svg_parts.append(svg_line(low_x, row_y - 5, low_x, row_y + 5,
                                      "#222222", width=1.4))
            svg_parts.append(svg_line(high_x, row_y - 5, high_x, row_y + 5,
                                      "#222222", width=1.4))
            svg_parts.append(
                f'<circle cx="{mean_x:.1f}" cy="{row_y:.1f}" r="4.7" '
                f'fill="#4C78A8" stroke="#1F3A5F" stroke-width="1.0" />')

    svg_parts.append("</svg>")

    # Save the figure
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("\n".join(svg_parts), encoding="utf-8")


def plot_train_test_metrics_summary(
        train_summary_path: str | Path, test_summary_path: str | Path,
        plot_path: str | Path, comparison_split_name: str = "test",
    ) -> None:
    """
    Plot train and comparison metric estimates on the same model rows.
    """
    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".svg":
        raise ValueError("metrics plot output must end with '.svg'.")

    combined_df = load_combined_summary(
        train_summary_path=train_summary_path,
        comparison_summary_path=test_summary_path,
        comparison_split_name=comparison_split_name,
    )
    comparison_label = comparison_split_name.title()
    title = title_with_split_strategy(
        base_title=f"Train/{comparison_split_name} model performance",
        summary_df=combined_df,
    )
    metric_specs = available_combined_metric_specs(
        combined_df,
        comparison_split_name=comparison_split_name,
    )
    plot_df = sort_combined_summary_for_plot(
        combined_df,
        comparison_split_name=comparison_split_name,
    )
    model_labels = clean_model_labels(plot_df)

    n_models = len(plot_df)
    n_metrics = len(metric_specs)
    n_cols = min(3, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    panel_height = PANEL_HEADER + n_models * ROW_HEIGHT + AXIS_HEIGHT
    figure_width = (
        LEFT_MARGIN
        + LABEL_WIDTH
        + n_cols * PLOT_WIDTH
        + (n_cols - 1) * PANEL_GAP
        + RIGHT_MARGIN
    )
    figure_height = (
        TOP_MARGIN
        + n_rows * panel_height
        + BOTTOM_MARGIN
    )
    legend_x = figure_width - RIGHT_MARGIN - 180

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{figure_width:.0f}" height="{figure_height:.0f}" '
            f'viewBox="0 0 {figure_width:.0f} {figure_height:.0f}">'
        ),
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        svg_text(
            LEFT_MARGIN,
            34,
            title,
            size=22,
            weight="700",
        ),
        svg_text(
            LEFT_MARGIN,
            58,
            (
                f"Green=train, orange={comparison_split_name}; arrows point "
                f"from train to {comparison_split_name}."
            ),
            size=13,
            fill="#555555",
        ),
        svg_circle(legend_x, 32, 5.0, TRAIN_COLOR, TRAIN_STROKE),
        svg_text(legend_x + 12, 36, "Train", 12, fill="#444444"),
        svg_circle(legend_x + 72, 32, 5.0, TEST_COLOR, TEST_STROKE),
        svg_text(legend_x + 84, 36, comparison_label, 12, fill="#444444"),
    ]

    for metric_idx, (
            train_mean_column,
            train_se_column,
            test_mean_column,
            test_se_column,
            metric_label,
        ) in enumerate(metric_specs):
        row_idx = metric_idx // n_cols
        col_idx = metric_idx % n_cols
        plot_x = LEFT_MARGIN + LABEL_WIDTH + col_idx * (
            PLOT_WIDTH + PANEL_GAP)
        plot_y = TOP_MARGIN + row_idx * panel_height + PANEL_HEADER
        axis_y = plot_y + n_models * ROW_HEIGHT

        svg_parts.append(
            svg_text(plot_x, plot_y - 18, metric_label, 15, weight="700"))

        for grid_value in GRID_VALUES:
            grid_x = estimate_to_x(grid_value, plot_x)
            svg_parts.append(
                svg_line(grid_x, plot_y - 8, grid_x, axis_y, "#E6E6E6"))
            svg_parts.append(
                svg_text(
                    grid_x,
                    axis_y + 22,
                    f"{grid_value:.2g}",
                    size=10,
                    fill="#555555",
                    anchor="middle",
                )
            )

        svg_parts.append(svg_line(plot_x, axis_y, plot_x + PLOT_WIDTH,
                                  axis_y, "#999999"))

        for model_idx, (_, model_row) in enumerate(plot_df.iterrows()):
            row_y = plot_y + model_idx * ROW_HEIGHT + ROW_HEIGHT / 2
            train_mean = finite_or_none(model_row.get(train_mean_column))
            test_mean = finite_or_none(model_row.get(test_mean_column))
            train_se = (
                finite_or_none(model_row.get(train_se_column))
                if train_se_column in plot_df.columns
                else None
            )
            test_se = (
                finite_or_none(model_row.get(test_se_column))
                if test_se_column in plot_df.columns
                else None
            )

            if col_idx == 0:
                svg_parts.append(
                    svg_text(
                        LEFT_MARGIN,
                        row_y + 4,
                        model_labels[model_idx],
                        size=13,
                    )
                )

            svg_parts.append(svg_line(plot_x, row_y, plot_x + PLOT_WIDTH,
                                      row_y, "#F4F4F4"))

            if train_mean is None and test_mean is None:
                svg_parts.append(
                    svg_text(
                        plot_x + 5,
                        row_y + 4,
                        "NA",
                        size=12,
                        fill="#777777",
                    )
                )
                continue

            if train_mean is not None and test_mean is not None:
                train_x = estimate_to_x(train_mean, plot_x)
                test_x = estimate_to_x(test_mean, plot_x)
                svg_parts.append(
                    svg_horizontal_arrow(
                        train_x,
                        row_y,
                        test_x,
                        COMPARISON_LINE,
                    )
                )

            add_estimate_marker(
                svg_parts=svg_parts,
                plot_x=plot_x,
                row_y=row_y,
                mean_value=train_mean,
                se_value=train_se,
                fill=TRAIN_COLOR,
                stroke=TRAIN_STROKE,
                radius=5.2,
            )
            add_estimate_marker(
                svg_parts=svg_parts,
                plot_x=plot_x,
                row_y=row_y,
                mean_value=test_mean,
                se_value=test_se,
                fill=TEST_COLOR,
                stroke=TEST_STROKE,
                radius=4.0,
            )

    svg_parts.append("</svg>")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("\n".join(svg_parts), encoding="utf-8")


def plot_train_test_metrics_summary_png(
        train_summary_path: str | Path, test_summary_path: str | Path,
        plot_path: str | Path, comparison_split_name: str = "test",
    ) -> None:
    """
    Plot train and comparison metric estimates as a raster PNG image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".png":
        raise ValueError("metrics PNG plot output must end with '.png'.")

    combined_df = load_combined_summary(
        train_summary_path=train_summary_path,
        comparison_summary_path=test_summary_path,
        comparison_split_name=comparison_split_name,
    )
    comparison_label = comparison_split_name.title()
    title = title_with_split_strategy(
        base_title=f"Train/{comparison_split_name} model performance",
        summary_df=combined_df,
    )
    metric_specs = available_combined_metric_specs(
        combined_df,
        comparison_split_name=comparison_split_name,
    )
    plot_df = sort_combined_summary_for_plot(
        combined_df,
        comparison_split_name=comparison_split_name,
    )
    model_labels = clean_model_labels(plot_df)

    n_models = len(plot_df)
    figure_height = max(8.5, 0.28 * n_models + 2.2)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, figure_height),
        sharey=True,
    )
    axes = axes.ravel()
    y_positions = list(range(n_models))

    for axis, (
            train_mean_column,
            train_se_column,
            test_mean_column,
            test_se_column,
            metric_label,
        ) in zip(axes, metric_specs):
        axis.set_title(metric_label, fontsize=12, fontweight="bold")
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks(list(GRID_VALUES))
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        axis.set_axisbelow(True)

        for y_position, (_, model_row) in zip(
                y_positions, plot_df.iterrows()):
            train_mean = finite_or_none(model_row.get(train_mean_column))
            test_mean = finite_or_none(model_row.get(test_mean_column))
            train_se = (
                finite_or_none(model_row.get(train_se_column))
                if train_se_column in plot_df.columns
                else None
            )
            test_se = (
                finite_or_none(model_row.get(test_se_column))
                if test_se_column in plot_df.columns
                else None
            )

            if train_mean is not None and test_mean is not None:
                axis.annotate(
                    "",
                    xy=(test_mean, y_position),
                    xytext=(train_mean, y_position),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": COMPARISON_LINE,
                        "lw": 1.6,
                        "shrinkA": 6.0,
                        "shrinkB": 6.0,
                        "mutation_scale": 9.5,
                    },
                    zorder=1,
                )

            if train_mean is not None:
                axis.errorbar(
                    train_mean,
                    y_position,
                    xerr=0.0 if train_se is None else train_se,
                    fmt="o",
                    color=TRAIN_COLOR,
                    ecolor=TRAIN_STROKE,
                    markersize=5.8,
                    elinewidth=1.1,
                    capsize=2.5,
                    zorder=3,
                )
            if test_mean is not None:
                axis.errorbar(
                    test_mean,
                    y_position,
                    xerr=0.0 if test_se is None else test_se,
                    fmt="o",
                    color=TEST_COLOR,
                    ecolor=TEST_STROKE,
                    markersize=5.0,
                    elinewidth=1.1,
                    capsize=2.5,
                    zorder=4,
                )

        axis.invert_yaxis()

    for axis in axes[len(metric_specs):]:
        axis.axis("off")

    metric_axes = axes[:len(metric_specs)]
    for axis in metric_axes:
        axis.set_yticks(y_positions)
    metric_axes[0].set_yticklabels(model_labels, fontsize=7)
    for axis in metric_axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=TRAIN_COLOR,
            label="Train"),
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=TEST_COLOR,
            label=comparison_label),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
    )
    figure.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        x=0.02,
        y=0.995,
        ha="left",
    )
    figure.text(
        0.02,
        0.966,
        (
            f"Green=train, orange={comparison_split_name}; arrows point "
            f"from train to {comparison_split_name}."
        ),
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def plot_train_test_f1_heatmap(
        train_summary_path: str | Path, test_summary_path: str | Path,
        plot_path: str | Path, comparison_split_name: str = "test",
    ) -> dict[str, Path]:
    """
    Plot train, comparison, and comparison-minus-train F1 heatmaps in one PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".png":
        raise ValueError("F1 heatmap output path must end with '.png'.")

    comparison_label = comparison_split_name.title()
    combined_df = load_combined_summary(
        train_summary_path=train_summary_path,
        comparison_summary_path=test_summary_path,
        comparison_split_name=comparison_split_name,
    )
    required_columns = {
        "classifier",
        "features",
        "f1_mean_train",
        f"f1_mean_{comparison_split_name}",
    }
    missing_columns = required_columns - set(combined_df.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot plot F1 heatmaps; missing columns: {missing_columns}")

    plot_df = combined_df.copy()
    plot_df["classifier_plot"] = plot_df["classifier"].map(
        clean_classifier_name)
    plot_df["features_plot"] = plot_df["features"].map(clean_feature_name)

    learned_classifiers = sorted(
        {
            row["classifier_plot"]
            for _, row in plot_df.iterrows()
            if (
                row["features_plot"] != "none"
                and row["classifier_plot"] not in BASELINE_CLASSIFIERS
            )
        },
        key=classifier_sort_key,
    )
    learned_features = sorted(
        {
            row["features_plot"]
            for _, row in plot_df.iterrows()
            if row["features_plot"] != "none"
        },
        key=feature_sort_key,
    )
    baseline_classifiers = sorted(
        {
            row["classifier_plot"]
            for _, row in plot_df.iterrows()
            if (
                row["features_plot"] == "none"
                and row["classifier_plot"] in BASELINE_CLASSIFIERS
            )
        },
        key=classifier_sort_key,
    )
    if not learned_classifiers and not baseline_classifiers:
        raise ValueError("Cannot plot F1 heatmaps without classifiers.")

    value_lookup = {
        "train": {},
        comparison_split_name: {},
        "difference": {},
    }
    for _, row in plot_df.iterrows():
        classifier_name = row["classifier_plot"]
        feature_name = row["features_plot"]
        value_key = (classifier_name, feature_name)
        train_f1 = finite_or_none(row["f1_mean_train"])
        comparison_f1 = finite_or_none(
            row[f"f1_mean_{comparison_split_name}"])
        if train_f1 is not None:
            value_lookup["train"][value_key] = train_f1
        if comparison_f1 is not None:
            value_lookup[comparison_split_name][value_key] = comparison_f1
        if train_f1 is not None and comparison_f1 is not None:
            value_lookup["difference"][value_key] = comparison_f1 - train_f1

    red_color_map = LinearSegmentedColormap.from_list(
        "f1_reds",
        ("#FFF5F0", "#FB6A4A", "#67000D"),
    )
    red_color_map.set_bad("#F2F2F2")
    difference_color_map = LinearSegmentedColormap.from_list(
        "f1_difference_reds",
        ("#FFFFFF", "#FB6A4A", "#67000D"),
    )
    difference_color_map.set_bad("#F2F2F2")

    finite_differences = [
        value
        for value in value_lookup["difference"].values()
        if math.isfinite(value)
    ]
    difference_limit = max(
        [abs(value) for value in finite_differences] + [0.05])

    variant_specs = {
        "train": {
            "row_title": "Train F1",
            "color_bar_label": "Training F1",
            "color_map": red_color_map,
            "vmin": 0.0,
            "vmax": 1.0,
            "norm": None,
        },
        comparison_split_name: {
            "row_title": f"{comparison_label} F1",
            "color_bar_label": f"{comparison_label} F1",
            "color_map": red_color_map,
            "vmin": 0.0,
            "vmax": 1.0,
            "norm": None,
        },
        "difference": {
            "row_title": f"{comparison_label}-train F1",
            "color_bar_label": f"|{comparison_label} F1 - Train F1|",
            "color_map": difference_color_map,
            "vmin": 0.0,
            "vmax": difference_limit,
            "norm": None,
        },
    }

    baseline_features = ["none"] if baseline_classifiers else []
    max_panel_rows = max(len(learned_classifiers), len(baseline_classifiers))
    figure_width = max(
        9.5,
        0.72 * max(len(learned_features), 1)
        + (3.2 if baseline_classifiers else 0.0)
        + 4.2,
    )
    row_height = max(2.8, 0.62 * max_panel_rows + 1.45)
    figure_height = max(8.5, row_height * 3 + 1.4)

    def panel_values(
            variant_name: str, row_names: list[str], column_names: list[str],
        ) -> list[list[float]]:
        """
        Return a dense heatmap matrix for one panel.
        """
        values = [
            [
                value_lookup[variant_name].get(
                    (row_name, column_name),
                    float("nan"),
                )
                for column_name in column_names
            ]
            for row_name in row_names
        ]

        return values

    def panel_color_values(
            variant_name: str, values: list[list[float]],
        ) -> list[list[float]]:
        """
        Return values used for cell color intensity.
        """
        if variant_name == "difference":
            color_values = [
                [
                    abs(value) if math.isfinite(value) else float("nan")
                    for value in row
                ]
                for row in values
            ]
        else:
            color_values = values

        return color_values

    def add_heatmap_panel(
            axis, variant_name: str, variant_spec: dict,
            row_names: list[str], column_names: list[str], panel_title: str,
            show_y_label: bool, y_tick_side: str = "left",
            compact_row_labels: bool = False,
        ):
        """
        Add one heatmap panel and return the image used for the color bar.
        """
        values = panel_values(variant_name, row_names, column_names)
        color_values = panel_color_values(variant_name, values)
        image = axis.imshow(
            color_values,
            cmap=variant_spec["color_map"],
            vmin=variant_spec["vmin"],
            vmax=variant_spec["vmax"],
            norm=variant_spec["norm"],
            aspect="equal",
        )
        axis.set_xlim(-0.5, len(column_names) - 0.5)
        axis.set_ylim(len(row_names) - 0.5, -0.5)
        axis.set_anchor("NW")

        axis.set_title(panel_title, fontsize=11, fontweight="bold", pad=10)
        axis.set_xticks(range(len(column_names)))
        axis.set_xticklabels(
            [clean_feature_label(feature_name) for feature_name in column_names],
            fontsize=8,
        )
        axis.set_yticks(range(len(row_names)))
        row_labels = [
            clean_classifier_label(classifier_name)
            for classifier_name in row_names
        ]
        if compact_row_labels:
            row_labels = [
                row_label.removeprefix("Always ")
                for row_label in row_labels
            ]
        axis.set_yticklabels(
            row_labels,
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel("Feature set", labelpad=12)
        axis.set_ylabel("Classifier" if show_y_label else "", labelpad=12)
        if y_tick_side == "right":
            axis.yaxis.tick_right()
            axis.tick_params(
                axis="y",
                labelleft=False,
                labelright=True,
                pad=6,
            )
        if len(column_names) > 1:
            axis.set_xticks(
                [
                    column_index - 0.5
                    for column_index in range(1, len(column_names))
                ],
                minor=True,
            )
        if len(row_names) > 1:
            axis.set_yticks(
                [
                    row_index - 0.5
                    for row_index in range(1, len(row_names))
                ],
                minor=True,
            )
        axis.grid(which="minor", color="#FFFFFF", linewidth=1.4)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.tick_params(axis="x", length=0)
        axis.tick_params(axis="y", length=0)

        for row_index in range(len(row_names)):
            for column_index in range(len(column_names)):
                cell_value = finite_or_none(values[row_index][column_index])
                if cell_value is None:
                    axis.text(
                        column_index,
                        row_index,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#777777",
                    )
                    continue

                if variant_name == "difference":
                    text_color = (
                        "#FFFFFF"
                        if abs(cell_value) >= difference_limit * 0.55
                        else "#222222"
                    )
                    display_value = (
                        0.0 if abs(cell_value) < 0.005 else cell_value
                    )
                    cell_text = f"{display_value:+.2f}"
                else:
                    text_color = "#FFFFFF" if cell_value >= 0.62 else "#4A1111"
                    cell_text = f"{cell_value:.2f}"

                axis.text(
                    column_index,
                    row_index,
                    cell_text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=text_color,
                )

        return image

    width_ratios = []
    if learned_classifiers and learned_features:
        width_ratios.append(len(learned_features))
    if baseline_classifiers:
        width_ratios.append(len(baseline_features))
        width_ratios.append(0.70)
    width_ratios.append(0.35)

    figure = plt.figure(figsize=(figure_width, figure_height))
    grid = figure.add_gridspec(
        3,
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=[1, 1, 1],
        wspace=0.22,
        hspace=0.70,
    )

    for row_index, (variant_name, variant_spec) in enumerate(
            variant_specs.items()):
        grid_column = 0
        panel_images = []
        if learned_classifiers and learned_features:
            axis = figure.add_subplot(grid[row_index, grid_column])
            panel_images.append(
                add_heatmap_panel(
                    axis=axis,
                    variant_name=variant_name,
                    variant_spec=variant_spec,
                    row_names=learned_classifiers,
                    column_names=learned_features,
                    panel_title=(
                        f"{variant_spec['row_title']}\nFeature Models"),
                    show_y_label=True,
                    y_tick_side="left",
                    compact_row_labels=False,
                )
            )
            grid_column += 1

        if baseline_classifiers:
            axis = figure.add_subplot(grid[row_index, grid_column])
            panel_images.append(
                add_heatmap_panel(
                    axis=axis,
                    variant_name=variant_name,
                    variant_spec=variant_spec,
                    row_names=baseline_classifiers,
                    column_names=baseline_features,
                    panel_title=f"{variant_spec['row_title']}\nBaselines",
                    show_y_label=not learned_classifiers,
                    y_tick_side="right",
                    compact_row_labels=True,
                )
            )
            grid_column += 1
            spacer_axis = figure.add_subplot(grid[row_index, grid_column])
            spacer_axis.axis("off")
            grid_column += 1

        color_bar_axis = figure.add_subplot(grid[row_index, grid_column])
        color_bar = figure.colorbar(panel_images[0], cax=color_bar_axis)
        color_bar.set_label(
            variant_spec["color_bar_label"],
            rotation=270,
            labelpad=14,
        )

    title = title_with_split_strategy(
        base_title=(
            f"Train/{comparison_split_name} F1 heatmaps by feature set and "
            "classifier"),
        summary_df=combined_df,
    )
    figure.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
        x=0.02,
        y=0.985,
        ha="left",
    )
    figure.text(
        0.02,
        0.955,
        (
            f"Rows show train, {comparison_split_name}, and "
            f"{comparison_split_name}-train F1; difference color uses "
            "absolute gap."
        ),
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.94,
        top=0.90,
        bottom=0.06,
        wspace=0.24,
        hspace=0.72,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    return {"stacked": plot_path}
