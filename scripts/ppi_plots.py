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


################
# Plot helpers #
################
def default_plot_path(summary_path: Path) -> Path:
    """
    Build the default plot path from the metrics summary path.
    """
    plot_path = summary_path.with_suffix(".svg")

    return plot_path


def default_combined_plot_path(train_plot_path: Path) -> Path:
    """
    Build the default train/test comparison plot path.
    """
    train_plot_path = Path(train_plot_path)
    if train_plot_path.name.startswith("train_"):
        plot_name = f"train_test_{train_plot_path.name.removeprefix('train_')}"
    else:
        plot_name = (
            f"{train_plot_path.stem}_train_test{train_plot_path.suffix}")

    plot_path = train_plot_path.with_name(plot_name)

    return plot_path


def default_png_plot_path(svg_plot_path: Path) -> Path:
    """
    Build the default PNG companion path for an SVG plot.
    """
    plot_path = Path(svg_plot_path).with_suffix(".png")

    return plot_path


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
        combined_df: pd.DataFrame,
    ) -> list[tuple[str, str, str, str, str]]:
    """
    Return metrics that have train and test estimates in a combined table.
    """
    metric_specs = []
    for metric_name, metric_label in PLOT_METRICS:
        train_mean_column = f"{metric_name}_mean_train"
        train_se_column = f"{metric_name}_standard_error_train"
        test_mean_column = f"{metric_name}_mean_test"
        test_se_column = f"{metric_name}_standard_error_test"
        if (
                train_mean_column in combined_df.columns
                and test_mean_column in combined_df.columns):
            metric_specs.append((
                train_mean_column,
                train_se_column,
                test_mean_column,
                test_se_column,
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
        combined_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Sort models so stronger test-set models appear near the top.
    """
    sort_column = "auprc_mean_test"
    if sort_column not in combined_df.columns:
        sort_column = "accuracy_mean_test"
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
        "split_seed",
        "target_train_size",
        "actual_train_size",
        "actual_test_size",
        "n_connected_components",
        "n_pruned_pairs",
        "pruned_pair_fraction",
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
    ) -> pd.DataFrame:
    """
    Align train and test summary rows by model configuration.
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
        suffixes=("_train", "_test"),
    )
    if combined_df.empty:
        raise ValueError("No train/test summary rows could be aligned.")

    return combined_df


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
            f"Model performance on the {split_name} split",
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
        plot_path: str | Path,
    ) -> None:
    """
    Plot train and test metric estimates on the same model rows.
    """
    train_summary_path = Path(train_summary_path)
    test_summary_path = Path(test_summary_path)
    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".svg":
        raise ValueError("metrics plot output must end with '.svg'.")

    train_summary_df = pd.read_csv(train_summary_path)
    test_summary_df = pd.read_csv(test_summary_path)
    if train_summary_df.empty:
        raise ValueError(f"Train metrics summary is empty: {train_summary_path}")
    if test_summary_df.empty:
        raise ValueError(f"Test metrics summary is empty: {test_summary_path}")

    combined_df = merge_train_test_summaries(
        train_summary_df=train_summary_df,
        test_summary_df=test_summary_df,
    )
    metric_specs = available_combined_metric_specs(combined_df)
    plot_df = sort_combined_summary_for_plot(combined_df)
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
            "Train/test model performance",
            size=22,
            weight="700",
        ),
        svg_text(
            LEFT_MARGIN,
            58,
            "Green=train, orange=test; connectors show the train-test gap.",
            size=13,
            fill="#555555",
        ),
        svg_circle(legend_x, 32, 5.0, TRAIN_COLOR, TRAIN_STROKE),
        svg_text(legend_x + 12, 36, "Train", 12, fill="#444444"),
        svg_circle(legend_x + 72, 32, 5.0, TEST_COLOR, TEST_STROKE),
        svg_text(legend_x + 84, 36, "Test", 12, fill="#444444"),
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
                svg_parts.append(svg_line(train_x, row_y, test_x, row_y,
                                          COMPARISON_LINE, width=2.0))

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
        plot_path: str | Path,
    ) -> None:
    """
    Plot train and test metric estimates as a raster PNG image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_summary_path = Path(train_summary_path)
    test_summary_path = Path(test_summary_path)
    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".png":
        raise ValueError("metrics PNG plot output must end with '.png'.")

    train_summary_df = pd.read_csv(train_summary_path)
    test_summary_df = pd.read_csv(test_summary_path)
    if train_summary_df.empty:
        raise ValueError(f"Train metrics summary is empty: {train_summary_path}")
    if test_summary_df.empty:
        raise ValueError(f"Test metrics summary is empty: {test_summary_path}")

    combined_df = merge_train_test_summaries(
        train_summary_df=train_summary_df,
        test_summary_df=test_summary_df,
    )
    metric_specs = available_combined_metric_specs(combined_df)
    plot_df = sort_combined_summary_for_plot(combined_df)
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
                axis.plot(
                    [train_mean, test_mean],
                    [y_position, y_position],
                    color=COMPARISON_LINE,
                    linewidth=1.6,
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

    for axis in axes:
        axis.set_yticks(y_positions)
    axes[0].set_yticklabels(model_labels, fontsize=7)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=TRAIN_COLOR,
            label="Train"),
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=TEST_COLOR,
            label="Test"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
    )
    figure.suptitle(
        "Train/test model performance",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        y=0.995,
        ha="left",
    )
    figure.text(
        0.02,
        0.966,
        "Green=train, orange=test; connectors show the train-test gap.",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
