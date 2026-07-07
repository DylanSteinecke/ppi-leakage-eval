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


################
# Plot helpers #
################
def default_plot_path(summary_path: Path) -> Path:
    """
    Build the default plot path from the metrics summary path.
    """
    plot_path = summary_path.with_suffix(".svg")

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
