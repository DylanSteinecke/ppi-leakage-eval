"""Cross-strategy benchmark plots."""

from pathlib import Path

import pandas as pd

from .plot_common import (
    COMPARISON_LINE,
    GRID_VALUES,
    SPLIT_STRATEGY_LABELS,
    SPLIT_STRATEGY_ORDER,
    TEST_COLOR,
    TEST_STROKE,
    TRAIN_COLOR,
    TRAIN_STROKE,
)


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

