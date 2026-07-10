"""
Result writing and aggregation helpers for PPI pipeline runs.

This module owns thread-safe output writes and metric summarization. Future
additions should include multiprocessing-friendly result stores, JSONL or
Parquet output, resumable run manifests, and richer experiment summaries.
"""

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd


METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auprc",
    "auroc",
)
SUMMARY_GROUP_COLUMNS = (
    "split",
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
    "target_val_size",
    "actual_train_size",
    "actual_val_size",
    "actual_test_size",
    "n_discarded_edges",
    "discarded_edge_fraction",
    "n_train",
    "n_val",
    "n_test",
)


##################
# Output writing #
##################
@contextmanager
def output_lock(output_path: Path) -> Iterator[None]:
    """
    Lock a sidecar file before writing output.
    """
    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Append a dataframe to a CSV with a file lock and one header row.
    """
    # Check for empty dataframe
    if df.empty:
        return

    # Define dataframe headers and columns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output_path):
        write_header = (
            not output_path.exists()
            or output_path.stat().st_size == 0
        )
        if not write_header:
            existing_columns = pd.read_csv(
                output_path,
                nrows=0,
            ).columns.tolist()
            incoming_columns = df.columns.tolist()
            if existing_columns != incoming_columns:
                raise ValueError(
                    f"Cannot append to {output_path}: existing columns do "
                    "not match the incoming dataframe schema."
                )

        # Save the dataframe
        df.to_csv(output_path, mode="a", header=write_header, index=False)


def write_dataframe_threadsafe(
        df: pd.DataFrame, output_path: Path,
    ) -> None:
    """
    Write a dataframe to CSV with a file lock.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output_path):
        df.to_csv(output_path, index=False)


def reset_output_file(output_path: Path, append_results: bool) -> None:
    """
    Start with a clean output file unless append mode is requested.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not append_results and output_path.exists():
        output_path.unlink()


####################
# Metric summaries #
####################
def default_summary_path(metrics_path: Path) -> Path:
    """
    Build the default summary path from the run-level metrics path.
    """
    summary_path = metrics_path.with_name(
        f"{metrics_path.stem}_summary{metrics_path.suffix}")

    return summary_path


def summarize_metrics(metrics_path: Path) -> pd.DataFrame:
    """
    Summarize per-run metrics with means and standard errors by model type.
    """
    metrics_df = pd.read_csv(metrics_path)
    group_columns = [
        column
        for column in SUMMARY_GROUP_COLUMNS
        if column in metrics_df.columns
    ]
    metric_columns = [
        column for column in METRIC_COLUMNS if column in metrics_df.columns
    ]
    if not metric_columns:
        raise ValueError(
            f"Metrics file contains no supported metric columns: {metrics_path}")

    synthetic_group_column = None
    if not group_columns:
        synthetic_group_column = "_summary_group"
        metrics_df[synthetic_group_column] = 0
        group_columns = [synthetic_group_column]

    grouped = metrics_df.groupby(group_columns, dropna=False)
    n_runs = grouped.size().rename("n_runs").reset_index()
    means = grouped[metric_columns].mean().add_suffix("_mean").reset_index()
    standard_errors = (
        grouped[metric_columns]
        .sem(ddof=1)
        .add_suffix("_standard_error")
        .reset_index()
    )

    summary_df = n_runs.merge(means, on=group_columns)
    summary_df = summary_df.merge(standard_errors, on=group_columns)
    summary_df = summary_df.sort_values(group_columns).reset_index(drop=True)
    if synthetic_group_column is not None:
        summary_df = summary_df.drop(columns=synthetic_group_column)

    return summary_df
