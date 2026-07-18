"""
Metric aggregation helpers for PPI pipeline runs.

This module owns metric summarization and stable summary schemas.
"""

from pathlib import Path

import pandas as pd


METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auprc",
    "auroc",
)
SUMMARY_AUXILIARY_COLUMNS = (
    "decision_threshold",
    "threshold_metric_value",
)
SUMMARY_GROUP_COLUMNS = (
    "evaluation_schema_version",
    "task",
    "split",
    "model_name",
    "estimator_id",
    "estimator_params",
    "feature_spec_sha256",
    "feature_identity",
    "fitted_extractor_sha256",
    "configuration_id",
    "features",
    "matrix_source",
    "matrix_schema_id",
    "pair_composition_schema_id",
    "matrix_contract_sha256",
    "row_identity_sha256",
    "matrix_sha256",
    "matrix_persisted",
    "reporting_group",
    "k",
    "bm25_k1",
    "bm25_b",
    "encoder_adapter",
    "encoder_fingerprint",
    "encoder_model",
    "encoder_revision",
    "encoder_pooling",
    "encoder_maximum_length",
    "encoder_precision",
    "encoder_label_independent",
    "encoder_checkpoint_sha256",
    "encoder_training_split_sha256",
    "max_iter",
    "threshold_selection",
    "threshold_metric",
    "split_strategy",
    "split_name",
    "split_seed",
    "target_train_size",
    "target_val_size",
    "target_test_size",
    "actual_train_size",
    "actual_val_size",
    "actual_test_size",
    "n_discarded_edges",
    "discarded_edge_fraction",
    "n_train",
    "n_val",
    "n_test",
)


####################
# Metric summaries #
####################
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
    value_columns = [
        column
        for column in METRIC_COLUMNS + SUMMARY_AUXILIARY_COLUMNS
        if column in metrics_df.columns
    ]
    if not value_columns:
        raise ValueError(
            f"Metrics file contains no supported metric columns: {metrics_path}")

    synthetic_group_column = None
    if not group_columns:
        synthetic_group_column = "_summary_group"
        metrics_df[synthetic_group_column] = 0
        group_columns = [synthetic_group_column]

    grouped = metrics_df.groupby(group_columns, dropna=False)
    n_runs = grouped.size().rename("n_runs").reset_index()
    means = grouped[value_columns].mean().add_suffix("_mean").reset_index()
    standard_errors = (
        grouped[value_columns]
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
