#!/usr/bin/env python3

"""
Command-line entry point for training and evaluating PPI predictors.

This script owns CLI parsing and high-level orchestration. Domain-specific
input validation, feature extraction, model evaluation, and result writing live
in focused helper modules so future pipeline variants can reuse them.
"""

import argparse
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppi_features import (
    FEATURE_CHOICES,
    build_feature_matrices,
    make_feature_name,
    normalize_feature_types,
)
from ppi_inputs import (
    PROVIDED_SPLIT_STRATEGY,
    load_or_make_split,
    prepare_input_data,
    read_fasta,
    RANDOM_SPLIT_STRATEGY,
    SPLIT_STRATEGY_CHOICES,
    validate_splits,
)
from ppi_models import (
    BASELINE_CLASSIFIER_CHOICES,
    CLASSIFIER_CHOICES,
    get_metrics,
    get_scores_and_predictions,
    make_classifier,
)
from ppi_plots import (
    legacy_f1_heatmap_output_paths,
    plot_metrics_summary,
    plot_train_test_f1_heatmap,
    plot_train_test_metrics_summary,
    plot_train_test_metrics_summary_png,
)
from ppi_results import (
    append_dataframe,
    reset_output_file,
    summarize_metrics,
    write_dataframe_threadsafe,
)

FEATURELESS_FEATURE = "none"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOGGER = logging.getLogger(__name__)
PLOTS_DIRNAME = "plots"
PREDICTIONS_FILENAME = "predictions.csv"
TRAIN_METRICS_FILENAME = "train_metrics.csv"
VAL_METRICS_FILENAME = "val_metrics.csv"
TEST_METRICS_FILENAME = "test_metrics.csv"
TRAIN_SUMMARY_FILENAME = "train_metrics_summary.csv"
VAL_SUMMARY_FILENAME = "val_metrics_summary.csv"
TEST_SUMMARY_FILENAME = "test_metrics_summary.csv"
TRAIN_PLOT_FILENAME = "train_metrics_summary.svg"
VAL_PLOT_FILENAME = "val_metrics_summary.svg"
TEST_PLOT_FILENAME = "test_metrics_summary.svg"
TRAIN_VAL_PLOT_FILENAME = "train_val_metrics_summary.svg"
TRAIN_VAL_PNG_FILENAME = "train_val_metrics_summary.png"
TRAIN_VAL_F1_HEATMAP_FILENAME = "train_val_f1_heatmap.png"
TRAIN_TEST_PLOT_FILENAME = "train_test_metrics_summary.svg"
TRAIN_TEST_PNG_FILENAME = "train_test_metrics_summary.png"
TRAIN_TEST_F1_HEATMAP_FILENAME = "train_test_f1_heatmap.png"


#######
# CLI #
#######
def positive_int(value: str) -> int:
    """
    Parse a positive integer argparse value.
    """
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def proportion(value: str) -> float:
    """
    Parse a float proportion between 0 and 1.
    """
    parsed_value = float(value)
    if (parsed_value <= 0.0) or (parsed_value >= 1.0):
        raise argparse.ArgumentTypeError("value must be between 0 and 1")

    return parsed_value


def nonnegative_proportion(value: str) -> float:
    """
    Parse a float proportion between 0 and 1, inclusive of 0 only.
    """
    parsed_value = float(value)
    if (parsed_value < 0.0) or (parsed_value >= 1.0):
        raise argparse.ArgumentTypeError("value must be at least 0 and less than 1")

    return parsed_value


def non_empty_string(value: str) -> str:
    """
    Parse a non-empty argparse string after trimming whitespace.
    """
    parsed_value = value.strip()
    if not parsed_value:
        raise argparse.ArgumentTypeError("value must not be empty")

    return parsed_value


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for protein-protein interaction prediction.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate protein-protein interaction classifiers over "
            "one or more feature/model configurations."))

    # Input data args
    input_group = parser.add_argument_group("Input data")
    input_group.add_argument(
        "--pairs", required=True,
        help="CSV with columns for protein_a, protein_b, label")
    input_group.add_argument(
        "--fasta", required=True,
        help="FASTA file of protein sequences")

    # Output data args
    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument(
        "--run-dir", required=True,
        help="Directory where canonical metrics, predictions, and plots are "
            "written.")
    output_group.add_argument(
        "--append-results", action="store_true",
        help="Append to existing metric/prediction CSVs in --run-dir instead "
            "of starting fresh.")
    output_group.add_argument(
        "--execution-id", default=None,
        help="Optional identifier stored with each model run row.")
    output_group.add_argument(
        "--log-level", choices=LOG_LEVEL_CHOICES, default="INFO",
        help="Logging verbosity.")

    # Plot args
    plot_group = parser.add_argument_group("Plots")
    plot_group.add_argument(
        "--no-metrics-plots", action="store_true",
        help="Do not create canonical metrics summary plots under "
            "--run-dir/plots.")

    # Feature extraction args
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument(
        "--features", choices=FEATURE_CHOICES, nargs="+", default=["tfidf"],
        help="One or more feature types to concatenate")
    feature_group.add_argument("--k", type=int, default=3, help="k-mer size")
    feature_group.add_argument("--bm25-k1", type=float, default=1.5)
    feature_group.add_argument("--bm25-b", type=float, default=0.75)

    # Model args
    model_group = parser.add_argument_group("Models")
    model_group.add_argument(
        "--classifier", "--classifiers", dest="classifiers",
        choices=CLASSIFIER_CHOICES, nargs="+", default=["logistic"],
        help="One or more classifier model types")

    # Training args
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--max-iter", type=int, default=1000)
    training_group.add_argument(
        "--train-size", type=proportion, default=0.80,
        help="Fraction of pairs assigned to the training set")
    training_group.add_argument(
        "--val-size", type=nonnegative_proportion, default=0.0,
        help="Fraction of pairs assigned to the validation set")
    training_group.add_argument(
        "--eval-test-set", action="store_true",
        help="Evaluate on the test split. In validation runs, omit this to "
            "keep test held out.")
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument(
        "--split-col", default=None,
        help="Optional column with train, validation, and test labels")
    training_group.add_argument(
        "--split-name", type=non_empty_string, default=None,
        help="Optional display name for a provided --split-col split")
    training_group.add_argument(
        "--split-strategy", choices=SPLIT_STRATEGY_CHOICES, default=None,
        help="Optional strategy for creating train/validation/test splits")
    training_group.add_argument(
        "--num-reruns", type=positive_int, default=1,
        help="Number of times to rerun each classifier with consecutive seeds")

    args = parser.parse_args()
    if args.split_col and args.split_strategy:
        parser.error("--split-col and --split-strategy cannot both be set.")
    if args.split_name and not args.split_col:
        parser.error("--split-name can only be used with --split-col.")
    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be less than 1.")

    if args.split_col:
        args.effective_split_strategy = PROVIDED_SPLIT_STRATEGY
    else:
        args.effective_split_strategy = (
            args.split_strategy or RANDOM_SPLIT_STRATEGY)
    args.n_connected_components = None
    args.n_pruned_pairs = 0
    args.pruned_pair_fraction = 0.0

    try:
        args.features = normalize_feature_types(args.features)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def configure_logging(args: argparse.Namespace) -> None:
    """
    Configure command-line logging.
    """
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )


####################
# Pipeline helpers #
####################
@dataclass(frozen=True)
class OutputPaths:
    """
    Pipeline output paths grouped by artifact type.
    """
    run_dir: Path
    plots_dir: Path
    predictions_path: Path | None
    train_metrics_path: Path
    val_metrics_path: Path | None
    test_metrics_path: Path | None
    train_summary_path: Path
    val_summary_path: Path | None
    test_summary_path: Path | None
    train_plot_path: Path | None
    val_plot_path: Path | None
    test_plot_path: Path | None
    train_val_plot_path: Path | None
    train_val_png_path: Path | None
    train_val_f1_heatmap_path: Path | None
    train_test_plot_path: Path | None
    train_test_png_path: Path | None
    train_test_f1_heatmap_path: Path | None


def make_model_name(feature_name: str, classifier_name: str) -> str:
    """
    Return a stable, machine-readable model configuration name.
    """
    if feature_name == FEATURELESS_FEATURE:
        model_name = classifier_name
    else:
        model_name = f"{feature_name}__{classifier_name}"

    return model_name


def is_baseline_classifier(classifier_name: str) -> bool:
    """
    Return whether a classifier ignores feature matrices.
    """
    is_baseline = classifier_name in BASELINE_CLASSIFIER_CHOICES

    return is_baseline


def feature_metadata(
        feature_name: str, args: argparse.Namespace
    ) -> dict[str, float | int | str]:
    """
    Return feature metadata stored with each result row.
    """
    if feature_name == FEATURELESS_FEATURE:
        feature_metadata = {
            "features": FEATURELESS_FEATURE,
            "k": np.nan,
            "bm25_k1": np.nan,
            "bm25_b": np.nan,
        }
    else:
        feature_metadata = {
            "features": feature_name,
            "k": args.k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
        }

    return feature_metadata


def make_metrics_df(
        metadata: dict[str, Any], metrics: dict[str, float], split_name: str,
    ) -> pd.DataFrame:
    """
    Return one split-specific metrics row.
    """
    metrics_df = pd.DataFrame(
        [
            {
                **metadata,
                "split": split_name,
                **metrics,
            }
        ]
    )

    return metrics_df


def train_and_evaluate_model_run(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, x_train: Any, x_val: Any | None,
        x_test: Any | None, feature_name: str, classifier_name: str,
        run_number: int,
        execution_id: str, args: argparse.Namespace
    ) -> tuple[pd.DataFrame, pd.DataFrame | None,
               pd.DataFrame | None, pd.DataFrame | None]:
    """
    Train and evaluate one model configuration for one run number.
    """
    # Define model metadata
    model_name = make_model_name(feature_name, classifier_name)
    is_baseline = is_baseline_classifier(classifier_name)
    run_seed = args.seed + run_number - 1
    model_seed = np.nan if is_baseline else run_seed
    max_iter = np.nan if is_baseline else args.max_iter
    feature_meta = feature_metadata(feature_name, args)

    # Define training labels
    y_train = train_df["label"].to_numpy()

    # Define model
    model = make_classifier(
        classifier_name=classifier_name,
        max_iter=args.max_iter,
        random_state=run_seed,
    )
    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Define the evaluate metrics tables
    n_val = 0 if val_df is None else len(val_df)
    n_total = len(train_df) + n_val + len(test_df)
    actual_train_size = len(train_df) / n_total
    actual_val_size = n_val / n_total
    actual_test_size = len(test_df) / n_total
    metrics_metadata = {
        "execution_id": execution_id,
        "model_name": model_name,
        "run_number": run_number,
        "model_seed": model_seed,
        "split_seed": args.seed,
        "n_train": len(train_df),
        "n_val": n_val,
        "n_test": len(test_df),
        **feature_meta,
        "classifier": classifier_name,
        "max_iter": max_iter,
        "split_strategy": args.effective_split_strategy,
        "split_name": args.split_name,
        "target_train_size": args.train_size,
        "target_val_size": args.val_size,
        "actual_train_size": actual_train_size,
        "actual_val_size": actual_val_size,
        "actual_test_size": actual_test_size,
        "n_connected_components": args.n_connected_components,
        "n_pruned_pairs": args.n_pruned_pairs,
        "pruned_pair_fraction": args.pruned_pair_fraction,
    }

    def evaluate_split(
            split_df: pd.DataFrame, x_split: Any, split_name: str,
        ) -> tuple[pd.DataFrame, Any, Any]:
        """
        Evaluate one split and return metrics plus raw predictions.
        """
        y_true = split_df["label"].to_numpy()
        y_score, y_pred = get_scores_and_predictions(model, x_split)
        metrics = get_metrics(
            y_true=y_true,
            y_score=y_score,
            y_pred=y_pred,
        )
        metrics_df = make_metrics_df(
            metadata=metrics_metadata,
            metrics=metrics,
            split_name=split_name,
        )

        return metrics_df, y_score, y_pred

    train_metrics_df, _, _ = evaluate_split(train_df, x_train, "train")
    val_metrics_df = None
    if val_df is not None and x_val is not None:
        val_metrics_df, _, _ = evaluate_split(val_df, x_val, "val")

    test_metrics_df = None
    predictions_df = None
    if args.evaluate_test_metrics:
        if x_test is None:
            raise ValueError("Test evaluation requested without test features.")
        test_metrics_df, y_test_score, y_test_pred = evaluate_split(
            test_df,
            x_test,
            "test",
        )

        # Define the predictions table
        predictions_df = test_df[["protein_a", "protein_b", "label"]].copy()
        predictions_df.insert(0, "execution_id", execution_id)
        predictions_df.insert(1, "model_name", model_name)
        predictions_df.insert(2, "run_number", run_number)
        predictions_df.insert(3, "model_seed", model_seed)
        predictions_df.insert(4, "features", feature_meta["features"])
        predictions_df.insert(5, "classifier", classifier_name)
        predictions_df.insert(6, "k", feature_meta["k"])
        predictions_df["pred_score"] = y_test_score
        predictions_df["pred_label"] = y_test_pred

    return train_metrics_df, val_metrics_df, test_metrics_df, predictions_df


def run_model_reruns(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, x_train: Any, x_val: Any | None,
        x_test: Any | None, feature_name: str, classifier_name: str,
        execution_id: str, args: argparse.Namespace,
        output_paths: OutputPaths,
    ) -> None:
    """
    Run one model configuration repeatedly and append each result.
    """
    model_name = make_model_name(feature_name, classifier_name)

    # Re-run the model
    for run_number in range(1, args.num_reruns + 1):
        (
            train_metrics_df,
            val_metrics_df,
            test_metrics_df,
            predictions_df,
        ) = (
            train_and_evaluate_model_run(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                feature_name=feature_name,
                classifier_name=classifier_name,
                run_number=run_number,
                execution_id=execution_id,
                args=args,
            )
        )
        # Save performance and prediction results
        append_dataframe(train_metrics_df, output_paths.train_metrics_path)
        if val_metrics_df is not None and output_paths.val_metrics_path is not None:
            append_dataframe(val_metrics_df, output_paths.val_metrics_path)
        if test_metrics_df is not None and output_paths.test_metrics_path is not None:
            append_dataframe(test_metrics_df, output_paths.test_metrics_path)
        if predictions_df is not None and output_paths.predictions_path is not None:
            append_dataframe(predictions_df, output_paths.predictions_path)
        LOGGER.info(
            f"Finished model={model_name} "
            f"run={run_number}/{args.num_reruns}"
        )


def prepare_outputs(args: argparse.Namespace) -> OutputPaths:
    """
    Resolve and initialize canonical output paths inside --run-dir.
    """
    run_dir = Path(args.run_dir)
    plots_dir = run_dir / PLOTS_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_predictions_path = run_dir / PREDICTIONS_FILENAME
    train_metrics_path = run_dir / TRAIN_METRICS_FILENAME
    all_val_metrics_path = run_dir / VAL_METRICS_FILENAME
    all_test_metrics_path = run_dir / TEST_METRICS_FILENAME
    train_summary_path = run_dir / TRAIN_SUMMARY_FILENAME
    all_val_summary_path = run_dir / VAL_SUMMARY_FILENAME
    all_test_summary_path = run_dir / TEST_SUMMARY_FILENAME
    all_train_plot_path = plots_dir / TRAIN_PLOT_FILENAME
    all_val_plot_path = plots_dir / VAL_PLOT_FILENAME
    all_test_plot_path = plots_dir / TEST_PLOT_FILENAME
    all_train_val_plot_path = plots_dir / TRAIN_VAL_PLOT_FILENAME
    all_train_val_png_path = plots_dir / TRAIN_VAL_PNG_FILENAME
    all_train_val_f1_heatmap_path = plots_dir / TRAIN_VAL_F1_HEATMAP_FILENAME
    all_train_test_plot_path = plots_dir / TRAIN_TEST_PLOT_FILENAME
    all_train_test_png_path = plots_dir / TRAIN_TEST_PNG_FILENAME
    all_train_test_f1_heatmap_path = (
        plots_dir / TRAIN_TEST_F1_HEATMAP_FILENAME)

    predictions_path = (
        all_predictions_path if args.evaluate_test_metrics else None)
    val_metrics_path = (
        all_val_metrics_path if args.has_validation_split else None)
    test_metrics_path = (
        all_test_metrics_path if args.evaluate_test_metrics else None)
    val_summary_path = (
        all_val_summary_path if args.has_validation_split else None)
    test_summary_path = (
        all_test_summary_path if args.evaluate_test_metrics else None)
    train_plot_path = all_train_plot_path if not args.no_metrics_plots else None
    val_plot_path = None
    test_plot_path = None
    train_val_plot_path = None
    train_val_png_path = None
    train_val_f1_heatmap_path = None
    train_test_plot_path = None
    train_test_png_path = None
    train_test_f1_heatmap_path = None
    if not args.no_metrics_plots:
        if args.has_validation_split:
            val_plot_path = all_val_plot_path
            train_val_plot_path = all_train_val_plot_path
            train_val_png_path = all_train_val_png_path
            train_val_f1_heatmap_path = all_train_val_f1_heatmap_path
        if args.evaluate_test_metrics:
            test_plot_path = all_test_plot_path
            train_test_plot_path = all_train_test_plot_path
            train_test_png_path = all_train_test_png_path
            train_test_f1_heatmap_path = all_train_test_f1_heatmap_path

    # Clear stale canonical files on fresh runs. In append mode, preserve
    # metric/prediction CSVs but always regenerate summaries and plots.
    all_metric_prediction_paths = [
        all_predictions_path,
        train_metrics_path,
        all_val_metrics_path,
        all_test_metrics_path,
    ]
    all_summary_plot_paths = [
        train_summary_path,
        all_val_summary_path,
        all_test_summary_path,
        all_train_plot_path,
        all_val_plot_path,
        all_test_plot_path,
        all_train_val_plot_path,
        all_train_val_png_path,
        all_train_val_f1_heatmap_path,
        all_train_test_plot_path,
        all_train_test_png_path,
        all_train_test_f1_heatmap_path,
    ]
    legacy_heatmap_paths = [
        *legacy_f1_heatmap_output_paths(
            all_train_val_f1_heatmap_path,
            comparison_split_name="val",
        ).values(),
        *legacy_f1_heatmap_output_paths(
            all_train_test_f1_heatmap_path,
            comparison_split_name="test",
        ).values(),
    ]
    if not args.append_results:
        for output_path in (
                all_metric_prediction_paths
                + all_summary_plot_paths
                + legacy_heatmap_paths):
            reset_output_file(output_path, append_results=False)

    reset_output_file(train_metrics_path, append_results=args.append_results)
    if val_metrics_path is not None:
        reset_output_file(val_metrics_path, append_results=args.append_results)
    if predictions_path is not None:
        reset_output_file(predictions_path, append_results=args.append_results)
    if test_metrics_path is not None:
        reset_output_file(test_metrics_path, append_results=args.append_results)
    reset_output_file(train_summary_path, append_results=False)
    if val_summary_path is not None:
        reset_output_file(val_summary_path, append_results=False)
    if test_summary_path is not None:
        reset_output_file(test_summary_path, append_results=False)
    if train_plot_path is not None:
        reset_output_file(train_plot_path, append_results=False)
    if val_plot_path is not None:
        reset_output_file(val_plot_path, append_results=False)
    if test_plot_path is not None:
        reset_output_file(test_plot_path, append_results=False)
    if train_val_plot_path is not None:
        reset_output_file(train_val_plot_path, append_results=False)
    if train_val_png_path is not None:
        reset_output_file(train_val_png_path, append_results=False)
    if train_val_f1_heatmap_path is not None:
        reset_output_file(train_val_f1_heatmap_path, append_results=False)
    if train_test_plot_path is not None:
        reset_output_file(train_test_plot_path, append_results=False)
    if train_test_png_path is not None:
        reset_output_file(train_test_png_path, append_results=False)
    if train_test_f1_heatmap_path is not None:
        reset_output_file(train_test_f1_heatmap_path, append_results=False)

    output_paths = OutputPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        predictions_path=predictions_path,
        train_metrics_path=train_metrics_path,
        val_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        train_summary_path=train_summary_path,
        val_summary_path=val_summary_path,
        test_summary_path=test_summary_path,
        train_plot_path=train_plot_path,
        val_plot_path=val_plot_path,
        test_plot_path=test_plot_path,
        train_val_plot_path=train_val_plot_path,
        train_val_png_path=train_val_png_path,
        train_val_f1_heatmap_path=train_val_f1_heatmap_path,
        train_test_plot_path=train_test_plot_path,
        train_test_png_path=train_test_png_path,
        train_test_f1_heatmap_path=train_test_f1_heatmap_path,
    )

    return output_paths


def main() -> None:
    """
    Run the full CLI pipeline.
    """
    args = argument_parser()
    configure_logging(args)

    # Load and process input data
    protein_pairs = pd.read_csv(args.pairs)
    sequences = read_fasta(args.fasta)
    protein_pairs = prepare_input_data(protein_pairs, sequences)

    # Split into train/validation/test sets
    train_df, val_df, test_df = load_or_make_split(protein_pairs, args)
    args.has_validation_split = val_df is not None and not val_df.empty
    args.evaluate_test_metrics = (
        args.eval_test_set or not args.has_validation_split)
    validate_splits(train_df=train_df, val_df=val_df, test_df=test_df)

    # Prepare to run the models
    execution_id = args.execution_id or uuid.uuid4().hex
    output_paths = prepare_outputs(args)

    # Define the model configurations to run
    baseline_classifiers = [
        classifier_name
        for classifier_name in args.classifiers
        if is_baseline_classifier(classifier_name)
    ]
    learned_classifiers = [
        classifier_name
        for classifier_name in args.classifiers
        if not is_baseline_classifier(classifier_name)
    ]

    # Run the baseline models
    for classifier_name in baseline_classifiers:
        run_model_reruns(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            x_train=train_df,
            x_val=val_df,
            x_test=test_df if args.evaluate_test_metrics else None,
            feature_name=FEATURELESS_FEATURE,
            classifier_name=classifier_name,
            execution_id=execution_id,
            args=args,
            output_paths=output_paths,
        )

    # Extract the combined feature set
    if learned_classifiers:
        feature_name = make_feature_name(args.features)
        x_train, x_val, x_test = build_feature_matrices(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df if args.evaluate_test_metrics else None,
            sequences=sequences,
            feature_types=args.features,
            args=args,
        )

    # Run the learned classifiers
    for classifier_name in learned_classifiers:
        run_model_reruns(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            feature_name=feature_name,
            classifier_name=classifier_name,
            execution_id=execution_id,
            args=args,
            output_paths=output_paths,
        )

    # Summarize model performance
    train_summary_df = summarize_metrics(output_paths.train_metrics_path)
    val_summary_df = (
        summarize_metrics(output_paths.val_metrics_path)
        if output_paths.val_metrics_path is not None
        else None
    )
    test_summary_df = (
        summarize_metrics(output_paths.test_metrics_path)
        if output_paths.test_metrics_path is not None
        else None
    )
    write_dataframe_threadsafe(
        train_summary_df,
        output_paths.train_summary_path,
    )
    if val_summary_df is not None and output_paths.val_summary_path is not None:
        write_dataframe_threadsafe(
            val_summary_df,
            output_paths.val_summary_path,
        )
    if test_summary_df is not None and output_paths.test_summary_path is not None:
        write_dataframe_threadsafe(
            test_summary_df,
            output_paths.test_summary_path,
        )
    if output_paths.train_plot_path is not None:
        plot_metrics_summary(
            summary_path=output_paths.train_summary_path,
            plot_path=output_paths.train_plot_path,
            split_name="train",
        )
    if output_paths.val_plot_path is not None:
        plot_metrics_summary(
            summary_path=output_paths.val_summary_path,
            plot_path=output_paths.val_plot_path,
            split_name="val",
        )
    if output_paths.test_plot_path is not None:
        plot_metrics_summary(
            summary_path=output_paths.test_summary_path,
            plot_path=output_paths.test_plot_path,
            split_name="test",
        )
    train_val_f1_heatmap_paths = {}
    if output_paths.train_val_plot_path is not None:
        plot_train_test_metrics_summary(
            train_summary_path=output_paths.train_summary_path,
            test_summary_path=output_paths.val_summary_path,
            plot_path=output_paths.train_val_plot_path,
            comparison_split_name="val",
        )
    if output_paths.train_val_png_path is not None:
        plot_train_test_metrics_summary_png(
            train_summary_path=output_paths.train_summary_path,
            test_summary_path=output_paths.val_summary_path,
            plot_path=output_paths.train_val_png_path,
            comparison_split_name="val",
        )
    if output_paths.train_val_f1_heatmap_path is not None:
        train_val_f1_heatmap_paths = plot_train_test_f1_heatmap(
            train_summary_path=output_paths.train_summary_path,
            test_summary_path=output_paths.val_summary_path,
            plot_path=output_paths.train_val_f1_heatmap_path,
            comparison_split_name="val",
        )
    if output_paths.train_test_plot_path is not None:
        plot_train_test_metrics_summary(
            train_summary_path=output_paths.train_summary_path,
            test_summary_path=output_paths.test_summary_path,
            plot_path=output_paths.train_test_plot_path,
            comparison_split_name="test",
        )
    if output_paths.train_test_png_path is not None:
        plot_train_test_metrics_summary_png(
            train_summary_path=output_paths.train_summary_path,
            test_summary_path=output_paths.test_summary_path,
            plot_path=output_paths.train_test_png_path,
            comparison_split_name="test",
        )
    train_test_f1_heatmap_paths = {}
    if output_paths.train_test_f1_heatmap_path is not None:
        train_test_f1_heatmap_paths = plot_train_test_f1_heatmap(
            train_summary_path=output_paths.train_summary_path,
            test_summary_path=output_paths.test_summary_path,
            plot_path=output_paths.train_test_f1_heatmap_path,
            comparison_split_name="test",
        )

    log_message = (
        f"\nTrain metric summary\n{train_summary_df.to_string(index=False)}"
        f"\nSaved train per-run metrics to: "
        f"{output_paths.train_metrics_path}"
        f"\nSaved train metric summary to: "
        f"{output_paths.train_summary_path}"
    )
    if val_summary_df is not None:
        log_message = (
            f"{log_message}\n\nValidation metric summary\n"
            f"{val_summary_df.to_string(index=False)}"
            f"\nSaved validation per-run metrics to: "
            f"{output_paths.val_metrics_path}"
            f"\nSaved validation metric summary to: "
            f"{output_paths.val_summary_path}"
        )
    if test_summary_df is not None:
        log_message = (
            f"{log_message}\n\nTest metric summary\n"
            f"{test_summary_df.to_string(index=False)}"
            f"\nSaved predictions to: {output_paths.predictions_path}"
            f"\nSaved test per-run metrics to: {output_paths.test_metrics_path}"
            f"\nSaved test metric summary to: {output_paths.test_summary_path}"
        )
    if output_paths.train_plot_path is not None:
        log_message = (
            f"{log_message}\nSaved train metric plot to: "
            f"{output_paths.train_plot_path}"
        )
    if output_paths.val_plot_path is not None:
        log_message = (
            f"{log_message}\nSaved validation metric plot to: "
            f"{output_paths.val_plot_path}"
        )
    if output_paths.test_plot_path is not None:
        log_message = (
            f"{log_message}\nSaved test metric plot to: "
            f"{output_paths.test_plot_path}"
        )
    if output_paths.train_val_plot_path is not None:
        log_message = (
            f"{log_message}\nSaved train/validation metric plot to: "
            f"{output_paths.train_val_plot_path}"
        )
    if output_paths.train_val_png_path is not None:
        log_message = (
            f"{log_message}\nSaved train/validation metric PNG to: "
            f"{output_paths.train_val_png_path}"
        )
    if train_val_f1_heatmap_paths:
        heatmap_paths = ", ".join(
            str(path) for path in train_val_f1_heatmap_paths.values())
        log_message = (
            f"{log_message}\nSaved train/validation F1 heatmap stack to: "
            f"{heatmap_paths}"
        )
    if output_paths.train_test_plot_path is not None:
        log_message = (
            f"{log_message}\nSaved train/test metric plot to: "
            f"{output_paths.train_test_plot_path}"
        )
    if output_paths.train_test_png_path is not None:
        log_message = (
            f"{log_message}\nSaved train/test metric PNG to: "
            f"{output_paths.train_test_png_path}"
        )
    if train_test_f1_heatmap_paths:
        heatmap_paths = ", ".join(
            str(path) for path in train_test_f1_heatmap_paths.values())
        log_message = (
            f"{log_message}\nSaved train/test F1 heatmap stack to: "
            f"{heatmap_paths}"
        )

    LOGGER.info(log_message)


if __name__ == "__main__":
    main()
