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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppi_features import FEATURE_CHOICES, build_feature_matrices
from ppi_inputs import (
    load_or_make_split,
    prepare_input_data,
    read_fasta,
    validate_train_test_splits,
)
from ppi_models import (
    BASELINE_CLASSIFIER_CHOICES,
    CLASSIFIER_CHOICES,
    get_metrics,
    get_scores_and_predictions,
    make_classifier,
)
from ppi_results import (
    append_dataframe,
    default_summary_path,
    reset_output_file,
    summarize_metrics,
    write_dataframe_threadsafe,
)

FEATURELESS_FEATURE = "none"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOGGER = logging.getLogger(__name__)


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
        "--pred-out", default="predictions.csv",
        help="CSV file for appended test-set predictions")
    output_group.add_argument(
        "--metrics-out", default="metrics.csv",
        help="CSV file for appended per-run metrics")
    output_group.add_argument(
        "--metrics-summary-out", default=None,
        help="CSV file for per-model metric means and standard errors. "
            "Defaults to <metrics-out stem>_summary.csv.")
    output_group.add_argument(
        "--append-results", action="store_true",
        help="Append to existing output files instead of starting fresh.")
    output_group.add_argument(
        "--execution-id", default=None,
        help="Optional identifier stored with each model run row.")
    output_group.add_argument(
        "--log-level", choices=LOG_LEVEL_CHOICES, default="INFO",
        help="Logging verbosity.")

    # Feature extraction args
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument(
        "--features", choices=FEATURE_CHOICES, nargs="+", default=["tfidf"],
        help="One or more protein sequence feature types")
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
    training_group.add_argument("--test-size", type=float, default=0.2)
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument(
        "--split-col", default=None,
        help="Optional column with train/test labels")
    training_group.add_argument(
        "--num-reruns", type=positive_int, default=1,
        help="Number of times to rerun each classifier with consecutive seeds")

    args = parser.parse_args()

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
def make_model_name(feature_type: str, classifier_name: str) -> str:
    """
    Return a stable, machine-readable model configuration name.
    """
    if feature_type == FEATURELESS_FEATURE:
        model_name = classifier_name
    else:
        model_name = f"{feature_type}__{classifier_name}"

    return model_name


def is_baseline_classifier(classifier_name: str) -> bool:
    """
    Return whether a classifier ignores feature matrices.
    """
    is_baseline = classifier_name in BASELINE_CLASSIFIER_CHOICES

    return is_baseline


def feature_metadata(
        feature_type: str, args: argparse.Namespace
    ) -> dict[str, float | int | str]:
    """
    Return feature metadata stored with each result row.
    """
    if feature_type == FEATURELESS_FEATURE:
        feature_metadata = {
            "features": FEATURELESS_FEATURE,
            "k": np.nan,
            "bm25_k1": np.nan,
            "bm25_b": np.nan,
        }
    else:
        feature_metadata = {
            "features": feature_type,
            "k": args.k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
        }

    return feature_metadata


def evaluate_model_run(
        train_df: pd.DataFrame, test_df: pd.DataFrame, x_train: Any,
        x_test: Any, feature_type: str, classifier_name: str, run_number: int,
        execution_id: str, args: argparse.Namespace
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate one model configuration for one run number.
    """
    # Define model metadata
    model_name = make_model_name(feature_type, classifier_name)
    is_baseline = is_baseline_classifier(classifier_name)
    run_seed = args.seed + run_number - 1
    model_seed = np.nan if is_baseline else run_seed
    max_iter = np.nan if is_baseline else args.max_iter
    feature_meta = feature_metadata(feature_type, args)

    # Define training and test labels
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    # Define model
    model = make_classifier(
        classifier_name=classifier_name,
        max_iter=args.max_iter,
        random_state=run_seed,
    )
    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Evaluate on the training data
    y_train_score, y_train_pred = get_scores_and_predictions(model, x_train)
    train_metrics = get_metrics(
        y_true=y_train,
        y_score=y_train_score,
        y_pred=y_train_pred,
        split_name="train",
    )

    # Evaluate on the test data
    y_test_score, y_test_pred = get_scores_and_predictions(model, x_test)
    test_metrics = get_metrics(
        y_true=y_test,
        y_score=y_test_score,
        y_pred=y_test_pred,
        split_name="test",
    )

    # Define the evaluate metrics table
    metrics_df = pd.DataFrame(
        [
            {
                "execution_id": execution_id,
                "model_name": model_name,
                "run_number": run_number,
                "model_seed": model_seed,
                "split_seed": args.seed,
                "n_train": len(train_df),
                "n_test": len(test_df),
                **feature_meta,
                "classifier": classifier_name,
                "max_iter": max_iter,
                "test_size": args.test_size,
                **train_metrics,
                **test_metrics,
            }
        ]
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

    return metrics_df, predictions_df


def run_model_reruns(
        train_df: pd.DataFrame, test_df: pd.DataFrame, x_train: Any,
        x_test: Any, feature_type: str, classifier_name: str,
        execution_id: str, args: argparse.Namespace, metrics_path: Path,
        predictions_path: Path,
    ) -> None:
    """
    Run one model configuration repeatedly and append each result.
    """
    model_name = make_model_name(feature_type, classifier_name)

    # Re-run the model
    for run_number in range(1, args.num_reruns + 1):
        metrics_df, predictions_df = evaluate_model_run(
            train_df=train_df,
            test_df=test_df,
            x_train=x_train,
            x_test=x_test,
            feature_type=feature_type,
            classifier_name=classifier_name,
            run_number=run_number,
            execution_id=execution_id,
            args=args,
        )
        append_dataframe(metrics_df, metrics_path)
        append_dataframe(predictions_df, predictions_path)
        LOGGER.info(
            "Finished model=%s run=%s/%s",
            model_name,
            run_number,
            args.num_reruns,
        )


def prepare_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """
    Resolve and initialize output paths.
    """
    # Define output paths
    predictions_path = Path(args.pred_out)
    metrics_path = Path(args.metrics_out)
    summary_path = (
        Path(args.metrics_summary_out)
        if args.metrics_summary_out
        else default_summary_path(metrics_path)
    )

    # Prepare to write to output paths
    reset_output_file(predictions_path, append_results=args.append_results)
    reset_output_file(metrics_path, append_results=args.append_results)
    reset_output_file(summary_path, append_results=False)

    return predictions_path, metrics_path, summary_path


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

    # Split into train/test sets
    train_df, test_df = load_or_make_split(protein_pairs, args)
    validate_train_test_splits(train_df, test_df)

    # Prepare to run the models
    execution_id = args.execution_id or uuid.uuid4().hex
    predictions_path, metrics_path, summary_path = prepare_outputs(args)

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
            test_df=test_df,
            x_train=train_df,
            x_test=test_df,
            feature_type=FEATURELESS_FEATURE,
            classifier_name=classifier_name,
            execution_id=execution_id,
            args=args,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )

    # Extract features
    for feature_type in args.features:
        if not learned_classifiers:
            continue
        x_train, x_test = build_feature_matrices(
            train_df=train_df,
            test_df=test_df,
            sequences=sequences,
            feature_type=feature_type,
            args=args,
        )

        # Run the learned classifiers
        for classifier_name in args.classifiers:
            if classifier_name not in learned_classifiers:
                continue
            run_model_reruns(
                train_df=train_df,
                test_df=test_df,
                x_train=x_train,
                x_test=x_test,
                feature_type=feature_type,
                classifier_name=classifier_name,
                execution_id=execution_id,
                args=args,
                metrics_path=metrics_path,
                predictions_path=predictions_path,
            )

    # Summarize model performance
    summary_df = summarize_metrics(metrics_path)
    write_dataframe_threadsafe(summary_df, summary_path)

    LOGGER.info(
        "\n%s\nSaved predictions to: %s"
        "\nSaved per-run metrics to: %s"
        "\nSaved metric summary to: %s",
        summary_df.to_string(index=False),
        predictions_path,
        metrics_path,
        summary_path,
    )


if __name__ == "__main__":
    main()
