#!/usr/bin/env python3

import argparse
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils import BM25Vectorizer


def read_fasta(sequences_path: str) -> dict[str, str]:
    """
    Read a protein sequence from a FASTA file
    """
    sequences = {}
    current_id = None
    chunks = []

    with open(sequences_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)

                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(chunks)

    return sequences


def check_input_data(
        protein_pairs: pd.DataFrame, sequences: dict[str, str]) -> None:
    """
    QC checks for the protein pairs and sequences. 
    """
    # Check for missing columns in pairs.csv
    required = {"protein_a", "protein_b", "label"}
    missing_cols = required - set(protein_pairs.columns)
    if missing_cols:
        raise ValueError(f"pairs.csv missing columns: {missing_cols}")

    # Check for missing sequences in FASTA
    proteins = set(protein_pairs["protein_a"]) |\
               set(protein_pairs["protein_b"])
    missing_sequences = sorted(proteins - set(sequences))
    if missing_sequences:
        raise ValueError(
            f"{len(missing_sequences)} proteins in pairs.csv are missing from FASTA. Examples: {missing_sequences[:10]}")
    



def make_vectorizer(args: argparse.Namespace) -> Any:
    """
    Create a vectorizer to extract features from protein sequences
    """
    if args.features == "tfidf":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(args.k, args.k),
            lowercase=False,
        )

    if args.features == "count":
        return CountVectorizer(
            analyzer="char",
            ngram_range=(args.k, args.k),
            lowercase=False,
        )

    if args.features == "binary":
        return CountVectorizer(
            analyzer="char",
            ngram_range=(args.k, args.k),
            lowercase=False,
            binary=True,
        )

    if args.features == "bm25":
        return BM25Vectorizer(
            k=args.k,
            k1=args.bm25_k1,
            b=args.bm25_b,
        )

    raise ValueError(f"Unknown feature type: {args.features}")


def make_pair_features(
        df: pd.DataFrame, sequences: dict[str, str], vectorizer: Any):
    """
    Create features for each protein pair.
    """
    # Get sequences for each protein in the pair NOTE: handle miss data
    sequence_a = df["protein_a"].map(sequences).tolist()
    sequence_b = df["protein_b"].map(sequences).tolist()

    # Extract features for each protein
    prot_a_fts = vectorizer.transform(sequence_a)
    prot_b_fts = vectorizer.transform(sequence_b)

    # Create protein pair features: sum, absolute difference, & product
    ft_sums = prot_a_fts + prot_b_fts
    ft_diffs = np.abs(prot_a_fts - prot_b_fts)
    ft_prods = prot_a_fts.multiply(prot_b_fts)
    feature_matrix = hstack([ft_sums, ft_diffs, ft_prods], format="csr")

    return feature_matrix


def make_classifier(args: argparse.Namespace) -> Any:
    """
    Create a classifier based on the specified type in args.
    """
    if args.classifier == "logistic":
        return LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=args.seed,
        )

    if args.classifier == "linear_svm":
        return LinearSVC(
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.seed,
        )

    if args.classifier == "sgd_logistic":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.seed,
        )

    raise ValueError(f"Unknown classifier: {args.classifier}")


def get_scores_and_predictions(model, x):
    """
    Returns:
      y_score: probability-like score for AUROC/AUPRC
      y_pred: hard 0/1 prediction
    """
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        return y_score, y_pred

    if hasattr(model, "decision_function"):
        y_score = model.decision_function(x)
        y_pred = (y_score >= 0.0).astype(int)
        return y_score, y_pred

    y_pred = model.predict(x)
    return y_pred.astype(float), y_pred


def get_metrics(
        y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray,
        split_name: str) -> dict[str, float]:
    """
    Return classificaiton metrics for the binary protein-protein interaction
    task.
    """
    # Precision
    if y_pred.sum() != 0:
        precision = y_true[y_pred == 1].sum() / y_pred.sum()
    else:
        precision = np.nan
    
    # Recall
    if y_true.sum() != 0:
        recall = y_pred[y_true == 1].sum() / y_true.sum()
    else:
        recall = np.nan

    # F1 score
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = np.nan
    
    # All metrics
    metrics = {
        f"accuracy ({split_name})": accuracy_score(y_true, y_pred),
        f"precision ({split_name})": precision,
        f"recall ({split_name})": recall,
        f"f1 ({split_name})": f1_score,
        f"auprc ({split_name})": average_precision_score(y_true, y_score),
    }
    try:
        metrics[f"auroc ({split_name})"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics[f"auroc ({split_name})"] = float("nan")

    return metrics


def load_or_make_split(pairs: pd.DataFrame, args):
    """
    Load the pre-defined train/test split of protein pairs
    OR
    Create a train/test split from the protein pairs
    """
    # Load the split
    if args.split_col and (args.split_col in pairs.columns):
        split_values = pairs[args.split_col].astype(str).str.lower()
        train_df = pairs[split_values == "train"].copy()
        test_df = pairs[split_values == "test"].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                "split-col must contain both 'train' and 'test' rows.")
    # Create a train/test split
    else:
        train_df, test_df = train_test_split(
            pairs,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=pairs["label"],
        )
    return train_df, test_df


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for the protein-protein interaction prediction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, 
                        help="CSV with protein_a, protein_b, label")
    parser.add_argument("--fasta", required=True, 
                        help="FASTA file with protein sequences")
    parser.add_argument("--pred-out", default="predictions.csv")
    parser.add_argument("--metrics-out", default="metrics.csv")

    # Args: Features
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument(
        "--features",
        choices=["tfidf", "bm25", "count", "binary"],
        default="tfidf",
        help="Protein sequence feature type",
    )
    feature_group.add_argument("--k", type=int, default=3, help="k-mer size")
    feature_group.add_argument("--bm25-k1", type=float, default=1.5)
    feature_group.add_argument("--bm25-b", type=float, default=0.75)
    
    # Args: Classifier
    parser.add_argument(
        "--classifier",
        choices=["logistic", "linear_svm", "sgd_logistic"],
        default="logistic",
        help="Classifier type",
    )

    # Args: Training
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-col", default=None, 
                        help="Optional column with train/test labels")

    args = parser.parse_args()
    
    return args


def main():

    # Parse CLI arguments
    args = argument_parser()

    """
    Load and process data
    """
    # Read input data
    protein_pairs = pd.read_csv(args.pairs)
    sequences = read_fasta(args.fasta)

    # QC checks
    check_input_data(protein_pairs, sequences)

    # Process data
    protein_pairs["label"] = protein_pairs["label"].astype(int)
    
    # Split into train/test # NOTE: add val_df
    train_df, test_df = load_or_make_split(protein_pairs, args) 
    train_proteins = sorted(
              set(train_df["protein_a"]) |\
              set(train_df["protein_b"]))
    train_sequences = [sequences[protein] for protein in train_proteins]

    # Prepare features
    vectorizer = make_vectorizer(args)
    vectorizer.fit(train_sequences)
    x_train = make_pair_features(train_df, sequences, vectorizer)
    x_test = make_pair_features(test_df, sequences, vectorizer)

    # Prepare labels
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    if len(set(y_train)) < 2:
        raise ValueError(
            "Training split has only 1 class. Needs 0 and 1 labels.")


    """
    Training
    """
    # Train the classifier
    model = make_classifier(args)
    model.fit(x_train, y_train)


    """
    Evaluation 
    """
    # Evaluate the classifier: training set
    y_train_score, y_train_pred = (
        get_scores_and_predictions(model, x_train))
    train_metrics = get_metrics(
        y_true=y_train, y_score=y_train_score, 
        y_pred=y_train_pred, split_name="train")

    # Evaluate the classifier: test set
    y_test_score, y_test_pred = (
        get_scores_and_predictions(model, x_test))
    test_metrics = get_metrics(
        y_true=y_test, y_score=y_test_score, 
        y_pred=y_test_pred, split_name="test")
    pred_test_df = test_df[["protein_a", "protein_b", "label"]].copy()
    pred_test_df["pred_score"] = y_test_score
    pred_test_df["pred_label"] = y_test_pred
    pred_test_df.to_csv(args.pred_out, index=False)

    # Evaluate: classification metrics
    metrics_df = pd.DataFrame([
        {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "features": args.features,
            "classifier": args.classifier,
            "k": args.k,
            **train_metrics,
            **test_metrics,
        }
    ])
    metrics_df.to_csv(args.metrics_out, index=False)
    print(
        f"{metrics_df.to_string(index=False)}"
        f"\nSaved predictions to: {args.pred_out}"
        f"\nSaved metrics to: {args.metrics_out}")

if __name__ == "__main__":
    main()