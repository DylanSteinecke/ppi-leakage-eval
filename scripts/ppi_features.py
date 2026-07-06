"""
Feature extraction utilities for PPI prediction.

This module owns sequence vectorizers and pair featurization. Future additions
should include embedding-based features, cached feature matrices, protein-level
descriptors, and additional pair-composition strategies.
"""

import argparse
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

FEATURE_CHOICES = ("tfidf", "bm25", "count", "binary")


###################
# BM25 vectorizer #
###################
class BM25Vectorizer:
    """
    Minimal BM25 vectorizer for protein k-mers.

    The API matches sklearn vectorizers with fit() and transform().
    """

    def __init__(self, k: int = 3, k1: float = 1.5, b: float = 0.75):
        self.k = k
        self.k1 = k1
        self.b = b
        self.count_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
        )
        self.idf_ = None
        self.avgdl_ = None

    def fit(self, sequences: list[str]) -> "BM25Vectorizer":
        """
        Fit BM25 inverse document frequency and average length statistics.
        """
        counts = self.count_vectorizer.fit_transform(sequences).tocsr()
        n_docs = counts.shape[0]

        doc_frequency = np.diff(counts.tocsc().indptr)

        self.idf_ = np.log(
            1.0 + (n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)
        )
        self.avgdl_ = float(np.asarray(counts.sum(axis=1)).ravel().mean())
        self.avgdl_ = max(self.avgdl_, 1e-12)

        return self

    def transform(self, sequences: list[str]) -> csr_matrix:
        """
        Transform sequences into BM25-weighted k-mer features.
        """
        if (self.idf_ is None) or (self.avgdl_ is None):
            raise RuntimeError(
                "BM25Vectorizer must be fit before transform."
            )

        counts = (
            self.count_vectorizer.transform(sequences)
            .tocsr()
            .astype(np.float64)
        )
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()

        coo = counts.tocoo()
        tf = coo.data
        rows = coo.row
        cols = coo.col

        denom = tf + self.k1 * (
            1.0 - self.b + self.b * doc_lengths[rows] / self.avgdl_
        )
        data = self.idf_[cols] * ((tf * (self.k1 + 1.0)) / denom)
        features = csr_matrix((data, (rows, cols)), shape=counts.shape)

        return features


#######################
# Feature vectorizers #
#######################
def make_vectorizer(feature_type: str, args: argparse.Namespace) -> Any:
    """
    Create a vectorizer to extract features from protein sequences.
    """
    # TF-IDF
    if feature_type == "tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(args.k, args.k),
            lowercase=False,
        )

    # Count-based k-mer features
    elif feature_type == "count":
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(args.k, args.k),
            lowercase=False,
        )

    # Binary k-mer features
    elif feature_type == "binary":
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(args.k, args.k),
            lowercase=False,
            binary=True,
        )

    # BM25 k-mer features
    elif feature_type == "bm25":
        vectorizer = BM25Vectorizer(
            k=args.k,
            k1=args.bm25_k1,
            b=args.bm25_b,
        )

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return vectorizer


#################
# Pair features #
#################
def make_pair_features(
        df: pd.DataFrame, sequences: dict[str, str], vectorizer: Any,
    ) -> Any:
    """
    Create features for each protein pair.
    """
    # Protein sequences
    sequence_a = df["protein_a"].map(sequences).tolist()
    sequence_b = df["protein_b"].map(sequences).tolist()

    # Protein features
    prot_a_fts = vectorizer.transform(sequence_a)
    prot_b_fts = vectorizer.transform(sequence_b)

    # Protein pair features
    ft_sums = prot_a_fts + prot_b_fts
    ft_diffs = np.abs(prot_a_fts - prot_b_fts)
    ft_prods = prot_a_fts.multiply(prot_b_fts)
    prot_pair_fts = hstack([ft_sums, ft_diffs, ft_prods], format="csr")

    return prot_pair_fts


def build_feature_matrices(
        train_df: pd.DataFrame, test_df: pd.DataFrame,
        sequences: dict[str, str], feature_type: str, args: argparse.Namespace
    ) -> tuple[Any, Any]:
    """
    Fit the requested feature extractor on train proteins and transform pairs.
    """
    # Define and extract features based on the training set
    train_proteins = sorted(
        set(train_df["protein_a"]) | set(train_df["protein_b"]))
    train_sequences = [sequences[protein] for protein in train_proteins]
    vectorizer = make_vectorizer(feature_type, args)
    vectorizer.fit(train_sequences)

    # Make train and test features
    x_train = make_pair_features(train_df, sequences, vectorizer)
    x_test = make_pair_features(test_df, sequences, vectorizer)

    return x_train, x_test
