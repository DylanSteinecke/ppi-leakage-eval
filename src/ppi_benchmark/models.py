"""
Model construction, scoring, and metrics for PPI prediction.

This module owns sklearn classifier factories, baseline estimators, score
extraction, and metric calculation. Framework adapters live in ``backends``.
Future additions should include new sklearn models, calibration, threshold
tuning, and expanded metrics.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from .evaluation import binary_classification_metrics


BASELINE_CLASSIFIER_CHOICES = ("always_positive", "always_negative")
SKLEARN_LEARNED_CLASSIFIER_CHOICES = (
    "logistic",
    "linear_svm",
    "sgd_logistic",
)
TORCH_CLASSIFIER_CHOICES = ("torch_mlp",)
LEARNED_CLASSIFIER_CHOICES = (
    SKLEARN_LEARNED_CLASSIFIER_CHOICES + TORCH_CLASSIFIER_CHOICES
)
CLASSIFIER_CHOICES = LEARNED_CLASSIFIER_CHOICES + BASELINE_CLASSIFIER_CHOICES


#######################
# Baseline estimators #
#######################
# Baseline classifier: predicts always positive or negative
class ConstantClassifier:
    """
    Minimal classifier that always predicts one class.
    """

    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def fit(self, x: Any, y: np.ndarray) -> "ConstantClassifier":
        """
        Keep an sklearn-like fit API for pipeline compatibility.
        """
        return self

    def predict(self, x: Any) -> np.ndarray:
        """
        Return the same hard class prediction for every row.
        """
        n_samples = self._n_samples(x)
        label = int(self.positive_probability >= 0.5)
        predictions = np.full(n_samples, label, dtype=int)

        return predictions

    def predict_proba(self, x: Any) -> np.ndarray:
        """
        Return a two-column class probability matrix.
        """
        n_samples = self._n_samples(x)
        positive = np.full(
            n_samples,
            self.positive_probability,
            dtype=float,
        )
        probabilities = np.column_stack((1.0 - positive, positive))

        return probabilities

    @staticmethod
    def _n_samples(x: Any) -> int:
        if hasattr(x, "shape"):
            n_samples = x.shape[0]
        else:
            n_samples = len(x)

        return n_samples


#################
# Model factory #
#################
def make_classifier(
        classifier_name: str, max_iter: int, random_state: int,
    ) -> Any:
    """
    Create a classifier based on the specified model name.
    """
    # Baseline classifiers
    if classifier_name == "always_positive":
        model = ConstantClassifier(positive_probability=1.0)
    elif classifier_name == "always_negative":
        model = ConstantClassifier(positive_probability=0.0)

    # Learned Classifiers
    elif classifier_name == "logistic":
        model = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )
    elif classifier_name == "linear_svm":
        model = LinearSVC(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )
    elif classifier_name == "sgd_logistic":
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )

    # Unknown classifier
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return model


####################
# Model evaluation #
####################
def get_scores_and_predictions(
        model: Any, x: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return probability-like scores and hard 0/1 predictions.
    """
    # Predicted probability
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)

    # Decision function
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x)
        y_pred = (y_score >= 0.0).astype(int)

    # Hard predictions only
    else:
        y_pred = model.predict(x)
        y_score = y_pred.astype(float)

    return y_score, y_pred


def default_decision_threshold(model: Any) -> float:
    """Return the estimator's conventional hard-decision threshold."""
    if hasattr(model, "predict_proba"):
        threshold = 0.5
    elif hasattr(model, "decision_function"):
        threshold = 0.0
    else:
        threshold = 0.5

    return threshold


def get_metrics(
        y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray,
        split_name: str | None = None,
    ) -> dict[str, float]:
    """
    Return binary classification metrics for a split.

    Inputs are flattened so callers can pass pandas Series, lists, or numpy
    arrays without changing metric behavior.
    """
    metrics = binary_classification_metrics(
        targets=y_true,
        scores=y_score,
        predictions=y_pred,
    )
    prefix = f"_{split_name}" if split_name else ""
    return {
        f"{metric_name}{prefix}": metric_value
        for metric_name, metric_value in metrics.items()
    }
