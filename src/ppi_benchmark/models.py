"""
Model registration, sklearn construction, and score extraction.

This module owns sklearn classifier factories, baseline estimators, score
extraction, and metric calculation. Framework adapters live in ``backends``.
Register each model once here so CLI choices and backend routing cannot drift.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

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


ClassifierFactory = Callable[[int, int], Any]


def _make_logistic(max_iter: int, random_state: int) -> Any:
    return LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )


def _make_linear_svm(max_iter: int, random_state: int) -> Any:
    return LinearSVC(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )


def _make_sgd_logistic(max_iter: int, random_state: int) -> Any:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )


def _make_always_positive(max_iter: int, random_state: int) -> Any:
    del max_iter, random_state
    return ConstantClassifier(positive_probability=1.0)


def _make_always_negative(max_iter: int, random_state: int) -> Any:
    del max_iter, random_state
    return ConstantClassifier(positive_probability=0.0)


@dataclass(frozen=True)
class ModelSpec:
    """One registered model and the backend that owns it."""

    name: str
    backend: str
    baseline: bool = False
    estimator_factory: ClassifierFactory | None = None


MODEL_SPECS = (
    ModelSpec("logistic", "sklearn", estimator_factory=_make_logistic),
    ModelSpec("linear_svm", "sklearn", estimator_factory=_make_linear_svm),
    ModelSpec(
        "sgd_logistic",
        "sklearn",
        estimator_factory=_make_sgd_logistic,
    ),
    ModelSpec("torch_mlp", "torch"),
    ModelSpec(
        "always_positive",
        "sklearn",
        baseline=True,
        estimator_factory=_make_always_positive,
    ),
    ModelSpec(
        "always_negative",
        "sklearn",
        baseline=True,
        estimator_factory=_make_always_negative,
    ),
)
_MODEL_SPECS_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
CLASSIFIER_CHOICES = tuple(spec.name for spec in MODEL_SPECS)


def model_spec(classifier_name: str) -> ModelSpec:
    """Return the single registered specification for a model."""
    try:
        return _MODEL_SPECS_BY_NAME[classifier_name]
    except KeyError as exc:
        raise ValueError(f"Unknown classifier: {classifier_name}") from exc


def is_baseline_classifier(classifier_name: str) -> bool:
    """Return whether a registered model ignores feature matrices."""
    return model_spec(classifier_name).baseline


def make_classifier(
        classifier_name: str, max_iter: int, random_state: int,
    ) -> Any:
    """
    Create the sklearn estimator owned by a registered model.
    """
    spec = model_spec(classifier_name)
    if spec.estimator_factory is None:
        raise ValueError(
            f"Classifier {classifier_name!r} is owned by the "
            f"{spec.backend!r} backend and has no sklearn estimator."
        )
    return spec.estimator_factory(max_iter, random_state)


def score_estimator(model: Any, x: Any) -> tuple[np.ndarray, float]:
    """Return one score per example and its conventional threshold."""
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)[:, 1]
        threshold = 0.5
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x)
        threshold = 0.0
    else:
        y_score = np.asarray(model.predict(x), dtype=float)
        threshold = 0.5
    return np.asarray(y_score).reshape(-1), threshold
