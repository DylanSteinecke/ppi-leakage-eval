"""Adapter for the benchmark's existing sklearn estimator path."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ..reporting.performance import solver_iteration_report
from .base import BackendFitResult, BackendPrediction, SupervisedSplit
from .models import make_classifier, score_estimator


class SklearnBackend:
    """Expose an sklearn estimator through the model-backend contract."""

    backend_name = "sklearn"

    def __init__(self, estimator: Any):
        self.estimator = estimator

    @classmethod
    def from_classifier(
            cls, classifier_name: str, max_iter: int, random_state: int,
        ) -> "SklearnBackend":
        """Build an adapter with the existing classifier factory."""
        estimator = make_classifier(
            classifier_name=classifier_name,
            max_iter=max_iter,
            random_state=random_state,
        )
        return cls(estimator)

    def fit(
            self, train: SupervisedSplit,
            validation: SupervisedSplit | None = None,
        ) -> BackendFitResult:
        """Fit only on the training split, as the existing path does."""
        del validation
        fit_started_at = perf_counter()
        self.estimator.fit(train.inputs, train.targets)
        fit_seconds = perf_counter() - fit_started_at
        return BackendFitResult(
            fit_seconds=fit_seconds,
            iteration_report=solver_iteration_report(self.estimator),
        )

    def predict(self, inputs: Any) -> BackendPrediction:
        """Use the existing sklearn score and threshold behavior."""
        scores, default_threshold = score_estimator(self.estimator, inputs)
        return BackendPrediction(
            scores=scores,
            default_threshold=default_threshold,
        )
