"""Task-neutral estimator construction and execution policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC


SKLEARN_BACKEND = "sklearn"
TORCH_BACKEND = "torch"
ITERATION_BUDGET_NONE = "none"
ITERATION_BUDGET_MAX_ITER = "max_iter"
ITERATION_BUDGET_TORCH_EPOCHS = "torch_max_epochs"
ITERATION_BUDGET_FIXED = "fixed"
ITERATION_BUDGET_SOURCES = (
    ITERATION_BUDGET_NONE,
    ITERATION_BUDGET_MAX_ITER,
    ITERATION_BUDGET_TORCH_EPOCHS,
    ITERATION_BUDGET_FIXED,
)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Task-neutral rules for reruns, thresholds, and iteration budgets."""

    run_per_model_seed: bool = True
    force_fixed_threshold: bool = False
    iteration_budget_source: str = ITERATION_BUDGET_MAX_ITER
    fixed_iteration_budget: int | None = None

    def __post_init__(self) -> None:
        if self.iteration_budget_source not in ITERATION_BUDGET_SOURCES:
            raise ValueError(
                "Unknown iteration budget source: "
                f"{self.iteration_budget_source!r}."
            )
        if self.iteration_budget_source == ITERATION_BUDGET_FIXED:
            if self.fixed_iteration_budget is None:
                raise ValueError("Fixed iteration budgets require a value.")
            if self.fixed_iteration_budget < 1:
                raise ValueError("Fixed iteration budgets must be positive.")
        elif self.fixed_iteration_budget is not None:
            raise ValueError(
                "fixed_iteration_budget is only valid with source='fixed'."
            )

    def resolve_iteration_budget(
        self,
        *,
        max_iter: int,
        torch_max_epochs: int,
    ) -> int | None:
        """Resolve the backend budget from one invocation's settings."""
        if self.iteration_budget_source == ITERATION_BUDGET_NONE:
            return None
        if self.iteration_budget_source == ITERATION_BUDGET_MAX_ITER:
            return max_iter
        if self.iteration_budget_source == ITERATION_BUDGET_TORCH_EPOCHS:
            return torch_max_epochs
        return self.fixed_iteration_budget


class ConstantClassifier:
    """Minimal sklearn-compatible constant binary classifier."""

    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def fit(self, x: Any, y: np.ndarray) -> "ConstantClassifier":
        """Keep an sklearn-like fit API for pipeline compatibility."""
        del x, y
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Return the same hard class prediction for every row."""
        label = int(self.positive_probability >= 0.5)
        return np.full(self._n_samples(x), label, dtype=int)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return a two-column class probability matrix."""
        positive = np.full(
            self._n_samples(x),
            self.positive_probability,
            dtype=float,
        )
        return np.column_stack((1.0 - positive, positive))

    @staticmethod
    def _n_samples(x: Any) -> int:
        return int(x.shape[0]) if hasattr(x, "shape") else len(x)


EstimatorFactory = Callable[[int, int, Mapping[str, Any]], Any]


def _make_logistic(
    max_iter: int,
    random_state: int,
    parameters: Mapping[str, Any],
) -> Any:
    return LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        **parameters,
    )


def _make_linear_svm(
    max_iter: int,
    random_state: int,
    parameters: Mapping[str, Any],
) -> Any:
    return LinearSVC(
        max_iter=max_iter,
        random_state=random_state,
        **parameters,
    )


def _make_sgd_logistic(
    max_iter: int,
    random_state: int,
    parameters: Mapping[str, Any],
) -> Any:
    return SGDClassifier(
        max_iter=max_iter,
        random_state=random_state,
        **parameters,
    )


def _make_constant(
    max_iter: int,
    random_state: int,
    parameters: Mapping[str, Any],
) -> Any:
    del max_iter, random_state
    return ConstantClassifier(**parameters)


def _make_hist_gradient_boosting(
    max_iter: int,
    random_state: int,
    parameters: Mapping[str, Any],
) -> Any:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        random_state=random_state,
        **parameters,
    )


@dataclass(frozen=True)
class EstimatorSpec:
    """One generic estimator implementation and its owning backend."""

    estimator_id: str
    backend: str
    factory: EstimatorFactory | None = None


ESTIMATOR_SPECS = (
    EstimatorSpec("constant", SKLEARN_BACKEND, _make_constant),
    EstimatorSpec("logistic", SKLEARN_BACKEND, _make_logistic),
    EstimatorSpec("linear_svm", SKLEARN_BACKEND, _make_linear_svm),
    EstimatorSpec("sgd_logistic", SKLEARN_BACKEND, _make_sgd_logistic),
    EstimatorSpec(
        "hist_gradient_boosting",
        SKLEARN_BACKEND,
        _make_hist_gradient_boosting,
    ),
    EstimatorSpec("torch_mlp", TORCH_BACKEND),
)
_ESTIMATOR_SPECS_BY_ID = {
    spec.estimator_id: spec for spec in ESTIMATOR_SPECS
}
ESTIMATOR_IDS = tuple(spec.estimator_id for spec in ESTIMATOR_SPECS)
BACKEND_CHOICES = tuple(dict.fromkeys(
    spec.backend for spec in ESTIMATOR_SPECS
))


def estimator_spec(estimator_id: str) -> EstimatorSpec:
    """Return the registered generic estimator specification."""
    try:
        return _ESTIMATOR_SPECS_BY_ID[estimator_id]
    except KeyError as exc:
        raise ValueError(f"Unknown estimator: {estimator_id}") from exc


def make_estimator(
    estimator_id: str,
    *,
    estimator_params: Mapping[str, Any],
    max_iter: int,
    random_state: int,
) -> Any:
    """Construct a generic sklearn estimator from explicit parameters."""
    spec = estimator_spec(estimator_id)
    if spec.factory is None:
        raise ValueError(
            f"Estimator {estimator_id!r} is owned by the {spec.backend!r} "
            "backend and has no sklearn factory."
        )
    return spec.factory(max_iter, random_state, dict(estimator_params))


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
