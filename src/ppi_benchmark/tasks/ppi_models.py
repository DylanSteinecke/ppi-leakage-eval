"""PPI-owned public model registry and matrix routing contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..backends.models import (
    ITERATION_BUDGET_FIXED,
    ITERATION_BUDGET_MAX_ITER,
    ITERATION_BUDGET_NONE,
    ITERATION_BUDGET_TORCH_EPOCHS,
    ExecutionPolicy,
    estimator_spec,
)


NO_MATRIX = "none"
TRAINING_DEGREE_MATRIX = "training_degree"
CONFIGURED_FEATURE_MATRIX = "configured_features"
MATRIX_SOURCES = (
    NO_MATRIX,
    TRAINING_DEGREE_MATRIX,
    CONFIGURED_FEATURE_MATRIX,
)
CONTROL_GROUP = "control"
PREDICTOR_GROUP = "predictor"
REPORTING_GROUPS = (CONTROL_GROUP, PREDICTOR_GROUP)


def _immutable_parameters(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    if any(not isinstance(key, str) for key in parameters):
        raise TypeError("Estimator parameter names must be strings.")
    try:
        canonical = json.dumps(
            dict(parameters),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Estimator parameters must contain JSON-compatible values."
        ) from exc
    ordered = json.loads(canonical)
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class PPIModelSpec:
    """One public PPI model configuration and its execution contract."""

    model_name: str
    estimator_id: str
    matrix_source: str
    reporting_group: str
    execution_policy: ExecutionPolicy
    estimator_params: Mapping[str, Any] = field(default_factory=dict)
    reporting_role: str = "predictive_model"

    def __post_init__(self) -> None:
        estimator_spec(self.estimator_id)
        if self.matrix_source not in MATRIX_SOURCES:
            raise ValueError(
                f"Unknown matrix source for {self.model_name!r}: "
                f"{self.matrix_source!r}."
            )
        if self.reporting_group not in REPORTING_GROUPS:
            raise ValueError(
                f"Unknown reporting group for {self.model_name!r}: "
                f"{self.reporting_group!r}."
            )
        object.__setattr__(
            self,
            "estimator_params",
            _immutable_parameters(self.estimator_params),
        )

    @property
    def backend(self) -> str:
        """Return the backend that owns this model's estimator."""
        return estimator_spec(self.estimator_id).backend


_CONSTANT_POLICY = ExecutionPolicy(
    run_per_model_seed=False,
    force_fixed_threshold=True,
    iteration_budget_source=ITERATION_BUDGET_NONE,
)
_SKLEARN_POLICY = ExecutionPolicy(
    iteration_budget_source=ITERATION_BUDGET_MAX_ITER,
)
_HGB_POLICY = ExecutionPolicy(
    iteration_budget_source=ITERATION_BUDGET_FIXED,
    fixed_iteration_budget=100,
)
_TORCH_POLICY = ExecutionPolicy(
    iteration_budget_source=ITERATION_BUDGET_TORCH_EPOCHS,
)


PPI_MODEL_SPECS = (
    PPIModelSpec(
        "logistic",
        "logistic",
        CONFIGURED_FEATURE_MATRIX,
        PREDICTOR_GROUP,
        _SKLEARN_POLICY,
        {"C": 1.0, "class_weight": "balanced", "solver": "liblinear"},
    ),
    PPIModelSpec(
        "linear_svm",
        "linear_svm",
        CONFIGURED_FEATURE_MATRIX,
        PREDICTOR_GROUP,
        _SKLEARN_POLICY,
        {"class_weight": "balanced"},
    ),
    PPIModelSpec(
        "sgd_logistic",
        "sgd_logistic",
        CONFIGURED_FEATURE_MATRIX,
        PREDICTOR_GROUP,
        _SKLEARN_POLICY,
        {
            "class_weight": "balanced",
            "loss": "log_loss",
            "penalty": "l2",
        },
    ),
    PPIModelSpec(
        "torch_mlp",
        "torch_mlp",
        CONFIGURED_FEATURE_MATRIX,
        PREDICTOR_GROUP,
        _TORCH_POLICY,
    ),
    PPIModelSpec(
        "always_positive",
        "constant",
        NO_MATRIX,
        CONTROL_GROUP,
        _CONSTANT_POLICY,
        {"positive_probability": 1.0},
        reporting_role="constant_control",
    ),
    PPIModelSpec(
        "always_negative",
        "constant",
        NO_MATRIX,
        CONTROL_GROUP,
        _CONSTANT_POLICY,
        {"positive_probability": 0.0},
        reporting_role="constant_control",
    ),
    PPIModelSpec(
        "degree_logistic",
        "logistic",
        TRAINING_DEGREE_MATRIX,
        CONTROL_GROUP,
        _SKLEARN_POLICY,
        {"C": 1.0, "class_weight": "balanced", "solver": "liblinear"},
        reporting_role="primary_degree_control",
    ),
    PPIModelSpec(
        "degree_hgb",
        "hist_gradient_boosting",
        TRAINING_DEGREE_MATRIX,
        CONTROL_GROUP,
        _HGB_POLICY,
        {
            "class_weight": "balanced",
            "early_stopping": False,
            "l2_regularization": 1.0,
            "learning_rate": 0.05,
            "max_depth": 3,
        },
        reporting_role="degree_sensitivity_control",
    ),
)
_PPI_MODEL_SPECS_BY_NAME = {
    spec.model_name: spec for spec in PPI_MODEL_SPECS
}
PPI_MODEL_CHOICES = tuple(spec.model_name for spec in PPI_MODEL_SPECS)
LEGACY_CLASSIFIER_ESTIMATOR_IDS = {
    "always_positive": "constant",
    "always_negative": "constant",
    "degree_logistic": "logistic",
    "degree_hgb": "hist_gradient_boosting",
    "logistic": "logistic",
    "linear_svm": "linear_svm",
    "sgd_logistic": "sgd_logistic",
    "torch_mlp": "torch_mlp",
}


def ppi_model_spec(model_name: str) -> PPIModelSpec:
    """Return one public PPI model specification."""
    try:
        return _PPI_MODEL_SPECS_BY_NAME[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown PPI model: {model_name}") from exc
