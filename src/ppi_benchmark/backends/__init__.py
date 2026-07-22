"""Model backend construction and public contracts."""

from pathlib import Path
from typing import Any, Mapping

from .base import (
    BackendFitResult,
    BackendPrediction,
    ModelBackend,
    SupervisedSplit,
)
from .models import (
    BACKEND_CHOICES,
    SKLEARN_BACKEND,
    TORCH_BACKEND,
    estimator_spec,
    plain_estimator_parameters,
)
from .sklearn import SklearnBackend


DEFAULT_BACKEND = SKLEARN_BACKEND


def make_model_backend(
        estimator_id: str, estimator_params: Mapping[str, Any],
        max_iter: int, random_state: int,
        backend_name: str | None = None,
        best_checkpoint_path: str | Path | None = None,
        last_checkpoint_path: str | Path | None = None,
        resume_from: str | Path | None = None,
        backend_options: Mapping[str, Any] | None = None,
        task_name: str = "binary_classification",
        task_schema_version: int = 1,
    ) -> ModelBackend:
    """Construct a model backend without coupling the runner to a framework."""
    if backend_name is not None and backend_name not in BACKEND_CHOICES:
        raise ValueError(f"Unknown model backend: {backend_name}")
    spec = estimator_spec(estimator_id)
    resolved_backend = backend_name or spec.backend
    if resolved_backend != spec.backend:
        raise ValueError(
            f"Estimator {estimator_id!r} requires backend "
            f"{spec.backend!r}, not {resolved_backend!r}."
        )
    options = dict(backend_options or {})
    if resolved_backend == DEFAULT_BACKEND:
        if options:
            raise ValueError("sklearn backend does not accept backend options.")
        return SklearnBackend.from_estimator(
            estimator_id=estimator_id,
            estimator_params=estimator_params,
            max_iter=max_iter,
            random_state=random_state,
        )
    if resolved_backend == TORCH_BACKEND:
        try:
            from .torch_mlp import TorchMLPBackend, TorchMLPConfig
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "torch_mlp requires PyTorch. Install the project with "
                    "`python -m pip install -e '.[torch]'`."
                ) from exc
            raise
        config = TorchMLPConfig(max_epochs=max_iter, **options)
        return TorchMLPBackend(
            config=config,
            random_state=random_state,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            resume_from=resume_from,
            task_name=task_name,
            task_schema_version=task_schema_version,
        )
    raise RuntimeError(f"No backend factory for {resolved_backend!r}.")


__all__ = [
    "BACKEND_CHOICES",
    "DEFAULT_BACKEND",
    "TORCH_BACKEND",
    "BackendFitResult",
    "BackendPrediction",
    "ModelBackend",
    "SklearnBackend",
    "SupervisedSplit",
    "make_model_backend",
    "plain_estimator_parameters",
]
