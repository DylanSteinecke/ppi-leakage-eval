"""Model backend construction and public contracts."""

from pathlib import Path
from typing import Any, Mapping

from ..models import MODEL_SPECS, model_spec
from .base import (
    BackendFitResult,
    BackendPrediction,
    ModelBackend,
    SupervisedSplit,
)
from .sklearn import SklearnBackend


BACKEND_CHOICES = tuple(dict.fromkeys(
    spec.backend for spec in MODEL_SPECS
))
DEFAULT_BACKEND = model_spec("logistic").backend
TORCH_BACKEND = model_spec("torch_mlp").backend


def backend_name_for_classifier(classifier_name: str) -> str:
    """Return the framework backend that owns a classifier."""
    return model_spec(classifier_name).backend


def make_model_backend(
        classifier_name: str, max_iter: int, random_state: int,
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
    spec = model_spec(classifier_name)
    resolved_backend = backend_name or spec.backend
    if resolved_backend != spec.backend:
        raise ValueError(
            f"Classifier {classifier_name!r} requires backend "
            f"{spec.backend!r}, not {resolved_backend!r}."
        )
    options = dict(backend_options or {})
    if resolved_backend == DEFAULT_BACKEND:
        if options:
            raise ValueError("sklearn backend does not accept backend options.")
        return SklearnBackend.from_classifier(
            classifier_name=classifier_name,
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
    "backend_name_for_classifier",
    "make_model_backend",
]
