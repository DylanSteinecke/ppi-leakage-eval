"""Task-neutral mini-batch training engine for PyTorch models."""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from .torch_checkpoints import (
    TorchCheckpointManager,
    cloned_component_states,
    load_component_states,
)
from .torch_utils import TORCH_TRAINING_PRECISIONS


MONITOR_MODES = ("max", "min")


@dataclass(frozen=True)
class TorchTrainerConfig:
    """Framework-level optimization and early-stopping configuration."""

    max_epochs: int
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    precision: str = "float32"
    validation_monitor: str = "auprc"
    validation_monitor_mode: str = "max"

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError(
                "max_epochs, batch_size, and patience must be positive."
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and nonnegative.")
        if self.precision not in TORCH_TRAINING_PRECISIONS:
            raise ValueError(
                f"precision must be one of: {TORCH_TRAINING_PRECISIONS}."
            )
        if not self.validation_monitor:
            raise ValueError("validation_monitor must not be empty.")
        if self.validation_monitor_mode not in MONITOR_MODES:
            raise ValueError("validation_monitor_mode must be max or min.")


@dataclass(frozen=True)
class TorchStepOutput:
    """Task-owned forward output consumed by the shared trainer."""

    scores: torch.Tensor
    n_observations: int
    loss: torch.Tensor | None = None
    targets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.n_observations < 1:
            raise ValueError("A Torch step must contain at least one example.")
        if self.scores.shape[0] != self.n_observations:
            raise ValueError("Torch step scores have the wrong batch size.")
        if self.targets is not None and self.targets.shape[0] != (
            self.n_observations
        ):
            raise ValueError("Torch step targets have the wrong batch size.")


@dataclass(frozen=True)
class TorchEpochSummary:
    """Loss, task metrics, and batch counts for one evaluation pass."""

    loss: float | None
    metrics: Mapping[str, float]
    n_observations: int
    n_batches: int


@dataclass(frozen=True)
class TorchTrainingResult:
    """Reusable observations returned after fitting one Torch task."""

    fit_seconds: float
    history: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@runtime_checkable
class TorchTaskRuntime(Protocol):
    """Task connector consumed by the framework-level Torch trainer."""

    name: str
    schema_version: int
    module: nn.Module

    def n_examples(self, prepared_split: Any) -> int:
        """Return the number of examples in a prepared split."""
        ...

    def iter_batches(
            self, prepared_split: Any, batch_size: int,
            row_order: np.ndarray,
        ) -> Iterator[Any]:
        """Yield task-specific batches in the requested row order."""
        ...

    def step(self, batch: Any, training: bool) -> TorchStepOutput:
        """Compute task loss, targets, and score-like predictions."""
        ...

    def metric_values(
            self, targets: np.ndarray, scores: np.ndarray,
        ) -> Mapping[str, float]:
        """Return validation metrics without framework assumptions."""
        ...

    def data_signature(
            self, train: Any, validation: Any | None,
        ) -> Mapping[str, Any]:
        """Return the exact task-specific split/input identity."""
        ...

    def checkpoint_identity(self) -> Mapping[str, Any]:
        """Return architecture and task-head identity fields."""
        ...

    def checkpoint_components(self) -> Mapping[str, nn.Module]:
        """Return encoder/head modules saved independently in checkpoints."""
        ...

    def runtime_state(self) -> Mapping[str, Any]:
        """Return task batching state needed for reporting/resumption."""
        ...

    def load_runtime_state(self, state: Mapping[str, Any]) -> None:
        """Restore task batching state from a last checkpoint."""
        ...

    def runtime_metadata(self) -> Mapping[str, Any]:
        """Return task-specific fit metadata."""
        ...


class TorchTrainer:
    """Own optimization, precision, monitoring, and resumable checkpoints."""

    def __init__(
            self, *, task: TorchTaskRuntime, config: TorchTrainerConfig,
            device: torch.device, random_state: int,
            checkpoint_manager: TorchCheckpointManager,
            backend_name: str, model_name: str,
            checkpoint_config: Mapping[str, Any],
            compatible_config_fields: tuple[str, ...],
            config_defaults: Mapping[str, Any] | None = None,
            scheduler_factory: Any | None = None,
        ):
        self.task = task
        self.config = config
        self.device = device
        self.random_state = random_state
        self.checkpoints = checkpoint_manager
        self.backend_name = backend_name
        self.model_name = model_name
        self.checkpoint_config = dict(checkpoint_config)
        self.compatible_config_fields = compatible_config_fields
        self.config_defaults = dict(config_defaults or {})
        self.scheduler_factory = scheduler_factory
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.scaler = None
        self._validate_precision()
        self.task.module.to(self.device)

    def _validate_precision(self) -> None:
        if self.device.type == "cpu" and self.config.precision == "float16":
            raise ValueError(
                "float16 Torch training on CPU is not supported; use "
                "float32 or bfloat16."
            )
        if self.device.type == "mps" and self.config.precision == "bfloat16":
            raise ValueError(
                "bfloat16 Torch training on MPS is not supported; use "
                "float16 or float32."
            )

    def _autocast_context(self):
        if self.config.precision == "float32":
            return nullcontext()
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.precision]
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.task.module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _evaluate(self, prepared_split: Any) -> TorchEpochSummary:
        self.task.module.eval()
        score_batches = []
        target_batches = []
        total_loss = 0.0
        loss_observations = 0
        n_batches = 0
        row_order = np.arange(self.task.n_examples(prepared_split))
        with torch.inference_mode():
            for batch in self.task.iter_batches(
                    prepared_split, self.config.batch_size, row_order):
                with self._autocast_context():
                    output = self.task.step(batch, training=False)
                score_batches.append(output.scores.detach().float().cpu().numpy())
                if output.targets is not None:
                    target_batches.append(
                        output.targets.detach().cpu().numpy()
                    )
                if output.loss is not None:
                    total_loss += (
                        float(output.loss.detach().item())
                        * output.n_observations
                    )
                    loss_observations += output.n_observations
                n_batches += 1
        scores = np.concatenate(score_batches, axis=0)
        targets = (
            None
            if not target_batches
            else np.concatenate(target_batches, axis=0)
        )
        metrics = (
            {}
            if targets is None
            else dict(self.task.metric_values(targets, scores))
        )
        return TorchEpochSummary(
            loss=(
                None
                if loss_observations == 0
                else total_loss / loss_observations
            ),
            metrics=metrics,
            n_observations=len(scores),
            n_batches=n_batches,
        )

    def predict(self, prepared_split: Any) -> np.ndarray:
        """Return score-like task predictions without requiring labels."""
        summary_scores = []
        self.task.module.eval()
        row_order = np.arange(self.task.n_examples(prepared_split))
        with torch.inference_mode():
            for batch in self.task.iter_batches(
                    prepared_split, self.config.batch_size, row_order):
                with self._autocast_context():
                    output = self.task.step(batch, training=False)
                summary_scores.append(
                    output.scores.detach().float().cpu().numpy()
                )
        return np.concatenate(summary_scores, axis=0)

    def _monitor_value(
            self, validation_summary: TorchEpochSummary | None,
            train_loss: float,
        ) -> tuple[str, float, str]:
        if validation_summary is None:
            return "train_loss", train_loss, "min"
        metric_name = self.config.validation_monitor
        monitor_name = f"validation_{metric_name}"
        if metric_name == "loss":
            value = validation_summary.loss
        else:
            value = validation_summary.metrics.get(metric_name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Validation monitor {monitor_name!r} was not a finite metric."
            )
        return (
            monitor_name,
            float(value),
            self.config.validation_monitor_mode,
        )

    def _is_improved(
            self, value: float, best: float, mode: str, first: bool,
        ) -> bool:
        if first:
            return True
        if mode == "max":
            return value > best + self.config.min_delta
        return value < best - self.config.min_delta

    def _base_checkpoint_payload(
            self, data_signature: Mapping[str, Any],
        ) -> dict[str, Any]:
        return self.checkpoints.base_payload(
            backend_name=self.backend_name,
            model_name=self.model_name,
            task_name=self.task.name,
            task_schema_version=self.task.schema_version,
            task_identity=self.task.checkpoint_identity(),
            config=self.checkpoint_config,
            random_state=self.random_state,
            data_signature=data_signature,
        )

    def fit(
            self, train: Any, validation: Any | None = None,
        ) -> TorchTrainingResult:
        """Train one task and restore its best monitored components."""
        self._synchronize_device()
        fit_started_at = perf_counter()
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        random_generator = np.random.default_rng(self.random_state)
        self.optimizer = self._make_optimizer()
        self.scheduler = (
            None
            if self.scheduler_factory is None
            else self.scheduler_factory(self.optimizer)
        )
        scaler_enabled = (
            self.device.type == "cuda"
            and self.config.precision == "float16"
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
        )
        data_signature = dict(self.task.data_signature(train, validation))
        base_payload = self._base_checkpoint_payload(data_signature)
        components = self.task.checkpoint_components()

        has_validation = validation is not None
        history: list[dict[str, Any]] = []
        best_epoch = 0
        best_monitor_value = -math.inf
        best_validation_loss = None
        best_validation_metrics: dict[str, float] = {}
        best_component_states = None
        epochs_without_improvement = 0
        optimizer_steps = 0
        stopped_early = False
        resumed_from_epoch = 0
        monitor_metric = (
            f"validation_{self.config.validation_monitor}"
            if has_validation else "train_loss"
        )
        monitor_mode = (
            self.config.validation_monitor_mode if has_validation else "min"
        )

        resume_checkpoint = self.checkpoints.load_resume(self.device)
        if resume_checkpoint is not None:
            self.checkpoints.validate_resume(
                resume_checkpoint,
                backend_name=self.backend_name,
                model_name=self.model_name,
                task_name=self.task.name,
                task_schema_version=self.task.schema_version,
                task_identity=self.task.checkpoint_identity(),
                random_state=self.random_state,
                data_signature=data_signature,
                config=self.checkpoint_config,
                compatible_config_fields=self.compatible_config_fields,
                config_defaults=self.config_defaults,
            )
            if resume_checkpoint.get("monitor_metric") != monitor_metric:
                raise ValueError(
                    "Resume checkpoint validation configuration differs."
                )
            component_names = tuple(components)
            load_component_states(
                components,
                self.checkpoints.checkpoint_component_states(
                    resume_checkpoint,
                    component_names,
                ),
            )
            self.optimizer.load_state_dict(
                resume_checkpoint["optimizer_state_dict"]
            )
            scheduler_state = resume_checkpoint.get("scheduler_state_dict")
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            scaler_state = resume_checkpoint.get("scaler_state_dict")
            if scaler_state is not None:
                self.scaler.load_state_dict(scaler_state)
            history = [dict(row) for row in resume_checkpoint["history"]]
            resumed_from_epoch = int(resume_checkpoint["epoch"])
            best_epoch = int(resume_checkpoint["best_epoch"])
            best_monitor_value = float(
                resume_checkpoint["best_monitor_value"]
            )
            best_validation_loss = resume_checkpoint.get(
                "best_validation_loss"
            )
            best_validation_metrics = dict(
                resume_checkpoint.get("best_validation_metrics", {})
            )
            if (
                not best_validation_metrics
                and monitor_metric == "validation_auprc"
            ):
                best_validation_metrics["auprc"] = best_monitor_value
            best_component_states = (
                self.checkpoints.checkpoint_component_states(
                    resume_checkpoint,
                    component_names,
                    best=True,
                )
            )
            epochs_without_improvement = int(
                resume_checkpoint["epochs_without_improvement"]
            )
            optimizer_steps = int(resume_checkpoint["optimizer_steps"])
            stopped_early = bool(resume_checkpoint["stopped_early"])
            self.task.load_runtime_state(
                resume_checkpoint.get("task_runtime_state", {
                    "max_dense_batch_rows": resume_checkpoint.get(
                        "max_dense_batch_rows", 0
                    ),
                })
            )
            self.checkpoints.restore_rng_state(
                resume_checkpoint,
                self.device,
                random_generator,
            )
            self.checkpoints.save_best(
                base_payload=base_payload,
                component_states=best_component_states,
                epoch=best_epoch,
                monitor_metric=monitor_metric,
                monitor_value=best_monitor_value,
            )

        start_epoch = resumed_from_epoch + 1
        trained_epoch = False
        for epoch in range(start_epoch, self.config.max_epochs + 1):
            trained_epoch = True
            stopped_early = False
            epoch_started_at = perf_counter()
            self.task.module.train()
            train_loss_total = 0.0
            train_observations = 0
            train_batches = 0
            row_order = random_generator.permutation(
                self.task.n_examples(train)
            )
            for batch in self.task.iter_batches(
                    train, self.config.batch_size, row_order):
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    output = self.task.step(batch, training=True)
                if output.loss is None or output.targets is None:
                    raise RuntimeError(
                        "Training task steps must return loss and targets."
                    )
                self.scaler.scale(output.loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_loss_total += (
                    float(output.loss.detach().item())
                    * output.n_observations
                )
                train_observations += output.n_observations
                train_batches += 1
                optimizer_steps += 1
            train_loss = train_loss_total / train_observations
            validation_summary = (
                None if validation is None else self._evaluate(validation)
            )
            monitor_metric, monitor_value, monitor_mode = self._monitor_value(
                validation_summary,
                train_loss,
            )
            improved = self._is_improved(
                monitor_value,
                best_monitor_value,
                monitor_mode,
                first=best_epoch == 0,
            )
            if improved:
                best_epoch = epoch
                best_monitor_value = monitor_value
                best_validation_loss = (
                    None
                    if validation_summary is None
                    else validation_summary.loss
                )
                best_validation_metrics = (
                    {}
                    if validation_summary is None
                    else dict(validation_summary.metrics)
                )
                epochs_without_improvement = 0
                best_component_states = cloned_component_states(components)
                self.checkpoints.save_best(
                    base_payload=base_payload,
                    component_states=best_component_states,
                    epoch=epoch,
                    monitor_metric=monitor_metric,
                    monitor_value=monitor_value,
                )
            else:
                epochs_without_improvement += 1

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            self._synchronize_device()
            should_stop = bool(
                has_validation
                and epochs_without_improvement >= self.config.patience
            )
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": (
                    None
                    if validation_summary is None
                    else validation_summary.loss
                ),
                "monitor_metric": monitor_metric,
                "monitor_value": monitor_value,
                "improved": improved,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(perf_counter() - epoch_started_at),
                "train_batches": train_batches,
                "validation_batches": (
                    0
                    if validation_summary is None
                    else validation_summary.n_batches
                ),
            }
            if validation_summary is not None:
                history_row.update({
                    f"validation_{metric_name}": metric_value
                    for metric_name, metric_value in (
                        validation_summary.metrics.items()
                    )
                })
            history.append(history_row)
            if best_component_states is None:
                raise RuntimeError("No best Torch task state was selected.")
            training_state = {
                "history": history,
                "best_epoch": best_epoch,
                "best_monitor_value": best_monitor_value,
                "best_validation_loss": best_validation_loss,
                "best_validation_metrics": best_validation_metrics,
                "epochs_without_improvement": epochs_without_improvement,
                "optimizer_steps": optimizer_steps,
                "stopped_early": should_stop,
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
            }
            self.checkpoints.save_last(
                base_payload=base_payload,
                component_states=cloned_component_states(components),
                best_component_states=best_component_states,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                random_generator=random_generator,
                device=self.device,
                training_state=training_state,
                task_runtime_state=self.task.runtime_state(),
            )
            if should_stop:
                stopped_early = True
                break

        if best_component_states is None:
            raise RuntimeError("No best Torch task state was selected.")
        if not trained_epoch and resume_checkpoint is not None:
            training_state = {
                "history": history,
                "best_epoch": best_epoch,
                "best_monitor_value": best_monitor_value,
                "best_validation_loss": best_validation_loss,
                "best_validation_metrics": best_validation_metrics,
                "epochs_without_improvement": epochs_without_improvement,
                "optimizer_steps": optimizer_steps,
                "stopped_early": stopped_early,
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
            }
            self.checkpoints.save_last(
                base_payload=base_payload,
                component_states=cloned_component_states(components),
                best_component_states=best_component_states,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                random_generator=random_generator,
                device=self.device,
                training_state=training_state,
                task_runtime_state=self.task.runtime_state(),
            )

        load_component_states(components, best_component_states)
        self._synchronize_device()
        fit_seconds = float(perf_counter() - fit_started_at)
        metadata = {
            "epochs_completed": len(history),
            "max_epochs": self.config.max_epochs,
            "optimizer_steps": optimizer_steps,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor_value,
            "best_validation_metrics": best_validation_metrics,
            "best_validation_auprc": best_validation_metrics.get("auprc"),
            "best_validation_loss": best_validation_loss,
            "early_stopping_metric": monitor_metric,
            "early_stopping_mode": monitor_mode,
            "stopped_early": stopped_early,
            "resumed_from": (
                None
                if self.checkpoints.resume_from is None
                else str(self.checkpoints.resume_from)
            ),
            "resumed_from_sha256": self.checkpoints.resume_sha256,
            "resumed_from_epoch": resumed_from_epoch,
            "best_checkpoint_path": (
                None
                if self.checkpoints.best_path is None
                else str(self.checkpoints.best_path)
            ),
            "last_checkpoint_path": (
                None
                if self.checkpoints.last_path is None
                else str(self.checkpoints.last_path)
            ),
            "device": str(self.device),
            "precision": self.config.precision,
            "batch_size": self.config.batch_size,
            "task": self.task.name,
            "task_schema_version": self.task.schema_version,
            **self.task.runtime_metadata(),
        }
        return TorchTrainingResult(
            fit_seconds=fit_seconds,
            history=tuple(history),
            metadata=metadata,
        )
