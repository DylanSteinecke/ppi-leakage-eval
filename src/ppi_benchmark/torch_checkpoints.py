"""Reusable, task-aware checkpoint management for PyTorch trainers."""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .datasets.common import file_sha256


TORCH_CHECKPOINT_FORMAT_VERSION = 3


def cloned_component_states(
        components: Mapping[str, nn.Module],
    ) -> dict[str, dict[str, torch.Tensor]]:
    """Copy component state dictionaries to immutable CPU tensors."""
    if not components:
        raise ValueError("At least one checkpoint component is required.")
    return {
        component_name: {
            parameter_name: tensor.detach().cpu().clone()
            for parameter_name, tensor in component.state_dict().items()
        }
        for component_name, component in components.items()
    }


def load_component_states(
        components: Mapping[str, nn.Module],
        component_states: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
    """Restore every declared component with exact name matching."""
    if set(components) != set(component_states):
        raise ValueError(
            "Checkpoint component names differ from the current task: "
            f"checkpoint={sorted(component_states)}, "
            f"current={sorted(components)}"
        )
    for component_name, component in components.items():
        component.load_state_dict(component_states[component_name])


class TorchCheckpointManager:
    """Write, load, validate, and restore generic Torch checkpoints."""

    format_version = TORCH_CHECKPOINT_FORMAT_VERSION

    def __init__(
            self, best_path: str | Path | None = None,
            last_path: str | Path | None = None,
            resume_from: str | Path | None = None,
        ):
        self.best_path = None if best_path is None else Path(best_path)
        self.last_path = None if last_path is None else Path(last_path)
        self.resume_from = (
            None if resume_from is None else Path(resume_from)
        )
        self.memory_best: dict[str, Any] | None = None
        self.memory_last: dict[str, Any] | None = None

    @staticmethod
    def _save_payload(payload: dict[str, Any], path: Path | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(
            f".{path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_resume(self, device: torch.device) -> dict[str, Any] | None:
        """Load the configured immutable resume source, if any."""
        if self.resume_from is None:
            return None
        if not self.resume_from.is_file():
            raise ValueError(
                f"Resume checkpoint does not exist: {self.resume_from}"
            )
        payload = torch.load(
            self.resume_from,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid Torch checkpoint payload: {self.resume_from}"
            )
        return payload

    @staticmethod
    def rng_state(device: torch.device) -> dict[str, Any]:
        """Capture framework RNG state required for exact resumption."""
        state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
            "torch_mps": None,
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        if device.type == "mps":
            state["torch_mps"] = torch.mps.get_rng_state()
        return state

    @staticmethod
    def restore_rng_state(
            checkpoint: Mapping[str, Any], device: torch.device,
            random_generator: np.random.Generator,
        ) -> None:
        """Restore Python, NumPy, and framework RNG state."""
        python_rng_state = checkpoint.get("python_rng_state")
        if python_rng_state is not None:
            random.setstate(python_rng_state)
        random_generator.bit_generator.state = checkpoint[
            "numpy_rng_state"
        ]
        rng_state = checkpoint["torch_rng_state"]
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if device.type == "cuda" and rng_state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        if device.type == "mps" and rng_state["torch_mps"] is not None:
            torch.mps.set_rng_state(rng_state["torch_mps"])

    def base_payload(
            self, *, backend_name: str, model_name: str,
            task_name: str, task_schema_version: int,
            task_identity: Mapping[str, Any], config: Mapping[str, Any],
            random_state: int, data_signature: Mapping[str, Any],
        ) -> dict[str, Any]:
        """Return common provenance for best and last checkpoints."""
        payload = {
            "format_version": self.format_version,
            "backend_name": backend_name,
            "classifier_name": model_name,
            "task_name": task_name,
            "task_schema_version": task_schema_version,
            "task_identity": dict(task_identity),
            "config": dict(config),
            "random_state": random_state,
            "data_signature": dict(data_signature),
        }
        input_dim = task_identity.get("input_dim")
        if input_dim is not None:
            payload["input_dim"] = int(input_dim)
        return payload

    @staticmethod
    def _legacy_model_state(
            component_states: Mapping[str, Mapping[str, torch.Tensor]],
        ) -> Mapping[str, torch.Tensor]:
        for preferred_name in ("task_head", "model"):
            if preferred_name in component_states:
                return component_states[preferred_name]
        return next(iter(component_states.values()))

    @classmethod
    def checkpoint_component_states(
            cls, checkpoint: Mapping[str, Any],
            component_names: tuple[str, ...], best: bool = False,
        ) -> dict[str, Mapping[str, torch.Tensor]]:
        """Read new component states or adapt a legacy single-model state."""
        field_name = (
            "best_component_state_dicts"
            if best else "component_state_dicts"
        )
        states = checkpoint.get(field_name)
        if states is not None:
            return dict(states)
        legacy_field = "best_model_state_dict" if best else "model_state_dict"
        legacy_state = checkpoint.get(legacy_field)
        if legacy_state is None or len(component_names) != 1:
            raise ValueError("Checkpoint is missing required component states.")
        return {component_names[0]: legacy_state}

    def save_best(
            self, *, base_payload: Mapping[str, Any],
            component_states: Mapping[str, Mapping[str, torch.Tensor]],
            epoch: int, monitor_metric: str, monitor_value: float,
        ) -> None:
        """Persist inference components from the best monitored epoch."""
        states = dict(component_states)
        payload = {
            **base_payload,
            "checkpoint_kind": "best",
            "component_state_dicts": states,
            "model_state_dict": self._legacy_model_state(states),
            "encoder_state_dict": states.get("encoder"),
            "task_head_state_dict": states.get("task_head"),
            "epoch": epoch,
            "monitor_metric": monitor_metric,
            "monitor_value": monitor_value,
        }
        self.memory_best = payload
        self._save_payload(payload, self.best_path)

    def save_last(
            self, *, base_payload: Mapping[str, Any],
            component_states: Mapping[str, Mapping[str, torch.Tensor]],
            best_component_states: Mapping[
                str, Mapping[str, torch.Tensor]
            ],
            optimizer: torch.optim.Optimizer,
            scheduler: Any | None, scaler: Any | None,
            random_generator: np.random.Generator,
            device: torch.device, training_state: Mapping[str, Any],
            task_runtime_state: Mapping[str, Any],
        ) -> None:
        """Persist all state required for exact training resumption."""
        states = dict(component_states)
        best_states = dict(best_component_states)
        history = training_state["history"]
        payload = {
            **base_payload,
            "checkpoint_kind": "last",
            "component_state_dicts": states,
            "best_component_state_dicts": best_states,
            "model_state_dict": self._legacy_model_state(states),
            "best_model_state_dict": self._legacy_model_state(best_states),
            "encoder_state_dict": states.get("encoder"),
            "task_head_state_dict": states.get("task_head"),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                None if scheduler is None else scheduler.state_dict()
            ),
            "scaler_state_dict": (
                None if scaler is None else scaler.state_dict()
            ),
            "epoch": int(history[-1]["epoch"]),
            **training_state,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": random_generator.bit_generator.state,
            "torch_rng_state": self.rng_state(device),
            "task_runtime_state": dict(task_runtime_state),
        }
        for key, value in task_runtime_state.items():
            payload.setdefault(key, value)
        self.memory_last = payload
        self._save_payload(payload, self.last_path)

    def validate_resume(
            self, checkpoint: Mapping[str, Any], *, backend_name: str,
            model_name: str, task_name: str, task_schema_version: int,
            task_identity: Mapping[str, Any], random_state: int,
            data_signature: Mapping[str, Any], config: Mapping[str, Any],
            compatible_config_fields: tuple[str, ...],
            config_defaults: Mapping[str, Any] | None = None,
        ) -> None:
        """Reject incompatible task, data, seed, and training state."""
        if checkpoint.get("checkpoint_kind") != "last":
            raise ValueError(
                "--torch-resume-from must point to a last checkpoint."
            )
        if checkpoint.get("format_version") != self.format_version:
            raise ValueError(
                "Unsupported Torch checkpoint format version: "
                f"{checkpoint.get('format_version')!r}; expected "
                f"{self.format_version}."
            )
        if checkpoint.get("backend_name") != backend_name:
            raise ValueError("Resume checkpoint backend does not match.")
        if checkpoint.get("classifier_name") != model_name:
            raise ValueError(f"Resume checkpoint is not for {model_name}.")
        checkpoint_task = checkpoint.get("task_name")
        if checkpoint_task is not None and checkpoint_task != task_name:
            raise ValueError("Resume checkpoint task identity does not match.")
        checkpoint_schema = checkpoint.get("task_schema_version")
        if (
            checkpoint_schema is not None
            and int(checkpoint_schema) != task_schema_version
        ):
            raise ValueError("Resume checkpoint task schema does not match.")
        stored_task_identity = checkpoint.get("task_identity")
        if (
            stored_task_identity is not None
            and stored_task_identity != dict(task_identity)
        ):
            raise ValueError("Resume checkpoint task identity does not match.")
        legacy_input_dim = checkpoint.get("input_dim")
        current_input_dim = task_identity.get("input_dim")
        if (
            stored_task_identity is None
            and legacy_input_dim is not None
            and current_input_dim is not None
            and int(legacy_input_dim) != int(current_input_dim)
        ):
            raise ValueError(
                "Resume checkpoint input dimension does not match features."
            )
        if int(checkpoint.get("random_state", -1)) != random_state:
            raise ValueError(
                "Resume checkpoint random_state does not match model seed."
            )
        if checkpoint.get("data_signature") != dict(data_signature):
            raise ValueError(
                "Resume checkpoint training/validation data do not match "
                "the current split and task inputs."
            )
        defaults = dict(config_defaults or {})
        old_config = checkpoint.get("config", {})
        changed_fields = [
            field_name
            for field_name in compatible_config_fields
            if old_config.get(field_name, defaults.get(field_name))
            != config[field_name]
        ]
        if changed_fields:
            raise ValueError(
                "Resume checkpoint training configuration differs for: "
                f"{changed_fields}"
            )
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint_epoch > int(config["max_epochs"]):
            raise ValueError(
                "torch_max_epochs must be at least the checkpoint epoch."
            )

    @property
    def resume_sha256(self) -> str | None:
        """Return the immutable resume source hash for provenance."""
        if self.resume_from is None:
            return None
        return file_sha256(self.resume_from)
