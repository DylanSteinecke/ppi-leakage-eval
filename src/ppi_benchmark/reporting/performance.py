"""Task-neutral runtime and resource reporting helpers."""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from ..schema import EVALUATION_SCHEMA_VERSION
from ..artifact_io import output_lock
from ..matrix_provenance import matrix_records_by_source_split


PERFORMANCE_FILENAME = "performance.jsonl"


def peak_memory_bytes() -> int:
    """Return this process's maximum resident-set size in bytes."""
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(maximum_rss)
    return int(maximum_rss * 1024)


def matrix_statistics(matrix: Any) -> dict[str, Any] | None:
    """Return JSON-friendly shape, density, and storage statistics."""
    if matrix is None or not hasattr(matrix, "shape"):
        return None
    shape = tuple(int(dimension) for dimension in matrix.shape)
    if len(shape) != 2:
        return None
    n_rows, n_columns = shape
    n_elements = n_rows * n_columns

    if sparse.issparse(matrix):
        n_nonzero = int(matrix.nnz)
        storage_bytes = int(
            matrix.data.nbytes
            + matrix.indices.nbytes
            + matrix.indptr.nbytes
        )
        matrix_format = matrix.getformat()
        dtype = str(matrix.dtype)
    elif isinstance(matrix, pd.DataFrame):
        n_nonzero = None
        storage_bytes = int(matrix.memory_usage(index=True, deep=True).sum())
        matrix_format = "dataframe"
        dtype = None
    else:
        array = np.asarray(matrix)
        n_nonzero = int(np.count_nonzero(array))
        storage_bytes = int(array.nbytes)
        matrix_format = "dense"
        dtype = str(array.dtype)

    density = (
        None
        if n_nonzero is None or n_elements == 0
        else float(n_nonzero / n_elements)
    )
    return {
        "shape": list(shape),
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_elements": n_elements,
        "n_nonzero": n_nonzero,
        "density": density,
        "storage_bytes": storage_bytes,
        "storage_mib": float(storage_bytes / (1024 ** 2)),
        "format": matrix_format,
        "dtype": dtype,
    }


def solver_iteration_report(model: Any) -> dict[str, Any] | None:
    """Return normalized solver iteration counts from an sklearn estimator."""
    raw_iterations = getattr(model, "n_iter_", None)
    if raw_iterations is None:
        return None
    values = np.asarray(raw_iterations).reshape(-1)
    if not len(values):
        return None
    iterations = [int(value) for value in values]
    return {
        "values": iterations,
        "maximum": max(iterations),
        "total": sum(iterations),
    }


def runtime_provenance(
    *,
    configured_precisions: dict[str, str | None],
    requested_devices: dict[str, str | None],
    observed_devices: dict[str, Any],
) -> dict[str, Any]:
    """Return runtime, library, and observed hardware provenance."""
    packages = {}
    for label, distribution in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit_learn", "scikit-learn"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    ):
        try:
            packages[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[label] = None

    accelerator_hardware: dict[str, Any] = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        accelerator_hardware["cuda_available"] = bool(
            torch.cuda.is_available())
        accelerator_hardware["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        accelerator_hardware["mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "packages": packages,
        "configured_precisions": configured_precisions,
        "requested_devices": requested_devices,
        "observed_devices": observed_devices,
        "accelerator_hardware": accelerator_hardware,
    }


@dataclass
class PerformanceTracker:
    """Accumulate named stage durations and model-level observations."""

    started_at: float = field(default_factory=perf_counter)
    stages_seconds: dict[str, float] = field(default_factory=dict)
    matrices: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_runs: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one possibly repeated pipeline stage."""
        stage_started_at = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - stage_started_at
            self.stages_seconds[name] = (
                self.stages_seconds.get(name, 0.0) + elapsed
            )

    def add_matrices(
        self,
        matrices: dict[str, Any],
        metadata: dict[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Record statistics and provenance for non-empty matrices."""
        metadata = metadata or {}
        for name, matrix in matrices.items():
            statistics = matrix_statistics(matrix)
            if statistics is not None:
                self.matrices[name] = {
                    **statistics,
                    **dict(metadata.get(name, {})),
                }

    def add_model_runs(self, model_runs: list[dict[str, Any]]) -> None:
        """Append model-level timing and solver observations."""
        self.model_runs.extend(model_runs)

    def add_observation(self, name: str, value: Any) -> None:
        """Record one JSON-compatible invocation-level observation."""
        self.observations[name] = value

    def report(
            self, execution_id: str, task: str | None = None,
            runtime: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
        """Return the complete invocation-level performance record."""
        peak_bytes = peak_memory_bytes()
        return {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "task": task,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_id": execution_id,
            "total_seconds": float(perf_counter() - self.started_at),
            "stages_seconds": {
                name: float(seconds)
                for name, seconds in self.stages_seconds.items()
            },
            "peak_memory_bytes": peak_bytes,
            "peak_memory_mib": float(peak_bytes / (1024 ** 2)),
            "matrices": self.matrices,
            "model_runs": self.model_runs,
            "observations": self.observations,
            "runtime_provenance": runtime,
        }


def _validate_append_compatibility(
    report: Mapping[str, Any],
    existing_reports: Iterator[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Require one schema and exact matrices within an append run."""
    incoming_version = report.get("evaluation_schema_version")
    if incoming_version is None:
        raise ValueError("Performance reports require evaluation_schema_version.")
    incoming = matrix_records_by_source_split(report.get("matrices", {}))
    for existing_report in existing_reports:
        existing_version = existing_report.get("evaluation_schema_version")
        if existing_version != incoming_version:
            raise ValueError(
                f"Cannot append to {output_path.parent}: evaluation schema "
                f"versions differ ({existing_version!r} != "
                f"{incoming_version!r})."
            )
        existing = matrix_records_by_source_split(
            existing_report.get("matrices", {}))
        for key in incoming.keys() & existing.keys():
            incoming_record = incoming[key]
            existing_record = existing[key]
            for hash_field in (
                "matrix_contract_sha256",
                "row_identity_sha256",
                "matrix_sha256",
            ):
                if incoming_record.get(hash_field) != existing_record.get(
                    hash_field
                ):
                    raise ValueError(
                        f"Cannot append to {output_path.parent}: matrix "
                        f"{key} has a different {hash_field}."
                    )


def _performance_reports(output_path: Path) -> Iterator[Mapping[str, Any]]:
    if not output_path.exists():
        return
    with output_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def validate_performance_append(
    *,
    evaluation_schema_version: int,
    matrices: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Preflight append compatibility before model artifacts are written."""
    if not output_path.exists():
        return
    report = {
        "evaluation_schema_version": evaluation_schema_version,
        "matrices": matrices,
    }
    with output_lock(output_path):
        _validate_append_compatibility(
            report,
            _performance_reports(output_path),
            output_path,
        )


def append_performance_report(
        report: dict[str, Any], output_path: Path,
    ) -> None:
    """Append one performance record as thread-safe JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True) + "\n"
    with output_lock(output_path):
        _validate_append_compatibility(
            report,
            _performance_reports(output_path),
            output_path,
        )
        with output_path.open("a", encoding="utf-8") as fout:
            fout.write(serialized)
