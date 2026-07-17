"""Task-neutral runtime and resource reporting helpers."""

from __future__ import annotations

import json
import resource
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy import sparse

from ..schema import EVALUATION_SCHEMA_VERSION
from ..artifact_io import output_lock


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

    def add_matrices(self, matrices: dict[str, Any]) -> None:
        """Record statistics for non-empty feature matrices."""
        for name, matrix in matrices.items():
            statistics = matrix_statistics(matrix)
            if statistics is not None:
                self.matrices[name] = statistics

    def add_model_runs(self, model_runs: list[dict[str, Any]]) -> None:
        """Append model-level timing and solver observations."""
        self.model_runs.extend(model_runs)

    def add_observation(self, name: str, value: Any) -> None:
        """Record one JSON-compatible invocation-level observation."""
        self.observations[name] = value

    def report(
            self, execution_id: str, task: str | None = None,
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
        }


def append_performance_report(
        report: dict[str, Any], output_path: Path,
    ) -> None:
    """Append one performance record as thread-safe JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True) + "\n"
    with output_lock(output_path):
        with output_path.open("a", encoding="utf-8") as fout:
            fout.write(serialized)
