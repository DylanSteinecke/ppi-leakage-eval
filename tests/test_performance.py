import json
from time import sleep
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from ppi_benchmark.performance import (
    PerformanceTracker,
    append_performance_report,
    matrix_statistics,
    peak_memory_bytes,
    solver_iteration_report,
)


def test_sparse_matrix_statistics_report_shape_density_and_storage():
    matrix = csr_matrix(np.asarray([
        [1.0, 0.0, 2.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float32))

    statistics = matrix_statistics(matrix)

    assert statistics["shape"] == [2, 3]
    assert statistics["n_elements"] == 6
    assert statistics["n_nonzero"] == 3
    assert statistics["density"] == 0.5
    assert statistics["storage_bytes"] == (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    assert statistics["format"] == "csr"
    assert statistics["dtype"] == "float32"


def test_dense_matrix_statistics_report_density_and_bytes():
    matrix = np.asarray([[1, 0], [2, 3]], dtype=np.int16)

    statistics = matrix_statistics(matrix)

    assert statistics["shape"] == [2, 2]
    assert statistics["n_nonzero"] == 3
    assert statistics["density"] == 0.75
    assert statistics["storage_bytes"] == matrix.nbytes
    assert statistics["format"] == "dense"


def test_solver_iteration_report_normalizes_scalar_and_array_values():
    assert solver_iteration_report(SimpleNamespace()) is None
    assert solver_iteration_report(SimpleNamespace(n_iter_=7)) == {
        "values": [7],
        "maximum": 7,
        "total": 7,
    }
    assert solver_iteration_report(SimpleNamespace(n_iter_=[3, 5])) == {
        "values": [3, 5],
        "maximum": 5,
        "total": 8,
    }


def test_tracker_and_jsonl_writer_report_invocation_resources(tmp_path):
    tracker = PerformanceTracker()
    with tracker.stage("example_stage"):
        sleep(0.001)
    tracker.add_matrices({"train": np.ones((2, 3)), "test": None})
    tracker.add_model_runs([{"model_name": "example", "fit_seconds": 0.1}])

    report = tracker.report("execution-1")
    output_path = tmp_path / "performance.jsonl"
    append_performance_report(report, output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written["execution_id"] == "execution-1"
    assert written["total_seconds"] >= written["stages_seconds"][
        "example_stage"]
    assert written["stages_seconds"]["example_stage"] > 0.0
    assert written["matrices"]["train"]["density"] == 1.0
    assert written["model_runs"][0]["fit_seconds"] == 0.1
    assert 0 < written["peak_memory_bytes"] <= peak_memory_bytes()
