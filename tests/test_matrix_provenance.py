import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ppi_benchmark.matrix_provenance import (
    canonical_matrix,
    canonicalize_matrix,
    matrix_contract_sha256,
    matrix_records_by_source_split,
    matrix_sha256,
    nested_matrix_records,
    row_identity_sha256,
)
from ppi_benchmark.reporting.schema_compat import (
    normalize_performance_matrices,
)
from ppi_benchmark.reporting.performance import write_performance_report


def examples():
    return pd.DataFrame({
        "source_row_index": [7, 3],
        "pair_id": ["pair-a", "pair-b"],
    })


def test_dense_canonicalization_is_stable_and_avoids_recopying():
    canonical = np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype="<f8")
    fortran = np.asfortranarray(canonical)
    big_endian = canonical.astype(">f8")

    assert canonicalize_matrix(canonical) is canonical
    from_fortran = canonicalize_matrix(fortran)
    from_big_endian = canonicalize_matrix(big_endian)

    assert from_fortran.flags.c_contiguous
    assert from_big_endian.flags.c_contiguous
    assert from_big_endian.dtype.byteorder in {"<", "="}
    assert matrix_sha256(canonical) == matrix_sha256(from_fortran)
    assert matrix_sha256(canonical) == matrix_sha256(from_big_endian)
    assert matrix_sha256(canonical) != matrix_sha256(
        canonical.astype("<f4"))
    assert matrix_sha256(canonical) != matrix_sha256(
        canonical[:, :1].copy())
    changed = canonical.copy()
    changed[0, 0] = 9.0
    assert matrix_sha256(canonical) != matrix_sha256(changed)


@pytest.mark.parametrize("shape", ((0, 3), (2, 0), (0, 0)))
def test_empty_dense_matrices_have_stable_hashes(shape):
    matrix = np.empty(shape, dtype="<f8")

    assert matrix_sha256(matrix) == matrix_sha256(matrix.copy())


def test_sparse_canonicalization_and_hashing_do_not_densify(monkeypatch):
    noncanonical = sparse.coo_matrix((
        np.asarray([1.0, 2.0, -3.0, 4.0, 0.0]),
        (
            np.asarray([0, 0, 0, 1, 1]),
            np.asarray([1, 1, 1, 0, 1]),
        ),
    ), shape=(2, 2))
    canonical = canonicalize_matrix(noncanonical)
    equivalent = sparse.csr_matrix(
        np.asarray([[0.0, 0.0], [4.0, 0.0]]))
    equivalent = canonicalize_matrix(equivalent)

    assert sparse.isspmatrix_csr(canonical)
    assert canonical.has_canonical_format
    assert canonical.has_sorted_indices
    assert canonical.nnz == 1
    assert canonicalize_matrix(canonical) is canonical
    monkeypatch.setattr(
        sparse.csr_matrix,
        "toarray",
        lambda self: pytest.fail("matrix hashing must not densify CSR"),
    )

    assert matrix_sha256(canonical) == matrix_sha256(equivalent)
    changed = equivalent.copy()
    changed.data[0] = 5.0
    assert matrix_sha256(canonical) != matrix_sha256(changed)


def test_sparse_canonicalization_handles_strided_and_read_only_storage():
    backing = np.asarray([1.0, 99.0, 2.0, 99.0])
    strided = sparse.csr_matrix(
        (
            backing[::2],
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
        ),
        shape=(2, 2),
        copy=False,
    )
    assert not strided.data.flags.c_contiguous

    canonical = canonicalize_matrix(strided)

    assert canonical.data.flags.c_contiguous
    assert matrix_sha256(canonical) == matrix_sha256(
        sparse.csr_matrix(np.diag([1.0, 2.0]))
    )

    read_only = sparse.csr_matrix(
        (
            np.asarray([1.0, 0.0, 2.0]),
            np.asarray([0, 1, 1], dtype=np.int32),
            np.asarray([0, 2, 3], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    read_only.data.flags.writeable = False
    read_only.indices.flags.writeable = False
    read_only.indptr.flags.writeable = False

    cleaned = canonicalize_matrix(read_only)

    assert cleaned.nnz == 2
    assert matrix_sha256(cleaned) == matrix_sha256(canonical)


def test_row_identity_hash_requires_stable_unique_identifiers():
    identity_kind, first_hash = row_identity_sha256(
        examples(), task_example_id_columns=("pair_id",))
    _, reordered_hash = row_identity_sha256(
        examples().iloc[::-1], task_example_id_columns=("pair_id",))

    assert identity_kind == "source_row_index"
    assert first_hash != reordered_hash
    without_source = examples().drop(columns="source_row_index")
    fallback_kind, fallback_hash = row_identity_sha256(
        without_source, task_example_id_columns=("pair_id",))
    assert fallback_kind == "pair_id"
    assert fallback_hash != first_hash

    with pytest.raises(ValueError, match="require unique source_row_index"):
        row_identity_sha256(
            without_source.drop(columns="pair_id"),
            task_example_id_columns=("pair_id",),
        )
    with pytest.raises(ValueError, match="must be unique"):
        row_identity_sha256(
            examples().assign(source_row_index=[1, 1]),
            task_example_id_columns=("pair_id",),
        )
    with pytest.raises(ValueError, match="null"):
        row_identity_sha256(
            examples().assign(source_row_index=[1, None]),
            task_example_id_columns=("pair_id",),
        )


@pytest.mark.parametrize(
    "field",
    (
        "matrix_schema_id",
        "pair_composition_schema_id",
        "feature_spec_sha256",
        "fitted_extractor_sha256",
        "encoder_fingerprint",
        "training_fit_policy",
        "split_transformation_policy",
        "positive_graph_sha256",
        "training_cohort_sha256",
    ),
)
def test_each_construction_contract_identity_changes_its_hash(field):
    contract = {
        "matrix_schema_id": "schema-v1",
        "pair_composition_schema_id": "pair-v1",
        "feature_spec_sha256": "feature-a",
        "fitted_extractor_sha256": "extractor-a",
        "encoder_fingerprint": None,
        "training_fit_policy": "train-only",
        "split_transformation_policy": "transform",
        "positive_graph_sha256": "graph-a",
        "training_cohort_sha256": "training-a",
    }
    changed = dict(contract)
    changed[field] = "changed"

    assert matrix_contract_sha256(contract) != matrix_contract_sha256(changed)


def test_canonical_matrix_records_contract_rows_and_non_persistence():
    matrix = np.asarray([[1.0], [2.0]], dtype=np.float64)
    contract = {"schema": "example", "policy": "train-only"}

    artifact = canonical_matrix(
        matrix,
        examples(),
        matrix_source="configured_features",
        split_name="train",
        matrix_schema_id="ppi.configured_features.v1",
        pair_composition_schema_id="ppi.sum_absdiff_product.v1",
        construction_contract=contract,
        task_example_id_columns=("pair_id",),
    )

    assert artifact.matrix is matrix
    assert artifact.metadata["matrix_contract"] == contract
    assert len(artifact.metadata["matrix_contract_sha256"]) == 64
    assert len(artifact.metadata["row_identity_sha256"]) == 64
    assert len(artifact.metadata["matrix_sha256"]) == 64
    assert artifact.metadata["matrix_persisted"] is False


def test_canonical_matrix_snapshots_nested_contract_values():
    contract = {"schema": {"columns": ["a", "b"]}}
    artifact = canonical_matrix(
        np.asarray([[1.0], [2.0]]),
        examples(),
        matrix_source="configured_features",
        split_name="train",
        matrix_schema_id="ppi.configured_features.v1",
        pair_composition_schema_id="ppi.sum_absdiff_product.v1",
        construction_contract=contract,
        task_example_id_columns=("pair_id",),
    )
    original_hash = artifact.metadata["matrix_contract_sha256"]

    contract["schema"]["columns"].append("changed")

    assert artifact.metadata["matrix_contract"] == {
        "schema": {"columns": ["a", "b"]}
    }
    assert artifact.metadata["matrix_contract_sha256"] == original_hash


def test_performance_writer_rejects_reusing_an_immutable_output(tmp_path):
    output_path = tmp_path / "performance.jsonl"
    base_matrix = {
        "matrix_source": "configured_features",
        "split": "train",
        "matrix_contract_sha256": "contract",
        "row_identity_sha256": "rows",
        "matrix_sha256": "values-a",
    }
    write_performance_report(
        {
            "evaluation_schema_version": 3,
            "execution_id": "first",
            "matrices": {"configured_features": {"train": base_matrix}},
        },
        output_path,
    )

    with pytest.raises(ValueError, match="already exists"):
        write_performance_report(
            {
                "evaluation_schema_version": 3,
                "execution_id": "second",
                "matrices": {},
            },
            output_path,
        )
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


def test_matrix_record_normalization_handles_schema_v2_and_v3():
    record = {
        "matrix_source": "configured_features",
        "split": "train",
        "matrix_contract_sha256": "contract",
        "row_identity_sha256": "rows",
        "matrix_sha256": "values",
    }
    flat = {"train": record}
    nested = {"configured_features": {"train": record}}

    assert matrix_records_by_source_split(flat, schema_version=2) == {
        ("configured_features", "train"): record
    }
    assert matrix_records_by_source_split(nested, schema_version=3) == {
        ("configured_features", "train"): record
    }
    assert nested_matrix_records(flat, schema_version=2) == nested


def test_schema_v1_matrix_normalization_infers_legacy_names():
    normalized = normalize_performance_matrices(
        {
            "degree_train": {"shape": [4, 3]},
            "val": {"shape": [2, 12]},
        },
        schema_version=1,
    )

    degree = normalized["training_degree"]["train"]
    configured = normalized["configured_features"]["val"]
    assert degree["matrix_source"] == "training_degree"
    assert degree["split"] == "train"
    assert degree["matrix_sha256"] is None
    assert configured["matrix_source"] == "configured_features"
    assert configured["split"] == "val"
    assert configured["matrix_contract_sha256"] is None


def test_matrix_normalization_rejects_invalid_version_or_missing_v2_keys():
    with pytest.raises(ValueError, match="matrix_source and split"):
        normalize_performance_matrices(
            {"train": {"shape": [4, 3]}},
            schema_version=2,
        )
    with pytest.raises(ValueError, match="Unsupported evaluation schema"):
        normalize_performance_matrices({}, schema_version=4)
