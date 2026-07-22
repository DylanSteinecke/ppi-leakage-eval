"""Canonical matrix construction and exact-byte provenance."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .artifact_io import (
    LengthDelimitedHasher,
    canonical_json_bytes,
    canonical_json_snapshot,
)

MATRIX_HASH_DOMAIN = b"ppi-benchmark.matrix-bytes.v1"
ROW_HASH_DOMAIN = b"ppi-benchmark.matrix-rows.v1"
CONTRACT_HASH_DOMAIN = b"ppi-benchmark.matrix-contract.v1"


@dataclass(frozen=True)
class CanonicalMatrix:
    """The exact model input and its auditable provenance."""

    matrix: Any
    metadata: Mapping[str, Any]


def _is_little_endian(dtype: np.dtype[Any]) -> bool:
    return (
        dtype.byteorder in {"<", "|"}
        or (dtype.byteorder == "=" and sys.byteorder == "little")
    )


def _require_numeric_matrix_dtype(dtype: np.dtype[Any]) -> None:
    if not (
        np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.bool_)
    ):
        raise TypeError(f"Feature matrices must be numeric, not {dtype}.")


def canonicalize_matrix(matrix: Any) -> Any:
    """Return canonical dense or CSR storage suitable for model input."""
    if sparse.issparse(matrix):
        canonical = matrix.tocsr(copy=False)
        _require_numeric_matrix_dtype(canonical.dtype)
        has_explicit_zeros = np.count_nonzero(canonical.data) != canonical.nnz
        needs_cleanup = (
            not canonical.has_canonical_format
            or not canonical.has_sorted_indices
            or has_explicit_zeros
        )
        writable = all(
            values.flags.writeable
            for values in (
                canonical.data,
                canonical.indices,
                canonical.indptr,
            )
        )
        if needs_cleanup:
            if not writable:
                canonical = canonical.copy()
            if not _is_little_endian(canonical.dtype):
                canonical = canonical.astype(
                    canonical.dtype.newbyteorder("<"), copy=True)
            canonical.sum_duplicates()
            canonical.sort_indices()
            canonical.eliminate_zeros()
        if (
            not _is_little_endian(canonical.dtype)
            or not canonical.data.flags.c_contiguous
        ):
            data = np.ascontiguousarray(
                canonical.data.astype(
                    canonical.dtype.newbyteorder("<"), copy=False)
            )
            canonical = sparse.csr_matrix(
                (data, canonical.indices, canonical.indptr),
                shape=canonical.shape,
                copy=False,
            )
        return canonical

    canonical = np.asarray(matrix)
    if canonical.ndim != 2:
        raise ValueError("Feature matrices must be two-dimensional.")
    _require_numeric_matrix_dtype(canonical.dtype)
    if not _is_little_endian(canonical.dtype):
        canonical = canonical.astype(
            canonical.dtype.newbyteorder("<"), copy=True)
    if not canonical.flags.c_contiguous:
        canonical = np.ascontiguousarray(canonical)
    return canonical


def _canonical_dtype_name(dtype: np.dtype[Any]) -> str:
    return dtype.newbyteorder("<").str


def matrix_sha256(matrix: Any) -> str:
    """Hash one canonical matrix as an exact-byte reproducibility check."""
    if sparse.issparse(matrix):
        if not sparse.isspmatrix_csr(matrix):
            raise ValueError("Sparse matrix hashing requires canonical CSR.")
        _require_numeric_matrix_dtype(matrix.dtype)
        if not matrix.has_canonical_format or not matrix.has_sorted_indices:
            raise ValueError("Sparse matrix hashing requires canonical CSR.")
        if np.count_nonzero(matrix.data) != matrix.nnz:
            raise ValueError("Sparse matrix hashing forbids explicit zeros.")
        if not _is_little_endian(matrix.dtype):
            raise ValueError("Sparse matrix hashing requires little-endian data.")
        if not matrix.data.flags.c_contiguous:
            raise ValueError("Sparse matrix hashing requires contiguous data.")
        hasher = LengthDelimitedHasher(MATRIX_HASH_DOMAIN)
        hasher.add_bytes("format", b"csr")
        hasher.add_bytes(
            "dtype", _canonical_dtype_name(matrix.dtype).encode("ascii"))
        hasher.add_int64_values("shape", matrix.shape)
        hasher.add_bytes("data", memoryview(matrix.data).cast("B"))
        hasher.add_int64_values("indices", matrix.indices)
        hasher.add_int64_values("indptr", matrix.indptr)
        return hasher.hexdigest()

    array = np.asarray(matrix)
    if array.ndim != 2 or not array.flags.c_contiguous:
        raise ValueError("Dense matrix hashing requires C-contiguous storage.")
    _require_numeric_matrix_dtype(array.dtype)
    if not _is_little_endian(array.dtype):
        raise ValueError("Dense matrix hashing requires little-endian data.")
    hasher = LengthDelimitedHasher(MATRIX_HASH_DOMAIN)
    hasher.add_bytes("format", b"dense")
    hasher.add_bytes(
        "dtype", _canonical_dtype_name(array.dtype).encode("ascii"))
    hasher.add_int64_values("shape", array.shape)
    hasher.add_bytes("data", memoryview(array.reshape(-1)).cast("B"))
    return hasher.hexdigest()


def stable_row_identities(
    examples: pd.DataFrame,
    task_example_id_columns: Sequence[str],
) -> tuple[str, pd.Series]:
    candidates = ("source_row_index", *task_example_id_columns)
    for column in candidates:
        if column not in examples.columns:
            continue
        values = examples[column]
        if values.isna().any():
            raise ValueError(
                f"Stable matrix row identity {column!r} contains null values."
            )
        if values.duplicated().any():
            raise ValueError(
                f"Stable matrix row identity {column!r} must be unique."
            )
        return column, values
    raise ValueError(
        "Matrix rows require unique source_row_index values or a unique "
        f"task example ID from {tuple(task_example_id_columns)!r}."
    )


def row_identity_sha256(
    examples: pd.DataFrame,
    *,
    task_example_id_columns: Sequence[str] = ("example_id",),
) -> tuple[str, str]:
    """Hash stable example identities in exact matrix-row order."""
    identity_kind, values = stable_row_identities(
        examples, task_example_id_columns)
    hasher = LengthDelimitedHasher(ROW_HASH_DOMAIN)
    hasher.add_bytes("identity_kind", identity_kind.encode("utf-8"))
    for value in values:
        encoded = canonical_json_bytes(
            value.item() if isinstance(value, np.generic) else value
        )
        hasher.add_bytes("row_identity", encoded)
    return identity_kind, hasher.hexdigest()


def matrix_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash a complete matrix construction contract."""
    payload = canonical_json_bytes(contract)
    hasher = LengthDelimitedHasher(CONTRACT_HASH_DOMAIN)
    hasher.add_bytes("contract", payload)
    return hasher.hexdigest()


def canonical_matrix(
    matrix: Any,
    examples: pd.DataFrame,
    *,
    matrix_source: str,
    split_name: str,
    matrix_schema_id: str,
    pair_composition_schema_id: str | None,
    construction_contract: Mapping[str, Any],
    task_example_id_columns: Sequence[str] = ("example_id",),
) -> CanonicalMatrix:
    """Canonicalize one model input and attach hashes without persisting it."""
    canonical = canonicalize_matrix(matrix)
    if canonical.shape[0] != len(examples):
        raise ValueError(
            f"{matrix_source} {split_name} matrix has {canonical.shape[0]} "
            f"rows for {len(examples)} examples."
        )
    identity_kind, row_hash = row_identity_sha256(
        examples,
        task_example_id_columns=task_example_id_columns,
    )
    contract = canonical_json_snapshot(construction_contract)
    return CanonicalMatrix(
        matrix=canonical,
        metadata={
            "matrix_source": matrix_source,
            "split": split_name,
            "matrix_schema_id": matrix_schema_id,
            "pair_composition_schema_id": pair_composition_schema_id,
            "matrix_contract_sha256": matrix_contract_sha256(contract),
            "matrix_contract": contract,
            "row_identity_kind": identity_kind,
            "row_identity_sha256": row_hash,
            "matrix_sha256": matrix_sha256(canonical),
            "matrix_persisted": False,
            "matrix_hash_semantics": "exact_canonical_bytes",
        },
    )


def matrix_records_by_source_split(
    matrices: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index serialized matrix records for append compatibility checks."""
    indexed = {}
    for name, metadata in matrices.items():
        contract_hash = metadata.get("matrix_contract_sha256")
        matrix_source = metadata.get("matrix_source")
        split_name = metadata.get("split")
        if not contract_hash or not matrix_source or not split_name:
            continue
        key = (str(matrix_source), str(split_name))
        if key in indexed and indexed[key] != metadata:
            raise ValueError(f"Duplicate incompatible matrix record {name!r}.")
        indexed[key] = metadata
    return indexed
