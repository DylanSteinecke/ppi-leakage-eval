"""Canonical matrix construction and exact-byte provenance."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


HASH_CHUNK_BYTES = 8 * 1024 * 1024
MATRIX_HASH_DOMAIN = b"ppi-benchmark.matrix-bytes.v1"
ROW_HASH_DOMAIN = b"ppi-benchmark.matrix-rows.v1"
CONTRACT_HASH_DOMAIN = b"ppi-benchmark.matrix-contract.v1"


@dataclass(frozen=True)
class CanonicalMatrix:
    """The exact model input and its auditable provenance."""

    matrix: Any
    metadata: Mapping[str, Any]


class _FieldHasher:
    """Domain-separated, length-delimited streaming SHA-256 helper."""

    def __init__(self, domain: bytes):
        self._hasher = hashlib.sha256()
        self.add_bytes("domain", domain)

    def add_bytes(self, name: str, payload: Any) -> None:
        """Hash one named bytes-like payload without materializing a copy."""
        name_bytes = name.encode("utf-8")
        view = memoryview(payload).cast("B")
        self._hasher.update(struct.pack("<Q", len(name_bytes)))
        self._hasher.update(name_bytes)
        self._hasher.update(struct.pack("<Q", view.nbytes))
        for start in range(0, view.nbytes, HASH_CHUNK_BYTES):
            self._hasher.update(view[start:start + HASH_CHUNK_BYTES])

    def add_int64_values(self, name: str, values: Any) -> None:
        """Hash integers as fixed-width little-endian chunks."""
        array = np.asarray(values)
        n_values = int(array.size)
        name_bytes = name.encode("utf-8")
        self._hasher.update(struct.pack("<Q", len(name_bytes)))
        self._hasher.update(name_bytes)
        self._hasher.update(struct.pack("<Q", n_values * 8))
        values_per_chunk = max(1, HASH_CHUNK_BYTES // 8)
        flat = array.reshape(-1)
        for start in range(0, n_values, values_per_chunk):
            chunk = np.asarray(
                flat[start:start + values_per_chunk],
                dtype="<i8",
                order="C",
            )
            self._hasher.update(memoryview(chunk).cast("B"))

    def hexdigest(self) -> str:
        """Return the final hexadecimal digest."""
        return self._hasher.hexdigest()


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
        if not _is_little_endian(canonical.dtype):
            canonical = canonical.astype(
                canonical.dtype.newbyteorder("<"), copy=True)
        canonical.sum_duplicates()
        canonical.sort_indices()
        canonical.eliminate_zeros()
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
        if not matrix.has_canonical_format or not matrix.has_sorted_indices:
            raise ValueError("Sparse matrix hashing requires canonical CSR.")
        if np.count_nonzero(matrix.data) != matrix.nnz:
            raise ValueError("Sparse matrix hashing forbids explicit zeros.")
        if not _is_little_endian(matrix.dtype):
            raise ValueError("Sparse matrix hashing requires little-endian data.")
        hasher = _FieldHasher(MATRIX_HASH_DOMAIN)
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
    if not _is_little_endian(array.dtype):
        raise ValueError("Dense matrix hashing requires little-endian data.")
    hasher = _FieldHasher(MATRIX_HASH_DOMAIN)
    hasher.add_bytes("format", b"dense")
    hasher.add_bytes(
        "dtype", _canonical_dtype_name(array.dtype).encode("ascii"))
    hasher.add_int64_values("shape", array.shape)
    hasher.add_bytes("data", memoryview(array).cast("B"))
    return hasher.hexdigest()


def stable_row_identities(
    examples: pd.DataFrame,
    task_example_id_columns: Sequence[str],
) -> tuple[str, list[Any]]:
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
        return column, values.tolist()
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
    hasher = _FieldHasher(ROW_HASH_DOMAIN)
    hasher.add_bytes("identity_kind", identity_kind.encode("utf-8"))
    for value in values:
        encoded = json.dumps(
            value.item() if isinstance(value, np.generic) else value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        hasher.add_bytes("row_identity", encoded)
    return identity_kind, hasher.hexdigest()


def matrix_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash a complete matrix construction contract."""
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    hasher = _FieldHasher(CONTRACT_HASH_DOMAIN)
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
    contract = dict(construction_contract)
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


def matrix_records_by_contract(
    matrices: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Index serialized matrix records for append compatibility checks."""
    indexed = {}
    for name, metadata in matrices.items():
        contract_hash = metadata.get("matrix_contract_sha256")
        matrix_source = metadata.get("matrix_source")
        split_name = metadata.get("split")
        if not contract_hash or not matrix_source or not split_name:
            continue
        key = (str(matrix_source), str(split_name), str(contract_hash))
        if key in indexed and indexed[key] != metadata:
            raise ValueError(f"Duplicate incompatible matrix record {name!r}.")
        indexed[key] = metadata
    return indexed
