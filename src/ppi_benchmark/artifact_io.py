"""Shared hashing, locking, and writing for benchmark artifacts."""

import fcntl
import hashlib
import json
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


HASH_CHUNK_BYTES = 8 * 1024 * 1024


class LengthDelimitedHasher:
    """Domain-separated, length-delimited streaming SHA-256 helper."""

    def __init__(self, domain: bytes):
        self._hasher = hashlib.sha256()
        self.add_bytes("domain", domain)

    def add_bytes(self, name: str, payload: Any) -> None:
        """Hash one named contiguous bytes-like payload without copying."""
        name_bytes = name.encode("utf-8")
        view = memoryview(payload).cast("B")
        self._hasher.update(struct.pack("<Q", len(name_bytes)))
        self._hasher.update(name_bytes)
        self._hasher.update(struct.pack("<Q", view.nbytes))
        for start in range(0, view.nbytes, HASH_CHUNK_BYTES):
            self._hasher.update(view[start:start + HASH_CHUNK_BYTES])

    def add_int64_values(self, name: str, values: Any) -> None:
        """Hash integers as fixed-width little-endian bounded chunks."""
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


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value canonically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_snapshot(value: Any) -> Any:
    """Return a detached JSON-compatible snapshot with stable key order."""
    return json.loads(canonical_json_bytes(value))


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using its canonical serialization."""
    return bytes_sha256(canonical_json_bytes(value))


def bytes_sha256(data: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(input_path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(input_path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def output_lock(output_path: Path) -> Iterator[None]:
    """Lock a sidecar file before writing output."""
    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Append a dataframe to a CSV with a lock and one header row."""
    if df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output_path):
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        if not write_header:
            existing_columns = pd.read_csv(output_path, nrows=0).columns.tolist()
            if existing_columns != df.columns.tolist():
                raise ValueError(
                    f"Cannot append to {output_path}: existing columns do "
                    "not match the incoming dataframe schema."
                )
        df.to_csv(output_path, mode="a", header=write_header, index=False)


def write_dataframe_threadsafe(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a dataframe to CSV with a file lock."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output_path):
        df.to_csv(output_path, index=False)
