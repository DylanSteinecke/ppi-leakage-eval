"""Shared hashing, locking, and writing for benchmark artifacts."""

import fcntl
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd


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


def reset_output_file(output_path: Path, append_results: bool) -> None:
    """Start with a clean output file unless append mode is requested."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not append_results and output_path.exists():
        output_path.unlink()
