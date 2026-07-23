"""Transactional, content-addressed storage for protein embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .base import EncoderSpec


CACHE_SCHEMA_VERSION = 1
SQLITE_QUERY_CHUNK_SIZE = 500


def default_embedding_cache_dir() -> Path:
    """Return the shared user cache used across benchmark run directories."""
    explicit_path = (
        os.environ.get("PROTEIN_BENCHMARK_EMBEDDING_CACHE_DIR")
        or os.environ.get("PPI_EMBEDDING_CACHE_DIR")
    )
    if explicit_path:
        return Path(explicit_path).expanduser()
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    return cache_root / "ppi-leakage" / "protein_embeddings"


@dataclass(frozen=True)
class CachedEmbedding:
    """One decoded cache record."""

    embedding: np.ndarray
    sequence_length: int


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


class EmbeddingCache:
    """SQLite-backed embedding cache isolated by ``EncoderSpec``."""

    def __init__(
            self, encoder_spec: EncoderSpec,
            cache_dir: str | Path | None = None,
        ):
        self.encoder_spec = encoder_spec
        self.cache_dir = (
            default_embedding_cache_dir()
            if cache_dir is None
            else Path(cache_dir).expanduser()
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / f"{encoder_spec.fingerprint}.sqlite3"
        self._connection = sqlite3.connect(self.path, timeout=60.0)
        self._connection.execute("PRAGMA busy_timeout = 60000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        try:
            self._initialize_schema()
        except Exception:
            self._connection.close()
            raise

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    sequence_sha256 TEXT PRIMARY KEY,
                    sequence_length INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            expected_metadata = {
                "cache_schema_version": str(CACHE_SCHEMA_VERSION),
                "encoder_fingerprint": self.encoder_spec.fingerprint,
                "encoder_spec": self.encoder_spec.canonical_json,
            }
            existing_metadata = dict(self._connection.execute(
                "SELECT key, value FROM cache_metadata"
            ).fetchall())
            if existing_metadata:
                mismatches = [
                    key
                    for key, value in expected_metadata.items()
                    if existing_metadata.get(key) != value
                ]
                if mismatches:
                    raise ValueError(
                        f"Embedding cache metadata mismatch at {self.path}: "
                        f"{mismatches}"
                    )
            else:
                self._connection.executemany(
                    "INSERT INTO cache_metadata(key, value) VALUES (?, ?)",
                    expected_metadata.items(),
                )

    def get_many(
            self, sequence_hashes: Iterable[str],
        ) -> dict[str, CachedEmbedding]:
        """Return all valid records matching the requested sequence hashes."""
        requested = list(dict.fromkeys(sequence_hashes))
        records: dict[str, CachedEmbedding] = {}
        for chunk in _chunks(requested, SQLITE_QUERY_CHUNK_SIZE):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                "SELECT sequence_sha256, sequence_length, dtype, "
                "shape_json, payload, payload_sha256 FROM embeddings "
                f"WHERE sequence_sha256 IN ({placeholders})",
                chunk,
            ).fetchall()
            for (
                sequence_hash,
                sequence_length,
                dtype_name,
                shape_json,
                payload,
                expected_payload_hash,
            ) in rows:
                payload_bytes = bytes(payload)
                observed_payload_hash = hashlib.sha256(
                    payload_bytes
                ).hexdigest()
                if observed_payload_hash != expected_payload_hash:
                    raise ValueError(
                        f"Corrupt embedding payload for {sequence_hash} "
                        f"in {self.path}."
                    )
                shape = tuple(int(value) for value in json.loads(shape_json))
                embedding = np.frombuffer(
                    payload_bytes,
                    dtype=np.dtype(dtype_name),
                )
                expected_elements = int(np.prod(shape, dtype=np.int64))
                if embedding.size != expected_elements:
                    raise ValueError(
                        f"Invalid embedding shape for {sequence_hash} "
                        f"in {self.path}."
                    )
                records[sequence_hash] = CachedEmbedding(
                    embedding=embedding.reshape(shape),
                    sequence_length=int(sequence_length),
                )
        return records

    def put_many(
            self, records: Mapping[str, CachedEmbedding],
        ) -> None:
        """Insert a batch atomically, preserving any concurrent prior entry."""
        if not records:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for sequence_hash, record in records.items():
            embedding = np.ascontiguousarray(record.embedding)
            if record.sequence_length < 1:
                raise ValueError("Cached sequence lengths must be positive.")
            if embedding.ndim < 1 or embedding.size < 1:
                raise ValueError("Cached embeddings must not be empty.")
            if not np.issubdtype(embedding.dtype, np.floating):
                raise ValueError("Cached embeddings must use a floating dtype.")
            if not np.isfinite(embedding).all():
                raise ValueError("Cached embeddings must contain finite values.")
            payload = embedding.tobytes(order="C")
            rows.append((
                sequence_hash,
                int(record.sequence_length),
                embedding.dtype.str,
                json.dumps(list(embedding.shape), separators=(",", ":")),
                sqlite3.Binary(payload),
                hashlib.sha256(payload).hexdigest(),
                timestamp,
            ))
        with self._connection:
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO embeddings(
                    sequence_sha256,
                    sequence_length,
                    dtype,
                    shape_json,
                    payload,
                    payload_sha256,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def count(self) -> int:
        """Return the number of cached unique sequences."""
        row = self._connection.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
