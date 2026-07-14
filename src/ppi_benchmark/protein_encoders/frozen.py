"""Deduplicated, token-budgeted encoding through a persistent cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .base import (
    ProteinEncoder,
    normalize_protein_sequence,
    sequence_sha256,
)
from .cache import CachedEmbedding, EmbeddingCache


@dataclass(frozen=True)
class ProteinEmbeddingTable:
    """Protein-aligned embedding rows and auditable cache statistics."""

    protein_ids: tuple[str, ...]
    embeddings: np.ndarray
    metadata: dict[str, object]


def token_budget_batches(
        token_lengths: Sequence[int], max_padded_tokens: int,
        max_sequences: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
    """Pack length-sorted rows under a padded-token budget."""
    if max_padded_tokens < 1:
        raise ValueError("max_padded_tokens must be positive.")
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("max_sequences must be positive when provided.")
    lengths = [int(length) for length in token_lengths]
    if any(length < 1 for length in lengths):
        raise ValueError("Token lengths must be positive.")
    if any(length > max_padded_tokens for length in lengths):
        longest = max(lengths)
        raise ValueError(
            f"A sequence has {longest} tokens, exceeding "
            f"max_padded_tokens={max_padded_tokens}. Increase the batch "
            "token budget or reduce the encoder maximum length."
        )
    if not lengths:
        return ()

    sorted_indices = sorted(
        range(len(lengths)),
        key=lambda index: (lengths[index], index),
    )
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_max_length = 0
    for index in sorted_indices:
        candidate_max_length = max(current_max_length, lengths[index])
        candidate_size = len(current) + 1
        exceeds_token_budget = (
            candidate_size * candidate_max_length > max_padded_tokens
        )
        exceeds_sequence_limit = (
            max_sequences is not None
            and candidate_size > max_sequences
        )
        if current and (exceeds_token_budget or exceeds_sequence_limit):
            batches.append(tuple(current))
            current = []
            current_max_length = 0
        current.append(index)
        current_max_length = max(current_max_length, lengths[index])
    if current:
        batches.append(tuple(current))
    return tuple(batches)


class FrozenProteinEncoder:
    """Encode each unique sequence once without accepting labels or splits."""

    def __init__(
            self, encoder: ProteinEncoder, cache: EmbeddingCache,
            max_batch_tokens: int, max_batch_sequences: int | None = None,
        ):
        if not encoder.spec.frozen:
            raise ValueError(
                "FrozenProteinEncoder requires an encoder with frozen=True."
            )
        if cache.encoder_spec != encoder.spec:
            raise ValueError("Encoder and embedding cache specs do not match.")
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be positive.")
        if max_batch_sequences is not None and max_batch_sequences < 1:
            raise ValueError(
                "max_batch_sequences must be positive when provided."
            )
        self.encoder = encoder
        self.cache = cache
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_sequences = max_batch_sequences

    def encode(
            self, protein_sequences: Mapping[str, str],
            protein_ids: Sequence[str] | None = None,
        ) -> ProteinEmbeddingTable:
        """Return protein-aligned embeddings with cache and batching audits."""
        selected_ids = (
            sorted(str(protein_id) for protein_id in protein_sequences)
            if protein_ids is None
            else list(dict.fromkeys(
                str(protein_id) for protein_id in protein_ids
            ))
        )
        if not selected_ids:
            raise ValueError("At least one protein ID is required for encoding.")

        hash_by_id: dict[str, str] = {}
        sequence_by_hash: dict[str, str] = {}
        for protein_id in selected_ids:
            if protein_id not in protein_sequences:
                raise ValueError(
                    f"Missing sequence for protein ID {protein_id!r}."
                )
            normalized = normalize_protein_sequence(
                protein_sequences[protein_id])
            sequence_hash = sequence_sha256(normalized)
            previous = sequence_by_hash.setdefault(sequence_hash, normalized)
            if previous != normalized:
                raise RuntimeError("Unexpected SHA-256 sequence collision.")
            hash_by_id[protein_id] = sequence_hash

        sequence_hashes = sorted(sequence_by_hash)
        cached = self.cache.get_many(sequence_hashes)
        cache_hits = len(cached)
        missing_hashes = [
            sequence_hash
            for sequence_hash in sequence_hashes
            if sequence_hash not in cached
        ]

        encoded_batches = 0
        maximum_batch_sequences = 0
        maximum_batch_padded_tokens = 0
        if missing_hashes:
            missing_sequences = [
                sequence_by_hash[sequence_hash]
                for sequence_hash in missing_hashes
            ]
            raw_token_lengths = np.asarray(
                self.encoder.token_lengths(missing_sequences)
            ).reshape(-1)
            if len(raw_token_lengths) != len(missing_sequences):
                raise ValueError(
                    "ProteinEncoder.token_lengths returned the wrong row count."
                )
            if (
                not np.issubdtype(raw_token_lengths.dtype, np.number)
                or not np.isfinite(raw_token_lengths).all()
                or not np.equal(
                    raw_token_lengths,
                    np.floor(raw_token_lengths),
                ).all()
            ):
                raise ValueError(
                    "ProteinEncoder.token_lengths must return finite integers."
                )
            token_lengths = raw_token_lengths.astype(np.int64, copy=False)
            batches = token_budget_batches(
                token_lengths=token_lengths,
                max_padded_tokens=self.max_batch_tokens,
                max_sequences=self.max_batch_sequences,
            )
            embedding_dimension = None
            for batch_indices in batches:
                batch_sequences = [
                    missing_sequences[index]
                    for index in batch_indices
                ]
                batch_embeddings = np.asarray(
                    self.encoder.encode_batch(batch_sequences)
                )
                if (
                    batch_embeddings.ndim != 2
                    or batch_embeddings.shape[0] != len(batch_indices)
                ):
                    raise ValueError(
                        "ProteinEncoder.encode_batch must return a 2D array "
                        "with one row per sequence."
                    )
                if not np.isfinite(batch_embeddings).all():
                    raise ValueError("Protein encoder returned non-finite values.")
                if embedding_dimension is None:
                    embedding_dimension = batch_embeddings.shape[1]
                elif batch_embeddings.shape[1] != embedding_dimension:
                    raise ValueError(
                        "Protein encoder embedding dimensions changed "
                        "between batches."
                    )
                new_records = {
                    missing_hashes[index]: CachedEmbedding(
                        embedding=np.ascontiguousarray(batch_embeddings[row]),
                        sequence_length=len(missing_sequences[index]),
                    )
                    for row, index in enumerate(batch_indices)
                }
                self.cache.put_many(new_records)
                persisted_records = self.cache.get_many(new_records)
                if len(persisted_records) != len(new_records):
                    raise RuntimeError(
                        "Embedding cache did not persist a complete batch."
                    )
                cached.update(persisted_records)
                encoded_batches += 1
                maximum_batch_sequences = max(
                    maximum_batch_sequences,
                    len(batch_indices),
                )
                maximum_batch_padded_tokens = max(
                    maximum_batch_padded_tokens,
                    len(batch_indices) * max(
                        int(token_lengths[index])
                        for index in batch_indices
                    ),
                )

        embedding_rows = []
        embedding_dimension = None
        for protein_id in selected_ids:
            embedding = np.asarray(
                cached[hash_by_id[protein_id]].embedding
            )
            if embedding.ndim != 1:
                raise ValueError(
                    "Pooled protein embeddings must be one-dimensional."
                )
            if embedding_dimension is None:
                embedding_dimension = embedding.shape[0]
            elif embedding.shape[0] != embedding_dimension:
                raise ValueError(
                    "Cached protein embedding dimensions are inconsistent."
                )
            embedding_rows.append(embedding)
        embeddings = np.stack(embedding_rows)
        n_unique_sequences = len(sequence_hashes)
        cache_misses = len(missing_hashes)
        metadata: dict[str, object] = {
            "encoder_fingerprint": self.encoder.spec.fingerprint,
            "encoder_spec": self.encoder.spec.to_dict(),
            "embedding_cache_path": str(self.cache.path),
            "n_proteins": len(selected_ids),
            "n_unique_sequences": n_unique_sequences,
            "n_duplicate_sequences": len(selected_ids) - n_unique_sequences,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_fraction": (
                cache_hits / n_unique_sequences
                if n_unique_sequences else 0.0
            ),
            "encoded_batches": encoded_batches,
            "max_batch_sequences": maximum_batch_sequences,
            "max_batch_padded_tokens": maximum_batch_padded_tokens,
            "configured_max_batch_tokens": self.max_batch_tokens,
            "configured_max_batch_sequences": self.max_batch_sequences,
            "embedding_dimension": int(embeddings.shape[1]),
            "embedding_dtype": str(embeddings.dtype),
        }
        return ProteinEmbeddingTable(
            protein_ids=tuple(selected_ids),
            embeddings=embeddings,
            metadata=metadata,
        )
