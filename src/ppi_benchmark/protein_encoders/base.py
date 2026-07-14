"""Model-neutral contracts and identities for protein sequence encoders."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


ENCODER_SPEC_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_protein_sequence(sequence: str) -> str:
    """Return the canonical sequence representation used for cache keys."""
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("Protein sequences must not be empty.")
    return normalized


def sequence_sha256(sequence: str) -> str:
    """Return the content hash of one canonicalized protein sequence."""
    normalized = normalize_protein_sequence(sequence)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncoderSpec:
    """Complete identity of a cached protein representation."""

    implementation: str
    model_name: str
    model_revision: str
    tokenizer_revision: str
    pooling: str
    maximum_length: int
    precision: str
    representation_type: str = "pooled_sequence"
    output_layer: str = "last_hidden_state"
    truncation_policy: str = "error"
    frozen: bool = True
    label_independent: bool = True
    checkpoint_sha256: str | None = None
    training_split_sha256: str | None = None
    spec_version: int = ENCODER_SPEC_VERSION

    def __post_init__(self) -> None:
        string_fields = (
            "implementation",
            "model_name",
            "model_revision",
            "tokenizer_revision",
            "pooling",
            "precision",
            "representation_type",
            "output_layer",
            "truncation_policy",
        )
        for field_name in string_fields:
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"EncoderSpec.{field_name} must not be empty.")
        if self.maximum_length < 1:
            raise ValueError("EncoderSpec.maximum_length must be positive.")
        if self.truncation_policy not in {"error", "truncate"}:
            raise ValueError(
                "EncoderSpec.truncation_policy must be error or truncate."
            )
        if not self.frozen:
            raise ValueError(
                "Cacheable EncoderSpec representations require frozen=True; "
                "write a checkpoint before caching trainable encoders."
            )
        if self.label_independent:
            if (
                self.checkpoint_sha256 is not None
                or self.training_split_sha256 is not None
            ):
                raise ValueError(
                    "Label-independent encoder caches must not carry "
                    "training-scoped checkpoint or split identities."
                )
        elif (
            not self.checkpoint_sha256
            or not self.training_split_sha256
        ):
            raise ValueError(
                "Label-dependent encoder caches require checkpoint_sha256 "
                "and training_split_sha256."
            )
        elif (
            not SHA256_PATTERN.fullmatch(self.checkpoint_sha256)
            or not SHA256_PATTERN.fullmatch(self.training_split_sha256)
        ):
            raise ValueError(
                "Fine-tuned encoder checkpoint_sha256 and "
                "training_split_sha256 must be 64-character SHA-256 "
                "hex digests."
            )
        if self.checkpoint_sha256 is not None:
            object.__setattr__(
                self,
                "checkpoint_sha256",
                self.checkpoint_sha256.lower(),
            )
        if self.training_split_sha256 is not None:
            object.__setattr__(
                self,
                "training_split_sha256",
                self.training_split_sha256.lower(),
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return asdict(self)

    @property
    def canonical_json(self) -> str:
        """Return the canonical serialization used for cache identity."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def fingerprint(self) -> str:
        """Return the content-addressed namespace for this encoder."""
        return hashlib.sha256(
            self.canonical_json.encode("utf-8")
        ).hexdigest()


@runtime_checkable
class ProteinEncoder(Protocol):
    """Interface implemented by frozen or checkpoint-scoped PLM adapters."""

    spec: EncoderSpec

    def token_lengths(self, sequences: Sequence[str]) -> np.ndarray:
        """Return encoded token lengths, including model special tokens."""
        ...

    def encode_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Return one pooled embedding row per sequence."""
        ...
