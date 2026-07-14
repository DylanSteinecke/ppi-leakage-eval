"""Token-level protein representations and residue-token alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ResidueTokenAlignment:
    """Exact mapping from zero-based residues to model-token positions."""

    token_attention_mask: np.ndarray
    residue_token_mask: np.ndarray
    residue_token_indices: np.ndarray
    residue_mask: np.ndarray

    def __post_init__(self) -> None:
        attention = np.asarray(self.token_attention_mask, dtype=bool)
        token_residues = np.asarray(self.residue_token_mask, dtype=bool)
        indices = np.asarray(self.residue_token_indices, dtype=np.int64)
        residue_mask = np.asarray(self.residue_mask, dtype=bool)
        if attention.ndim != 2 or token_residues.shape != attention.shape:
            raise ValueError(
                "Token attention and residue masks must be matching 2D arrays."
            )
        if indices.ndim != 2 or residue_mask.shape != indices.shape:
            raise ValueError(
                "Residue indices and masks must be matching 2D arrays."
            )
        if indices.shape[0] != attention.shape[0]:
            raise ValueError("Token and residue alignment batch sizes differ.")
        if np.any(token_residues & ~attention):
            raise ValueError("Residue tokens must also be attended tokens.")
        valid_indices = indices[residue_mask]
        if len(valid_indices):
            if np.any(valid_indices < 0) or np.any(
                valid_indices >= attention.shape[1]
            ):
                raise ValueError("Residue-token indices are out of bounds.")
            batch_rows = np.repeat(
                np.arange(indices.shape[0]),
                residue_mask.sum(axis=1),
            )
            if not np.all(token_residues[batch_rows, valid_indices]):
                raise ValueError(
                    "Residue-token indices must point to residue tokens."
                )
        if np.any(indices[~residue_mask] != -1):
            raise ValueError("Padded residue-token indices must be -1.")
        if not np.array_equal(
            residue_mask.sum(axis=1),
            token_residues.sum(axis=1),
        ):
            raise ValueError(
                "Every residue token must have exactly one residue index."
            )
        object.__setattr__(self, "token_attention_mask", attention)
        object.__setattr__(self, "residue_token_mask", token_residues)
        object.__setattr__(self, "residue_token_indices", indices)
        object.__setattr__(self, "residue_mask", residue_mask)

    @classmethod
    def from_token_masks(
            cls, token_attention_mask: Any, residue_token_mask: Any,
        ) -> "ResidueTokenAlignment":
        """Build padded residue indices from aligned token masks."""
        attention = np.asarray(token_attention_mask, dtype=bool)
        token_residues = np.asarray(residue_token_mask, dtype=bool)
        if attention.ndim != 2 or token_residues.shape != attention.shape:
            raise ValueError(
                "Token attention and residue masks must be matching 2D arrays."
            )
        residue_counts = token_residues.sum(axis=1)
        maximum_residues = (
            0 if not len(residue_counts) else int(residue_counts.max())
        )
        indices = np.full(
            (attention.shape[0], maximum_residues),
            -1,
            dtype=np.int64,
        )
        residue_mask = np.zeros_like(indices, dtype=bool)
        for batch_index, count in enumerate(residue_counts):
            count = int(count)
            indices[batch_index, :count] = np.flatnonzero(
                token_residues[batch_index]
            )
            residue_mask[batch_index, :count] = True
        return cls(
            token_attention_mask=attention,
            residue_token_mask=token_residues,
            residue_token_indices=indices,
            residue_mask=residue_mask,
        )

    @property
    def residue_counts(self) -> np.ndarray:
        """Return the number of represented residues per sequence."""
        return self.residue_mask.sum(axis=1).astype(np.int64, copy=False)

    def token_index(self, batch_index: int, residue_index: int) -> int:
        """Return one token index, failing if that residue was truncated."""
        if batch_index < 0 or batch_index >= self.residue_mask.shape[0]:
            raise IndexError("Batch index is out of range.")
        if (
            residue_index < 0
            or residue_index >= self.residue_mask.shape[1]
            or not self.residue_mask[batch_index, residue_index]
        ):
            raise ValueError(
                f"Residue {residue_index} is not represented in batch row "
                f"{batch_index}; it may have been truncated."
            )
        return int(self.residue_token_indices[batch_index, residue_index])


@dataclass(frozen=True)
class TokenizedProteinBatch:
    """Tokenizer outputs plus exact residue alignment for one batch."""

    sequences: tuple[str, ...]
    model_inputs: Mapping[str, Any]
    alignment: ResidueTokenAlignment

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("A tokenized protein batch must not be empty.")
        if len(self.sequences) != self.alignment.residue_mask.shape[0]:
            raise ValueError(
                "Tokenized sequences and residue alignment have different "
                "batch sizes."
            )


@dataclass(frozen=True)
class TokenRepresentationBatch:
    """Token-level hidden states paired with residue-token alignment."""

    tokenized: TokenizedProteinBatch
    representations: np.ndarray

    def __post_init__(self) -> None:
        representations = np.asarray(self.representations)
        expected_shape = self.tokenized.alignment.token_attention_mask.shape
        if representations.ndim != 3:
            raise ValueError("Token representations must be a 3D array.")
        if representations.shape[:2] != expected_shape:
            raise ValueError(
                "Token representations and token alignment shapes differ."
            )
        if not np.isfinite(representations).all():
            raise ValueError("Token representations contain non-finite values.")
        object.__setattr__(self, "representations", representations)


@runtime_checkable
class ResidueAlignmentProvider(Protocol):
    """Tokenizer contract required by residue/window task collators."""

    def tokenize_with_alignment(
            self, sequences: Sequence[str],
        ) -> TokenizedProteinBatch:
        """Tokenize sequences with exact zero-based residue mappings."""
        ...


@runtime_checkable
class TokenRepresentationEncoder(ResidueAlignmentProvider, Protocol):
    """Optional token-level extension to a pooled protein encoder."""

    def encode_token_batch(
            self, sequences: Sequence[str],
        ) -> TokenRepresentationBatch:
        """Return token-level representations and residue alignment."""
        ...
