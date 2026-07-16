"""Explicit construction of model-specific protein encoder adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import ProteinEncoder
from .huggingface import (
    HuggingFaceESM2Encoder,
    HuggingFaceProtBertEncoder,
    HuggingFaceProtT5Encoder,
)


DEFAULT_PROTEIN_ENCODER_ADAPTER = "esm2"

EncoderFactory = Callable[..., ProteinEncoder]

_ENCODER_ADAPTERS: dict[str, EncoderFactory] = {
    "esm2": HuggingFaceESM2Encoder,
    "protbert": HuggingFaceProtBertEncoder,
    "prott5": HuggingFaceProtT5Encoder,
}

PROTEIN_ENCODER_ADAPTER_CHOICES = tuple(sorted(_ENCODER_ADAPTERS))


def create_protein_encoder(
        adapter: str = DEFAULT_PROTEIN_ENCODER_ADAPTER,
        **adapter_kwargs: Any,
    ) -> ProteinEncoder:
    """Construct one explicitly selected protein encoder adapter.

    Model families own their sequence preprocessing, token accounting,
    pooling, and residue alignment. Selecting an adapter explicitly avoids
    unsafe inference from a mutable model identifier.
    """
    try:
        factory = _ENCODER_ADAPTERS[adapter]
    except KeyError as exc:
        choices = ", ".join(PROTEIN_ENCODER_ADAPTER_CHOICES)
        raise ValueError(
            f"Unknown protein encoder adapter {adapter!r}; choose from: "
            f"{choices}."
        ) from exc
    return factory(**adapter_kwargs)
