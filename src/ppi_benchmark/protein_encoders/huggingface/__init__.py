"""Frozen Hugging Face protein-model adapters."""

from .base import (
    PLM_POOLING_CHOICES,
    PLM_PRECISION_CHOICES,
    PLM_TRUNCATION_CHOICES,
    HuggingFaceProteinEncoderBase,
)
from .esm import DEFAULT_ESM2_MODEL, HuggingFaceESM2Encoder
from .protbert import HuggingFaceProtBertEncoder, prepare_prottrans_sequence
from .prott5 import HuggingFaceProtT5Encoder

__all__ = [
    "DEFAULT_ESM2_MODEL",
    "PLM_POOLING_CHOICES",
    "PLM_PRECISION_CHOICES",
    "PLM_TRUNCATION_CHOICES",
    "HuggingFaceESM2Encoder",
    "HuggingFaceProtBertEncoder",
    "HuggingFaceProtT5Encoder",
    "HuggingFaceProteinEncoderBase",
    "prepare_prottrans_sequence",
]
