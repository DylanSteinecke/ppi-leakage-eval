"""Lazy frozen ESM-2 adapter implemented with Hugging Face Transformers."""

from .base import HuggingFaceProteinEncoderBase


DEFAULT_ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"


class HuggingFaceESM2Encoder(HuggingFaceProteinEncoderBase):
    """Frozen ESM-2 adapter with native residue tokenization."""

    implementation = "huggingface_esm2_v1"
    expected_model_type = "esm"
    sequence_preprocessing = "esm2_native"
    n_special_tokens = 2
    supports_cls_pooling = True
