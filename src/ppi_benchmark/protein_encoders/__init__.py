"""Public API for reusable protein encoders and embedding caches."""

from .base import (
    SEQUENCE_NORMALIZATION_SCHEMA_ID,
    EncoderSpec,
    ProteinEncoder,
    encoder_identity_strength,
    normalize_protein_sequence,
    sequence_sha256,
)
from .cache import (
    CachedEmbedding,
    EmbeddingCache,
    default_embedding_cache_dir,
)
from .frozen import (
    FrozenProteinEncoder,
    ProteinEmbeddingTable,
    token_budget_batches,
)
from .factory import (
    DEFAULT_PROTEIN_ENCODER_ADAPTER,
    PROTEIN_ENCODER_ADAPTER_CHOICES,
    create_protein_encoder,
)
from .huggingface import (
    DEFAULT_ESM2_MODEL,
    PLM_POOLING_CHOICES,
    PLM_PRECISION_CHOICES,
    PLM_TRUNCATION_CHOICES,
    HuggingFaceESM2Encoder,
    HuggingFaceProtBertEncoder,
    HuggingFaceProtT5Encoder,
    prepare_prottrans_sequence,
)
from .presets import (
    ACCELERATOR_RESOURCE_TIER,
    LAPTOP_RESOURCE_TIER,
    PROTEIN_ENCODER_PRESET_CHOICES,
    PROTEIN_ENCODER_PRESETS,
    ProteinEncoderPreset,
    get_protein_encoder_preset,
)
from .representations import (
    ResidueAlignmentProvider,
    ResidueTokenAlignment,
    TokenizedProteinBatch,
    TokenRepresentationBatch,
    TokenRepresentationEncoder,
)

__all__ = [
    "ACCELERATOR_RESOURCE_TIER",
    "DEFAULT_ESM2_MODEL",
    "DEFAULT_PROTEIN_ENCODER_ADAPTER",
    "PLM_POOLING_CHOICES",
    "PLM_PRECISION_CHOICES",
    "PLM_TRUNCATION_CHOICES",
    "PROTEIN_ENCODER_ADAPTER_CHOICES",
    "SEQUENCE_NORMALIZATION_SCHEMA_ID",
    "CachedEmbedding",
    "EmbeddingCache",
    "EncoderSpec",
    "FrozenProteinEncoder",
    "HuggingFaceESM2Encoder",
    "HuggingFaceProtBertEncoder",
    "HuggingFaceProtT5Encoder",
    "LAPTOP_RESOURCE_TIER",
    "ProteinEmbeddingTable",
    "ProteinEncoder",
    "ProteinEncoderPreset",
    "PROTEIN_ENCODER_PRESET_CHOICES",
    "PROTEIN_ENCODER_PRESETS",
    "ResidueAlignmentProvider",
    "ResidueTokenAlignment",
    "TokenizedProteinBatch",
    "TokenRepresentationBatch",
    "TokenRepresentationEncoder",
    "create_protein_encoder",
    "default_embedding_cache_dir",
    "encoder_identity_strength",
    "get_protein_encoder_preset",
    "normalize_protein_sequence",
    "prepare_prottrans_sequence",
    "sequence_sha256",
    "token_budget_batches",
]
