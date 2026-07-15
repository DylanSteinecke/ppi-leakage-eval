"""Public API for reusable protein encoders and embedding caches."""

from .base import (
    EncoderSpec,
    ProteinEncoder,
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
from .huggingface_esm import (
    DEFAULT_ESM2_MODEL,
    HuggingFaceESM2Encoder,
    PLM_POOLING_CHOICES,
    PLM_PRECISION_CHOICES,
    PLM_TRUNCATION_CHOICES,
)
from .representations import (
    ResidueAlignmentProvider,
    ResidueTokenAlignment,
    TokenizedProteinBatch,
    TokenRepresentationBatch,
    TokenRepresentationEncoder,
)

__all__ = [
    "DEFAULT_ESM2_MODEL",
    "DEFAULT_PROTEIN_ENCODER_ADAPTER",
    "PLM_POOLING_CHOICES",
    "PLM_PRECISION_CHOICES",
    "PLM_TRUNCATION_CHOICES",
    "PROTEIN_ENCODER_ADAPTER_CHOICES",
    "CachedEmbedding",
    "EmbeddingCache",
    "EncoderSpec",
    "FrozenProteinEncoder",
    "HuggingFaceESM2Encoder",
    "ProteinEmbeddingTable",
    "ProteinEncoder",
    "ResidueAlignmentProvider",
    "ResidueTokenAlignment",
    "TokenizedProteinBatch",
    "TokenRepresentationBatch",
    "TokenRepresentationEncoder",
    "create_protein_encoder",
    "default_embedding_cache_dir",
    "normalize_protein_sequence",
    "sequence_sha256",
    "token_budget_batches",
]
