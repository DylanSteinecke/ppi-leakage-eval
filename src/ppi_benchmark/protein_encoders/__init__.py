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
    "PLM_POOLING_CHOICES",
    "PLM_PRECISION_CHOICES",
    "PLM_TRUNCATION_CHOICES",
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
    "default_embedding_cache_dir",
    "normalize_protein_sequence",
    "sequence_sha256",
    "token_budget_batches",
]
