"""Shared lazy inference mechanics for Hugging Face protein encoders."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from ...torch_utils import resolve_torch_device
from ..base import EncoderSpec, normalize_protein_sequence
from ..representations import (
    ResidueTokenAlignment,
    TokenizedProteinBatch,
    TokenRepresentationBatch,
)


PLM_POOLING_CHOICES = ("mean", "cls")
PLM_PRECISION_CHOICES = ("float32", "float16", "bfloat16")
PLM_TRUNCATION_CHOICES = ("error", "truncate")
HUGGINGFACE_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_huggingface_revision(model_name: str, revision: str) -> None:
    """Require immutable revisions for remote Hugging Face repositories."""
    if Path(model_name).expanduser().exists():
        return
    if not HUGGINGFACE_COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(
            "Remote PLM revisions must be immutable 40-character Hugging "
            "Face commit hashes; floating revisions such as 'main' would "
            "make embedding caches unsafe."
        )


class HuggingFaceProteinEncoderBase:
    """Common frozen inference for one-token-per-residue model adapters."""

    implementation = ""
    expected_model_type = ""
    sequence_preprocessing = ""
    n_special_tokens = 0
    supports_cls_pooling = False

    def __init__(
            self, model_name: str, model_revision: str,
            tokenizer_revision: str | None = None,
            pooling: str = "mean", maximum_length: int = 1024,
            precision: str = "float32", truncation_policy: str = "error",
            device: str = "auto", checkpoint_sha256: str | None = None,
            training_split_sha256: str | None = None,
        ):
        model_path = Path(model_name).expanduser()
        normalized_model_name = (
            str(model_path.resolve()) if model_path.exists() else model_name
        )
        tokenizer_revision = tokenizer_revision or model_revision
        validate_huggingface_revision(normalized_model_name, model_revision)
        validate_huggingface_revision(
            normalized_model_name, tokenizer_revision)
        if HUGGINGFACE_COMMIT_PATTERN.fullmatch(model_revision):
            model_revision = model_revision.lower()
        if HUGGINGFACE_COMMIT_PATTERN.fullmatch(tokenizer_revision):
            tokenizer_revision = tokenizer_revision.lower()
        allowed_pooling = (
            PLM_POOLING_CHOICES
            if self.supports_cls_pooling
            else ("mean",)
        )
        if pooling not in allowed_pooling:
            raise ValueError(
                f"{type(self).__name__} pooling must be one of: "
                f"{', '.join(allowed_pooling)}."
            )
        if precision not in PLM_PRECISION_CHOICES:
            raise ValueError(
                "precision must be one of: "
                f"{', '.join(PLM_PRECISION_CHOICES)}."
            )
        if maximum_length <= self.n_special_tokens:
            raise ValueError(
                "maximum_length must leave room for at least one residue."
            )
        label_independent = (
            checkpoint_sha256 is None
            and training_split_sha256 is None
        )
        self.spec = EncoderSpec(
            implementation=self.implementation,
            model_name=normalized_model_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            pooling=pooling,
            maximum_length=maximum_length,
            precision=precision,
            truncation_policy=truncation_policy,
            frozen=True,
            label_independent=label_independent,
            checkpoint_sha256=checkpoint_sha256,
            training_split_sha256=training_split_sha256,
        )
        self.requested_device = device
        self._device = None
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Return whether model weights have been materialized."""
        return self._model is not None

    @property
    def active_device(self) -> str | None:
        """Return the device used by the materialized model, when any."""
        return None if self._device is None else str(self._device)

    def _transformers_imports(self):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Frozen PLM features require Transformers. Install the "
                "project with `python -m pip install -e '.[plm]'`."
            ) from exc
        return AutoModel, AutoTokenizer

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    def _load_tokenizer(self):
        """Load the family tokenizer; adapters may override legacy formats."""
        _, auto_tokenizer = self._transformers_imports()
        return auto_tokenizer.from_pretrained(
            self.spec.model_name,
            revision=self.spec.tokenizer_revision,
        )

    def _torch_dtype(self):
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.spec.precision]

    def _load_model(self, dtype):
        auto_model, _ = self._transformers_imports()
        return auto_model.from_pretrained(
            self.spec.model_name,
            revision=self.spec.model_revision,
            torch_dtype=dtype,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        self._device = resolve_torch_device(self.requested_device)
        if self._device.type == "cpu" and self.spec.precision == "float16":
            raise ValueError(
                "float16 PLM inference on CPU is not supported; use float32 "
                "or bfloat16."
            )
        if self._device.type == "mps" and self.spec.precision == "bfloat16":
            raise ValueError(
                "bfloat16 PLM inference on MPS is not supported; use "
                "float16 or float32."
            )
        dtype = self._torch_dtype()
        model = self._load_model(dtype)
        observed_model_type = getattr(model.config, "model_type", None)
        if observed_model_type != self.expected_model_type:
            raise ValueError(
                f"{type(self).__name__} requires model_type="
                f"{self.expected_model_type!r}, got {observed_model_type!r}."
            )
        model.to(device=self._device, dtype=dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        return model

    def release_model(self) -> None:
        """Release materialized weights after pooled embeddings are cached."""
        device_type = (
            None if self._device is None else self._device.type
        )
        self._model = None
        self._device = None
        gc.collect()
        if device_type == "cuda":
            import torch

            torch.cuda.empty_cache()

    def _prepare_for_tokenizer(self, sequence: str) -> str:
        """Return the model-family-specific tokenizer input."""
        return sequence

    def _normalized_sequences(
            self, sequences: Sequence[str],
        ) -> tuple[str, ...]:
        return tuple(
            normalize_protein_sequence(sequence)
            for sequence in sequences
        )

    def token_lengths(self, sequences: Sequence[str]) -> np.ndarray:
        """Return exact residue-plus-special-token lengths without loading."""
        lengths = np.asarray([
            len(normalize_protein_sequence(sequence)) + self.n_special_tokens
            for sequence in sequences
        ], dtype=np.int64)
        if self.spec.truncation_policy == "truncate":
            np.minimum(lengths, self.spec.maximum_length, out=lengths)
        elif len(lengths) and np.max(lengths) > self.spec.maximum_length:
            longest = int(np.max(lengths))
            raise ValueError(
                f"A protein tokenizes to {longest} tokens, exceeding "
                f"maximum_length={self.spec.maximum_length}. Increase the "
                "limit or use truncation_policy='truncate'."
            )
        return lengths

    def tokenize_with_alignment(
            self, sequences: Sequence[str],
        ) -> TokenizedProteinBatch:
        """Tokenize sequences while preserving exact residue positions."""
        if not sequences:
            raise ValueError("Cannot tokenize an empty sequence batch.")
        normalized = self._normalized_sequences(sequences)
        tokenizer_inputs = [
            self._prepare_for_tokenizer(sequence)
            for sequence in normalized
        ]
        truncate = self.spec.truncation_policy == "truncate"
        tokenizer_options = {
            "add_special_tokens": True,
            "padding": True,
            "truncation": truncate,
            "return_special_tokens_mask": True,
            "return_tensors": "pt",
        }
        if truncate:
            tokenizer_options["max_length"] = self.spec.maximum_length
        encoded = self._ensure_tokenizer()(
            tokenizer_inputs,
            **tokenizer_options,
        )
        observed_lengths = np.asarray(
            encoded["attention_mask"].sum(dim=1).cpu().numpy(),
            dtype=np.int64,
        )
        planned_lengths = self.token_lengths(normalized)
        if not np.array_equal(observed_lengths, planned_lengths):
            raise ValueError(
                f"The selected tokenizer is not compatible with "
                f"one-token-per-residue {type(self).__name__} batching."
            )
        special_tokens_mask = encoded.pop("special_tokens_mask").bool()
        token_attention_mask = encoded["attention_mask"].bool()
        residue_token_mask = token_attention_mask & ~special_tokens_mask
        alignment = ResidueTokenAlignment.from_token_masks(
            token_attention_mask.cpu().numpy(),
            residue_token_mask.cpu().numpy(),
        )
        sequence_lengths = np.asarray(
            [len(sequence) for sequence in normalized],
            dtype=np.int64,
        )
        expected_residue_counts = sequence_lengths.copy()
        if truncate:
            np.minimum(
                expected_residue_counts,
                self.spec.maximum_length - self.n_special_tokens,
                out=expected_residue_counts,
            )
        if not np.array_equal(
                alignment.residue_counts, expected_residue_counts):
            raise ValueError(
                f"{type(self).__name__} tokenizer did not preserve exactly "
                "one token per represented input residue."
            )
        return TokenizedProteinBatch(
            sequences=normalized,
            model_inputs=dict(encoded),
            alignment=alignment,
        )

    def _hidden_states(self, tokenized: TokenizedProteinBatch):
        import torch

        model = self._ensure_model()
        model_inputs = {
            key: value.to(self._device)
            for key, value in tokenized.model_inputs.items()
        }
        with torch.inference_mode():
            return model(**model_inputs).last_hidden_state

    def _numpy_precision(self, tensor):
        import torch

        tensor = tensor.cpu()
        if self.spec.precision == "float16":
            tensor = tensor.to(torch.float16)
        else:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()

    def encode_token_batch(
            self, sequences: Sequence[str],
        ) -> TokenRepresentationBatch:
        """Return frozen token states with residue-to-token alignment."""
        tokenized = self.tokenize_with_alignment(sequences)
        hidden = self._hidden_states(tokenized)
        return TokenRepresentationBatch(
            tokenized=tokenized,
            representations=self._numpy_precision(hidden),
        )

    def encode_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Encode and pool one token-budgeted sequence batch."""
        import torch

        tokenized = self.tokenize_with_alignment(sequences)
        hidden = self._hidden_states(tokenized)
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            residue_token_mask = torch.from_numpy(
                tokenized.alignment.residue_token_mask
            ).to(self._device)
            token_counts = residue_token_mask.sum(dim=1)
            if torch.any(token_counts == 0):
                raise ValueError(
                    "Mean pooling found a sequence without residue tokens."
                )
            pooled = (
                hidden * residue_token_mask.unsqueeze(-1)
            ).sum(dim=1) / token_counts.unsqueeze(-1)
        return self._numpy_precision(pooled)
