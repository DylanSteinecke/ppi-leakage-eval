"""Lazy frozen ESM-2 adapter implemented with Hugging Face Transformers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from ..torch_utils import resolve_torch_device
from .base import EncoderSpec, normalize_protein_sequence


DEFAULT_ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"
PLM_POOLING_CHOICES = ("mean", "cls")
PLM_PRECISION_CHOICES = ("float32", "float16", "bfloat16")
PLM_TRUNCATION_CHOICES = ("error", "truncate")
HUGGINGFACE_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
ESM_SEQUENCE_SPECIAL_TOKENS = 2


def _validate_revision(model_name: str, revision: str) -> None:
    if Path(model_name).expanduser().exists():
        return
    if not HUGGINGFACE_COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(
            "Remote PLM revisions must be immutable 40-character Hugging "
            "Face commit hashes; floating revisions such as 'main' would "
            "make embedding caches unsafe."
        )


class HuggingFaceESM2Encoder:
    """Frozen ESM-2 encoder that loads weights only for cache misses."""

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
        _validate_revision(normalized_model_name, model_revision)
        _validate_revision(normalized_model_name, tokenizer_revision)
        if HUGGINGFACE_COMMIT_PATTERN.fullmatch(model_revision):
            model_revision = model_revision.lower()
        if HUGGINGFACE_COMMIT_PATTERN.fullmatch(tokenizer_revision):
            tokenizer_revision = tokenizer_revision.lower()
        if pooling not in PLM_POOLING_CHOICES:
            raise ValueError(
                f"pooling must be one of: {', '.join(PLM_POOLING_CHOICES)}."
            )
        if precision not in PLM_PRECISION_CHOICES:
            raise ValueError(
                "precision must be one of: "
                f"{', '.join(PLM_PRECISION_CHOICES)}."
            )
        label_independent = (
            checkpoint_sha256 is None
            and training_split_sha256 is None
        )
        self.spec = EncoderSpec(
            implementation="huggingface_esm2_v1",
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
            _, auto_tokenizer = self._transformers_imports()
            self._tokenizer = auto_tokenizer.from_pretrained(
                self.spec.model_name,
                revision=self.spec.tokenizer_revision,
            )
        return self._tokenizer

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import torch

        auto_model, _ = self._transformers_imports()
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
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.spec.precision]
        self._model = auto_model.from_pretrained(
            self.spec.model_name,
            revision=self.spec.model_revision,
            torch_dtype=dtype,
        )
        if getattr(self._model.config, "model_type", None) != "esm":
            raise ValueError(
                "HuggingFaceESM2Encoder requires an ESM/ESM-2 model."
            )
        self._model.to(device=self._device, dtype=dtype)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        return self._model

    def _tokenize(self, sequences: Sequence[str]):
        tokenizer = self._ensure_tokenizer()
        normalized = [
            normalize_protein_sequence(sequence)
            for sequence in sequences
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
        encoded = tokenizer(
            normalized,
            **tokenizer_options,
        )
        lengths = encoded["attention_mask"].sum(dim=1).cpu().numpy()
        if (
            self.spec.truncation_policy == "error"
            and np.any(lengths > self.spec.maximum_length)
        ):
            longest = int(np.max(lengths))
            raise ValueError(
                f"A protein tokenizes to {longest} tokens, exceeding "
                f"maximum_length={self.spec.maximum_length}. Increase the "
                "limit or use truncation_policy='truncate'."
            )
        return encoded, np.asarray(lengths, dtype=np.int64)

    def token_lengths(self, sequences: Sequence[str]) -> np.ndarray:
        """Return exact ESM residue-plus-special-token sequence lengths."""
        lengths = np.asarray([
            len(normalize_protein_sequence(sequence))
            + ESM_SEQUENCE_SPECIAL_TOKENS
            for sequence in sequences
        ], dtype=np.int64)
        if self.spec.truncation_policy == "truncate":
            np.minimum(
                lengths,
                self.spec.maximum_length,
                out=lengths,
            )
        elif len(lengths) and np.max(lengths) > self.spec.maximum_length:
            longest = int(np.max(lengths))
            raise ValueError(
                f"A protein tokenizes to {longest} tokens, exceeding "
                f"maximum_length={self.spec.maximum_length}. Increase the "
                "limit or use truncation_policy='truncate'."
            )
        return lengths

    def encode_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Encode and pool one token-budgeted sequence batch."""
        import torch

        if not sequences:
            raise ValueError("Cannot encode an empty sequence batch.")
        encoded, observed_lengths = self._tokenize(sequences)
        planned_lengths = self.token_lengths(sequences)
        if not np.array_equal(observed_lengths, planned_lengths):
            raise ValueError(
                "The selected tokenizer is not compatible with standard "
                "one-token-per-residue ESM-2 batching."
            )
        special_tokens_mask = encoded.pop("special_tokens_mask").bool()
        model = self._ensure_model()
        model_inputs = {
            key: value.to(self._device)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            hidden = model(**model_inputs).last_hidden_state
            if self.spec.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                valid_tokens = (
                    model_inputs["attention_mask"].bool()
                    & ~special_tokens_mask.to(self._device)
                )
                token_counts = valid_tokens.sum(dim=1)
                if torch.any(token_counts == 0):
                    raise ValueError(
                        "Mean pooling found a sequence without residue tokens."
                    )
                pooled = (
                    hidden * valid_tokens.unsqueeze(-1)
                ).sum(dim=1) / token_counts.unsqueeze(-1)
        pooled = pooled.cpu()
        if self.spec.precision == "float16":
            pooled = pooled.to(torch.float16)
        else:
            pooled = pooled.to(torch.float32)
        return pooled.numpy()
