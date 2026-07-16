"""Frozen ProtBERT adapter implemented with Hugging Face Transformers."""

import inspect
import re

from .base import HuggingFaceProteinEncoderBase


PROTTRANS_RARE_RESIDUES = re.compile(r"[UZOB]")


def prepare_prottrans_sequence(sequence: str) -> str:
    """Apply documented ProtTrans residue mapping and whitespace format."""
    mapped = PROTTRANS_RARE_RESIDUES.sub("X", sequence)
    return " ".join(mapped)


class HuggingFaceProtBertEncoder(HuggingFaceProteinEncoderBase):
    """Frozen ProtBERT adapter with exact residue-token alignment."""

    implementation = "huggingface_protbert_v1"
    expected_model_type = "bert"
    sequence_preprocessing = "prottrans_space_separated_map_uzob_to_x"
    n_special_tokens = 2
    supports_cls_pooling = True

    def _load_tokenizer(self):
        """Load ProtBERT's legacy vocab-only tokenizer across HF versions."""
        try:
            from transformers import BertTokenizer
            from transformers.utils.hub import cached_file
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ProtBERT requires Transformers. Install the project with "
                "`python -m pip install -e '.[plm]'`."
            ) from exc
        vocab_path = cached_file(
            self.spec.model_name,
            "vocab.txt",
            revision=self.spec.tokenizer_revision,
        )
        vocab_argument = (
            "vocab"
            if "vocab" in inspect.signature(BertTokenizer).parameters
            else "vocab_file"
        )
        return BertTokenizer(
            **{vocab_argument: vocab_path},
            do_lower_case=False,
        )

    def _prepare_for_tokenizer(self, sequence: str) -> str:
        return prepare_prottrans_sequence(sequence)
