"""Frozen ProtT5 encoder adapter implemented with Transformers."""

from .base import HuggingFaceProteinEncoderBase
from .protbert import prepare_prottrans_sequence


class HuggingFaceProtT5Encoder(HuggingFaceProteinEncoderBase):
    """Frozen encoder-only ProtT5 adapter with residue mean pooling."""

    implementation = "huggingface_prott5_v1"
    expected_model_type = "t5"
    sequence_preprocessing = "prottrans_space_separated_map_uzob_to_x"
    n_special_tokens = 1
    supports_cls_pooling = False

    def _prepare_for_tokenizer(self, sequence: str) -> str:
        return prepare_prottrans_sequence(sequence)

    def _load_model(self, dtype):
        try:
            from transformers import T5EncoderModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ProtT5 requires Transformers and SentencePiece. Install "
                "the project with `python -m pip install -e '.[plm]'`."
            ) from exc
        return T5EncoderModel.from_pretrained(
            self.spec.model_name,
            revision=self.spec.model_revision,
            torch_dtype=dtype,
        )
