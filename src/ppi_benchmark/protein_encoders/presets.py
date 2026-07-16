"""Approved self-supervised checkpoints and conservative runtime defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass


LAPTOP_RESOURCE_TIER = "laptop"
ACCELERATOR_RESOURCE_TIER = "accelerator"


@dataclass(frozen=True)
class ProteinEncoderPreset:
    """Auditable checkpoint identity plus safe inference defaults."""

    name: str
    adapter: str
    model_name: str
    model_revision: str
    pooling: str
    maximum_length: int
    truncation_policy: str
    precision: str
    device: str
    max_batch_tokens: int
    max_batch_sequences: int
    resource_tier: str
    pretraining_scope: str
    pretraining_objective: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible checkpoint provenance."""
        return asdict(self)

    def train_args(self) -> tuple[str, ...]:
        """Return canonical ``ppi-train`` arguments for this preset."""
        return (
            "--plm-preset", self.name,
            "--plm-adapter", self.adapter,
            "--plm-model", self.model_name,
            "--plm-revision", self.model_revision,
            "--plm-pooling", self.pooling,
            "--plm-max-length", str(self.maximum_length),
            "--plm-truncation-policy", self.truncation_policy,
            "--plm-precision", self.precision,
            "--plm-device", self.device,
            "--plm-max-batch-tokens", str(self.max_batch_tokens),
            "--plm-max-batch-sequences", str(self.max_batch_sequences),
        )


def _esm2_preset(
        name: str, model_name: str, model_revision: str,
        *, laptop: bool = False,
    ) -> ProteinEncoderPreset:
    """Build one ESM2 size preset without duplicating family defaults."""
    return ProteinEncoderPreset(
        name=name,
        adapter="esm2",
        model_name=model_name,
        model_revision=model_revision,
        pooling="mean",
        maximum_length=1024,
        truncation_policy="truncate",
        precision="float32" if laptop else "float16",
        device="cpu" if laptop else "cuda",
        max_batch_tokens=1024,
        max_batch_sequences=8 if laptop else 1,
        resource_tier=(
            LAPTOP_RESOURCE_TIER if laptop else ACCELERATOR_RESOURCE_TIER
        ),
        pretraining_scope="self_supervised",
        pretraining_objective="masked_language_modeling",
    )


PROTEIN_ENCODER_PRESETS = {
    "esm2_8m": _esm2_preset(
        name="esm2_8m",
        model_name="facebook/esm2_t6_8M_UR50D",
        model_revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        laptop=True,
    ),
    "esm2_35m": _esm2_preset(
        name="esm2_35m",
        model_name="facebook/esm2_t12_35M_UR50D",
        model_revision="6fbf070e65b0b7291e7bbcd451118c216cff79d8",
    ),
    "esm2_150m": _esm2_preset(
        name="esm2_150m",
        model_name="facebook/esm2_t30_150M_UR50D",
        model_revision="a695f6045e2e32885fa60af20c13cb35398ce30c",
    ),
    "esm2_650m": _esm2_preset(
        name="esm2_650m",
        model_name="facebook/esm2_t33_650M_UR50D",
        model_revision="08e4846e537177426273712802403f7ba8261b6c",
    ),
    "esm2_3b": _esm2_preset(
        name="esm2_3b",
        model_name="facebook/esm2_t36_3B_UR50D",
        model_revision="476b639933c8baad5ad09a60ac1a87f987b656fc",
    ),
    "esm2_15b": _esm2_preset(
        name="esm2_15b",
        model_name="facebook/esm2_t48_15B_UR50D",
        model_revision="5fbca39631164edc1d402a5aa369f982f72ee282",
    ),
    "protbert": ProteinEncoderPreset(
        name="protbert",
        adapter="protbert",
        model_name="Rostlab/prot_bert",
        model_revision="7a894481acdc12202f0a415dd567f6cfdb698908",
        pooling="mean",
        maximum_length=512,
        truncation_policy="truncate",
        precision="float32",
        device="cpu",
        max_batch_tokens=512,
        max_batch_sequences=1,
        resource_tier=LAPTOP_RESOURCE_TIER,
        pretraining_scope="self_supervised",
        pretraining_objective="masked_language_modeling",
    ),
    "prott5_xl": ProteinEncoderPreset(
        name="prott5_xl",
        adapter="prott5",
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        model_revision="94a6abc029ae13029317b140b7424e012bf8dfbf",
        pooling="mean",
        maximum_length=512,
        truncation_policy="truncate",
        precision="float16",
        device="cuda",
        max_batch_tokens=1024,
        max_batch_sequences=2,
        resource_tier=ACCELERATOR_RESOURCE_TIER,
        pretraining_scope="self_supervised",
        pretraining_objective="denoising_language_modeling",
    ),
}

PROTEIN_ENCODER_PRESET_CHOICES = tuple(PROTEIN_ENCODER_PRESETS)


def get_protein_encoder_preset(name: str) -> ProteinEncoderPreset:
    """Return one approved preset or fail with the supported choices."""
    try:
        return PROTEIN_ENCODER_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(PROTEIN_ENCODER_PRESET_CHOICES)
        raise ValueError(
            f"Unknown protein encoder preset {name!r}; choose from: {choices}."
        ) from exc
