import hashlib
import numpy as np
import pandas as pd
import pytest

from ppi_benchmark.features import compose_pair_features
from ppi_benchmark.protein_encoders import (
    DEFAULT_PROTEIN_ENCODER_ADAPTER,
    PROTEIN_ENCODER_PRESET_CHOICES,
    PROTEIN_ENCODER_ADAPTER_CHOICES,
    CachedEmbedding,
    EmbeddingCache,
    EncoderSpec,
    FrozenProteinEncoder,
    HuggingFaceESM2Encoder,
    HuggingFaceProtBertEncoder,
    HuggingFaceProtT5Encoder,
    ProteinEncoder,
    ResidueTokenAlignment,
    TokenRepresentationEncoder,
    create_protein_encoder,
    default_embedding_cache_dir,
    encoder_identity_strength,
    get_protein_encoder_preset,
    prepare_prottrans_sequence,
    sequence_sha256,
    token_budget_batches,
)


def encoder_spec(**overrides):
    values = {
        "implementation": "recording_encoder_v1",
        "model_name": "test/protein-model",
        "model_revision": "revision-1",
        "tokenizer_revision": "tokenizer-1",
        "pooling": "mean",
        "maximum_length": 128,
        "precision": "float32",
    }
    values.update(overrides)
    return EncoderSpec(**values)


class RecordingProteinEncoder:
    def __init__(self, spec):
        self.spec = spec
        self.length_calls = []
        self.encode_calls = []

    def token_lengths(self, sequences):
        self.length_calls.append(tuple(sequences))
        return np.asarray([len(sequence) + 2 for sequence in sequences])

    def encode_batch(self, sequences):
        self.encode_calls.append(tuple(sequences))
        return np.asarray([
            [len(sequence), sum(map(ord, sequence))]
            for sequence in sequences
        ], dtype=np.float32)


def test_encoder_fingerprint_covers_cache_changing_configuration():
    base = encoder_spec()
    assert base.fingerprint == encoder_spec().fingerprint
    for field_name, changed_value in (
        ("implementation", "another_adapter_v1"),
        ("model_revision", "revision-2"),
        ("tokenizer_revision", "tokenizer-2"),
        ("pooling", "cls"),
        ("maximum_length", 64),
        ("precision", "float16"),
        ("truncation_policy", "truncate"),
    ):
        assert encoder_spec(**{field_name: changed_value}).fingerprint != (
            base.fingerprint)


def test_encoder_factory_selects_explicit_model_family(tiny_esm_model):
    assert DEFAULT_PROTEIN_ENCODER_ADAPTER == "esm2"
    assert PROTEIN_ENCODER_ADAPTER_CHOICES == (
        "esm2", "protbert", "prott5")

    encoder = create_protein_encoder(
        adapter="esm2",
        model_name=str(tiny_esm_model),
        model_revision="local-test-revision",
    )

    assert isinstance(encoder, HuggingFaceESM2Encoder)
    assert isinstance(encoder, ProteinEncoder)
    assert isinstance(encoder, TokenRepresentationEncoder)
    assert encoder.spec.implementation == "huggingface_esm2_v1"
    assert encoder.is_loaded is False

    with pytest.raises(ValueError, match="Unknown protein encoder adapter"):
        create_protein_encoder(adapter="unknown")


def test_approved_presets_are_self_supervised_and_resource_scoped():
    assert PROTEIN_ENCODER_PRESET_CHOICES == (
        "esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b",
        "esm2_15b", "protbert", "prott5_xl")
    for preset_name in PROTEIN_ENCODER_PRESET_CHOICES:
        preset = get_protein_encoder_preset(preset_name)
        assert preset.pretraining_scope == "self_supervised"
        assert len(preset.model_revision) == 40
    esm2_presets = [
        get_protein_encoder_preset(name)
        for name in PROTEIN_ENCODER_PRESET_CHOICES
        if name.startswith("esm2_")
    ]
    assert all(preset.adapter == "esm2" for preset in esm2_presets)
    assert all(
        preset.model_name.startswith("facebook/esm2_")
        for preset in esm2_presets
    )
    assert get_protein_encoder_preset("esm2_8m").resource_tier == "laptop"
    assert all(
        preset.resource_tier == "accelerator"
        for preset in esm2_presets[1:]
    )
    assert get_protein_encoder_preset("protbert").resource_tier == "laptop"
    assert get_protein_encoder_preset("prott5_xl").resource_tier == (
        "accelerator")


def test_protbert_adapter_maps_rare_residues_and_preserves_alignment(
        tiny_protbert_model):
    pytest.importorskip("torch")
    assert prepare_prottrans_sequence("AUZOB") == "A X X X X"
    encoder = HuggingFaceProtBertEncoder(
        model_name=str(tiny_protbert_model),
        model_revision="local-test-revision",
        maximum_length=32,
        device="cpu",
    )

    assert encoder.token_lengths(["AUZOB"]).tolist() == [7]
    tokenized = encoder.tokenize_with_alignment(["AUZOB"])
    observed = encoder.encode_batch(["AUZOB"])
    mapped = encoder.encode_batch(["AXXXX"])

    assert tokenized.alignment.residue_counts.tolist() == [5]
    assert tokenized.alignment.token_index(0, 0) == 1
    assert observed.shape == (1, 8)
    np.testing.assert_allclose(observed, mapped, atol=0.0, rtol=0.0)


def test_prott5_adapter_is_mean_only_and_preserves_alignment(
        tiny_prott5_model):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="pooling must be one of: mean"):
        HuggingFaceProtT5Encoder(
            model_name=str(tiny_prott5_model),
            model_revision="local-test-revision",
            pooling="cls",
        )
    encoder = HuggingFaceProtT5Encoder(
        model_name=str(tiny_prott5_model),
        model_revision="local-test-revision",
        maximum_length=32,
        device="cpu",
    )

    assert encoder.token_lengths(["AUZOB"]).tolist() == [6]
    tokenized = encoder.tokenize_with_alignment(["AUZOB"])
    observed = encoder.encode_batch(["AUZOB"])
    mapped = encoder.encode_batch(["AXXXX"])

    assert tokenized.alignment.residue_counts.tolist() == [5]
    assert tokenized.alignment.token_index(0, 0) == 0
    assert observed.shape == (1, 8)
    np.testing.assert_allclose(observed, mapped, atol=0.0, rtol=0.0)


def test_fine_tuned_cache_identity_requires_checkpoint_and_split_scope(
        tmp_path):
    with pytest.raises(ValueError, match="require checkpoint_sha256"):
        encoder_spec(label_independent=False)
    with pytest.raises(ValueError, match="require frozen=True"):
        encoder_spec(frozen=False)

    first = encoder_spec(
        label_independent=False,
        checkpoint_sha256="a" * 64,
        training_split_sha256="1" * 64,
    )
    changed_checkpoint = encoder_spec(
        label_independent=False,
        checkpoint_sha256="b" * 64,
        training_split_sha256="1" * 64,
    )
    changed_split = encoder_spec(
        label_independent=False,
        checkpoint_sha256="a" * 64,
        training_split_sha256="2" * 64,
    )
    assert len({
        first.fingerprint,
        changed_checkpoint.fingerprint,
        changed_split.fingerprint,
    }) == 3
    sequence_hash = sequence_sha256("ACDE")
    with EmbeddingCache(first, tmp_path) as cache:
        table = FrozenProteinEncoder(
            encoder=RecordingProteinEncoder(first),
            cache=cache,
            max_batch_tokens=32,
        ).encode({"P1": "ACDE"})
        assert table.metadata["encoder_spec"]["label_independent"] is False
        assert cache.get_many([sequence_hash])
    with EmbeddingCache(changed_checkpoint, tmp_path) as cache:
        assert cache.get_many([sequence_hash]) == {}
    with EmbeddingCache(changed_split, tmp_path) as cache:
        assert cache.get_many([sequence_hash]) == {}


def test_embedding_cache_round_trips_and_isolated_namespaces(tmp_path):
    sequence_hash = sequence_sha256("ac de")
    first_spec = encoder_spec()
    second_spec = encoder_spec(pooling="cls")
    record = CachedEmbedding(
        embedding=np.asarray([1.5, 2.5], dtype=np.float16),
        sequence_length=4,
    )

    with EmbeddingCache(first_spec, tmp_path) as cache:
        cache.put_many({sequence_hash: record})
        loaded = cache.get_many([sequence_hash])[sequence_hash]
        assert cache.count() == 1
        assert loaded.sequence_length == 4
        assert loaded.embedding.dtype == np.float16
        np.testing.assert_array_equal(loaded.embedding, record.embedding)

    with EmbeddingCache(second_spec, tmp_path) as cache:
        assert cache.path != tmp_path / f"{first_spec.fingerprint}.sqlite3"
        assert cache.get_many([sequence_hash]) == {}


def test_embedding_cache_detects_corrupt_payloads(tmp_path):
    spec = encoder_spec()
    sequence_hash = sequence_sha256("ACDE")
    with EmbeddingCache(spec, tmp_path) as cache:
        cache.put_many({
            sequence_hash: CachedEmbedding(
                embedding=np.asarray([1.0, 2.0], dtype=np.float32),
                sequence_length=4,
            ),
        })
        with cache._connection:
            cache._connection.execute(
                "UPDATE embeddings SET payload = ? WHERE sequence_sha256 = ?",
                (b"corrupt", sequence_hash),
            )
        with pytest.raises(ValueError, match="Corrupt embedding payload"):
            cache.get_many([sequence_hash])


def test_embedding_cache_directory_environment_precedence(tmp_path, monkeypatch):
    generic = tmp_path / "generic"
    legacy = tmp_path / "legacy"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("PROTEIN_BENCHMARK_EMBEDDING_CACHE_DIR", str(generic))
    monkeypatch.setenv("PPI_EMBEDDING_CACHE_DIR", str(legacy))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert default_embedding_cache_dir() == generic

    monkeypatch.delenv("PROTEIN_BENCHMARK_EMBEDDING_CACHE_DIR")
    assert default_embedding_cache_dir() == legacy
    monkeypatch.delenv("PPI_EMBEDDING_CACHE_DIR")
    assert default_embedding_cache_dir() == (
        xdg / "ppi-leakage" / "protein_embeddings"
    )


def test_embedding_contract_and_realized_table_hashes_are_independent(tmp_path):
    first_spec = encoder_spec(model_name=str(tmp_path / "model-a"))
    second_spec = encoder_spec(model_name=str(tmp_path / "model-b"))
    proteins = {"P1": "ACDE", "P2": "GGGG"}
    with EmbeddingCache(first_spec, tmp_path / "cache-a") as cache:
        first = FrozenProteinEncoder(
            encoder=RecordingProteinEncoder(first_spec),
            cache=cache,
            max_batch_tokens=32,
        ).encode(proteins)
    with EmbeddingCache(second_spec, tmp_path / "cache-b") as cache:
        second = FrozenProteinEncoder(
            encoder=RecordingProteinEncoder(second_spec),
            cache=cache,
            max_batch_tokens=32,
        ).encode({"different-a": "GGGG", "different-b": "ACDE"})

    assert first.metadata["unique_embedding_inputs_sha256"] == (
        second.metadata["unique_embedding_inputs_sha256"]
    )
    assert first.metadata["embedding_table_sha256"] == (
        second.metadata["embedding_table_sha256"]
    )
    assert first.metadata["embedding_contract_sha256"] != (
        second.metadata["embedding_contract_sha256"]
    )


def test_encoder_identity_strength_distinguishes_exact_and_declared(tmp_path):
    local_model = tmp_path / "model"
    local_model.mkdir()
    local = encoder_spec(model_name=str(local_model))
    remote = encoder_spec(
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
    )
    checkpoint = encoder_spec(
        label_independent=False,
        checkpoint_sha256="c" * 64,
        training_split_sha256="d" * 64,
    )

    assert encoder_identity_strength(local) == "declared_local_revision"
    assert encoder_identity_strength(remote) == "immutable_remote_revision"
    assert encoder_identity_strength(checkpoint) == "checkpoint_hash"


def test_token_budget_batches_bound_padding_and_group_similar_lengths():
    lengths = [5, 6, 20, 21, 22]
    batches = token_budget_batches(
        lengths,
        max_padded_tokens=42,
        max_sequences=3,
    )

    assert sorted(index for batch in batches for index in batch) == list(
        range(len(lengths)))
    assert batches == ((0, 1), (2, 3), (4,))
    for batch in batches:
        assert len(batch) <= 3
        if len(batch) > 1:
            assert len(batch) * max(lengths[index] for index in batch) <= 42

    with pytest.raises(ValueError, match="exceeding max_padded_tokens"):
        token_budget_batches([43], max_padded_tokens=42)


def test_frozen_encoder_deduplicates_sequences_and_reuses_cache(tmp_path):
    spec = encoder_spec()
    first_encoder = RecordingProteinEncoder(spec)
    proteins = {
        "P1": "acde",
        "P2": "AC DE",
        "P3": "GGGGGG",
        "P4": "W",
    }
    with EmbeddingCache(spec, tmp_path) as cache:
        table = FrozenProteinEncoder(
            encoder=first_encoder,
            cache=cache,
            max_batch_tokens=12,
            max_batch_sequences=2,
        ).encode(proteins)

    assert table.protein_ids == ("P1", "P2", "P3", "P4")
    np.testing.assert_array_equal(table.embeddings[0], table.embeddings[1])
    assert sum(len(batch) for batch in first_encoder.encode_calls) == 3
    assert sorted(
        sequence
        for batch in first_encoder.encode_calls
        for sequence in batch
    ) == ["ACDE", "GGGGGG", "W"]
    assert table.metadata["n_proteins"] == 4
    assert table.metadata["n_unique_sequences"] == 3
    assert table.metadata["n_duplicate_sequences"] == 1
    assert table.metadata["cache_hits"] == 0
    assert table.metadata["cache_misses"] == 3
    assert table.metadata["max_batch_padded_tokens"] <= 12

    second_encoder = RecordingProteinEncoder(spec)
    with EmbeddingCache(spec, tmp_path) as cache:
        second = FrozenProteinEncoder(
            encoder=second_encoder,
            cache=cache,
            max_batch_tokens=12,
        ).encode({"new_id": "ACDE", "other": "GGGGGG"})

    assert second_encoder.length_calls == []
    assert second_encoder.encode_calls == []
    assert second.metadata["cache_hits"] == 2
    assert second.metadata["cache_misses"] == 0


def test_dense_pair_composition_is_symmetric_and_preserves_precision():
    protein_ids = pd.Index(["A", "B"])
    protein_features = np.asarray([
        [1.0, 3.0],
        [2.0, -1.0],
    ], dtype=np.float16)
    forward = pd.DataFrame({"protein_a": ["A"], "protein_b": ["B"]})
    reverse = pd.DataFrame({"protein_a": ["B"], "protein_b": ["A"]})

    forward_features = compose_pair_features(
        forward,
        protein_ids,
        protein_features,
    )
    reverse_features = compose_pair_features(
        reverse,
        protein_ids,
        protein_features,
    )

    assert forward_features.dtype == np.float16
    np.testing.assert_array_equal(forward_features, reverse_features)
    np.testing.assert_array_equal(
        forward_features,
        np.asarray([[3.0, 2.0, 1.0, 4.0, 2.0, -3.0]], dtype=np.float16),
    )


def test_huggingface_esm_adapter_is_lazy_and_cacheable(tiny_esm_model):
    pytest.importorskip("torch")
    encoder = HuggingFaceESM2Encoder(
        model_name=str(tiny_esm_model),
        model_revision="local-test-revision",
        pooling="mean",
        maximum_length=32,
        precision="float32",
        device="cpu",
    )
    assert encoder.is_loaded is False

    lengths = encoder.token_lengths(["ACDE", "ACDEFG"])
    assert lengths.tolist() == [6, 8]
    assert encoder.is_loaded is False
    assert encoder._tokenizer is None
    embeddings = encoder.encode_batch(["ACDE", "ACDEFG"])

    assert encoder.is_loaded is True
    assert embeddings.shape == (2, 8)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert all(
        not parameter.requires_grad
        for parameter in encoder._model.parameters()
    )


def test_huggingface_token_representations_preserve_residue_alignment(
        tiny_esm_model):
    pytest.importorskip("torch")
    encoder = HuggingFaceESM2Encoder(
        model_name=str(tiny_esm_model),
        model_revision="local-test-revision",
        pooling="mean",
        maximum_length=32,
        precision="float32",
        device="cpu",
    )

    tokenized = encoder.tokenize_with_alignment(["ACDE", "FG"])
    token_batch = encoder.encode_token_batch(["ACDE", "FG"])
    pooled = encoder.encode_batch(["ACDE", "FG"])

    assert tokenized.alignment.residue_counts.tolist() == [4, 2]
    assert tokenized.alignment.token_index(0, 0) == 1
    assert tokenized.alignment.token_index(0, 3) == 4
    assert token_batch.representations.shape == (2, 6, 8)
    expected_pooled = np.stack([
        token_batch.representations[row][
            token_batch.tokenized.alignment.residue_token_mask[row]
        ].mean(axis=0)
        for row in range(2)
    ])
    np.testing.assert_allclose(pooled, expected_pooled, atol=1e-6, rtol=1e-6)


def test_residue_alignment_rejects_padded_or_nonresidue_indices():
    alignment = ResidueTokenAlignment.from_token_masks(
        [[1, 1, 1, 1]],
        [[0, 1, 1, 0]],
    )
    assert alignment.token_index(0, 0) == 1
    assert alignment.token_index(0, 1) == 2
    with pytest.raises(ValueError, match="not represented"):
        alignment.token_index(0, 2)


def test_remote_huggingface_encoder_rejects_floating_revision():
    with pytest.raises(ValueError, match="immutable 40-character"):
        HuggingFaceESM2Encoder(
            model_name="facebook/esm2_t6_8M_UR50D",
            model_revision="main",
        )


def test_huggingface_adapter_accepts_checkpoint_scoped_provenance(
        tiny_esm_model):
    encoder = HuggingFaceESM2Encoder(
        model_name=str(tiny_esm_model),
        model_revision="fine-tuned-local-revision",
        checkpoint_sha256="a" * 64,
        training_split_sha256="b" * 64,
    )

    assert encoder.spec.frozen is True
    assert encoder.spec.label_independent is False
    assert encoder.spec.checkpoint_sha256 == "a" * 64
    assert encoder.spec.training_split_sha256 == "b" * 64


def test_sequence_hash_is_normalized_and_label_independent():
    expected = hashlib.sha256(b"ACDE").hexdigest()
    assert sequence_sha256("ac de\n") == expected
