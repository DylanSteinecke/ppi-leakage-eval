import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_COMMAND = "ppi-train"
TEST_THREAD_ENV = {
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Tiny test workloads are faster and more reproducible without a large BLAS
# thread pool. Set these before test modules import NumPy/scikit-learn.
for variable, value in TEST_THREAD_ENV.items():
    os.environ.setdefault(variable, value)


@pytest.fixture(scope="session")
def tiny_esm_model(tmp_path_factory):
    """Write a network-free tiny ESM model and tokenizer."""
    transformers = pytest.importorskip("transformers")
    vocab = [
        "<cls>", "<pad>", "<eos>", "<unk>",
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
        "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X", "<mask>",
    ]
    model_dir = tmp_path_factory.mktemp("models") / "tiny_esm"
    model_dir.mkdir()
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.EsmTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(model_dir)
    config = transformers.EsmConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=64,
        pad_token_id=vocab.index("<pad>"),
        mask_token_id=vocab.index("<mask>"),
        bos_token_id=vocab.index("<cls>"),
        eos_token_id=vocab.index("<eos>"),
        token_dropout=False,
    )
    transformers.EsmModel(
        config,
        add_pooling_layer=False,
    ).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def tiny_protbert_model(tmp_path_factory):
    """Write a network-free BERT model with a protein residue vocabulary."""
    transformers = pytest.importorskip("transformers")
    vocab = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
        "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X",
    ]
    model_dir = tmp_path_factory.mktemp("models") / "tiny_protbert"
    model_dir.mkdir()
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(
        vocab=str(vocab_path),
        do_lower_case=False,
    )
    tokenizer.save_pretrained(model_dir)
    # The real Rostlab/prot_bert checkpoint predates tokenizer.json.
    (model_dir / "tokenizer.json").unlink(missing_ok=True)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=64,
        pad_token_id=vocab.index("[PAD]"),
    )
    transformers.BertModel(
        config,
        add_pooling_layer=False,
    ).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def tiny_prott5_model(tmp_path_factory):
    """Write a network-free encoder-only T5 protein model."""
    pytest.importorskip("sentencepiece")
    transformers = pytest.importorskip("transformers")
    model_dir = tmp_path_factory.mktemp("models") / "tiny_prott5"
    model_dir.mkdir()
    residues = "ACDEFGHIKLMNPQRSTVWYX"
    vocab_scores = [
        ("<pad>", 0.0),
        ("</s>", 0.0),
        ("<unk>", 0.0),
        ("▁", -2.0),
        *((f"▁{residue}", 0.0) for residue in residues),
    ]
    tokenizer = transformers.T5Tokenizer(
        vocab=vocab_scores,
        extra_ids=0,
    )
    tokenizer.save_pretrained(model_dir)
    config = transformers.T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    transformers.T5EncoderModel(config).save_pretrained(model_dir)
    return model_dir


@pytest.fixture
def ppi_test_data(tmp_path):
    """
    Write a tiny balanced PPI dataset with enough rows for stratified splits.
    """
    protein_ids = []
    rows = ["pair_id,protein_a,protein_b,label"]
    for index in range(24):
        protein_a = f"P{index:02d}A"
        protein_b = f"P{index:02d}B"
        label = index % 2
        protein_ids.extend([protein_a, protein_b])
        rows.append(f"pair_{index},{protein_a},{protein_b},{label}")

    pairs_path = tmp_path / "pairs.csv"
    fasta_path = tmp_path / "proteins.fasta"
    pairs_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    suffixes = "MNPQRSTVWY"
    fasta_path.write_text(
        "".join(
            f">{protein_id}\nACDEFGHIKL{suffixes[index % len(suffixes)]}\n"
            for index, protein_id in enumerate(protein_ids)
        ),
        encoding="utf-8",
    )

    return pairs_path, fasta_path


@pytest.fixture(scope="session")
def test_process_environment(tmp_path_factory):
    """Return deterministic environment overrides shared by test processes."""
    environment = dict(TEST_THREAD_ENV)
    environment["MPLCONFIGDIR"] = str(
        tmp_path_factory.mktemp("matplotlib-config")
    )
    environment["PYTHONHASHSEED"] = "0"
    return environment


@pytest.fixture
def run_train(monkeypatch, test_process_environment):
    """Run the training application in-process for fast workflow tests."""
    from ppi_benchmark.cli.train import main

    for variable, value in test_process_environment.items():
        monkeypatch.setenv(variable, value)

    def _run_train(*args):
        main([*map(str, args)])

    return _run_train


@pytest.fixture
def run_cli(test_process_environment):
    """
    Return a helper that runs the installed CLI in a separate Python process.

    Use this only when process isolation or actual console behavior is part of
    the contract. Most workflow tests should use ``run_train`` instead.
    """
    def _run_cli(*args, check=True):
        env = os.environ.copy()
        env.update(test_process_environment)
        train_command = shutil.which(TRAIN_COMMAND)
        if train_command is None:
            raise AssertionError(
                "ppi-train is not installed. Run `python -m pip install -e .` "
                "before running the test suite."
            )
        command = [train_command, *map(str, args)]
        completed_process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if check and completed_process.returncode != 0:
            raise AssertionError(
                "Command failed:\n"
                f"{' '.join(command)}\n"
                f"STDOUT:\n{completed_process.stdout}\n"
                f"STDERR:\n{completed_process.stderr}"
            )

        return completed_process

    return _run_cli
