import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_COMMAND = "ppi-train"


@pytest.fixture
def tiny_esm_model(tmp_path):
    """Write a network-free tiny ESM model and tokenizer."""
    transformers = pytest.importorskip("transformers")
    vocab = [
        "<cls>", "<pad>", "<eos>", "<unk>",
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
        "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X", "<mask>",
    ]
    model_dir = tmp_path / "tiny_esm"
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


@pytest.fixture
def run_cli(tmp_path):
    """
    Return a helper that runs the training CLI inside this Python environment.
    """
    def _run_cli(*args, check=True):
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
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
