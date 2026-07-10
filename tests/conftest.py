import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train_test_ppi_pred.py"


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
        command = [sys.executable, str(TRAIN_SCRIPT), *map(str, args)]
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
