import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
LEGACY_SCRIPTS = (
    "train_test_ppi_pred.py",
    "prepare_ppi_dataset.py",
    "aggregate_benchmark_results.py",
    "make_toy_data.py",
)
CLI_MODULES = (
    "ppi_benchmark.cli.train",
    "ppi_benchmark.cli.prepare",
    "ppi_benchmark.cli.aggregate",
    "ppi_benchmark.cli.make_toy_data",
)


@pytest.mark.parametrize("script_name", LEGACY_SCRIPTS)
def test_legacy_script_entry_points_still_work(script_name):
    completed_process = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "usage:" in completed_process.stdout


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_package_module_entry_points_work_without_install(module_name):
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [str(SRC_DIR)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    completed_process = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "usage:" in completed_process.stdout
