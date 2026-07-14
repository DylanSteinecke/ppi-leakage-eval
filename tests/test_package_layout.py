import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CLI_COMMANDS = (
    "ppi-train",
    "ppi-prepare",
    "ppi-aggregate",
    "ppi-grid",
    "ppi-make-toy-data",
)
CLI_MODULES = (
    "ppi_benchmark.cli.train",
    "ppi_benchmark.cli.prepare",
    "ppi_benchmark.cli.aggregate",
    "ppi_benchmark.cli.grid",
    "ppi_benchmark.cli.make_toy_data",
)


@pytest.mark.parametrize("command", CLI_COMMANDS)
def test_installed_cli_entry_points_work(command):
    command_path = shutil.which(command)
    assert command_path is not None, (
        f"{command} is not installed; run `python -m pip install -e .` first."
    )
    completed_process = subprocess.run(
        [command_path, "--help"],
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
