import importlib
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration
CLI_ENTRY_POINTS = (
    ("ppi-train", "ppi_benchmark.cli.train:main"),
    ("ppi-prepare", "ppi_benchmark.cli.prepare:main"),
    ("ppi-aggregate", "ppi_benchmark.cli.aggregate:main"),
    ("ppi-grid", "ppi_benchmark.cli.grid:main"),
    ("ppi-make-toy-data", "ppi_benchmark.cli.make_toy_data:main"),
)
CLI_MODULES = (
    "ppi_benchmark.cli.train",
    "ppi_benchmark.cli.prepare",
    "ppi_benchmark.cli.aggregate",
    "ppi_benchmark.cli.grid",
    "ppi_benchmark.cli.make_toy_data",
)


@pytest.mark.parametrize(("command", "target"), CLI_ENTRY_POINTS)
def test_installed_cli_entry_points_are_registered(command, target):
    registered = {
        entry_point.name: entry_point.value
        for entry_point in distribution("ppi-leakage").entry_points
        if entry_point.group == "console_scripts"
    }

    assert registered[command] == target
    assert shutil.which(command) is not None, (
        f"{command} is not installed; run `python -m pip install -e .` first."
    )


def test_installed_training_entry_point_executes():
    command = "ppi-train"
    command_path = shutil.which(command)
    assert command_path is not None
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
def test_package_module_entry_points_work_without_install(
        module_name, monkeypatch, capsys):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out
