from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from aqdp.cli import main


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = {
        "project": "TEST",
        "pi": "Test PI",
        "mooring": "Test Mooring",
        "latitude": 69.5,
        "longitude": -12.3,
        "water_depth": 250.0,
        "instrument_depth": 50.0,
        "qc_enabled": False,
        "plots_enabled": False,
        "output_dir": str(tmp_path / "output"),
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    return p


def test_cli_process_creates_netcdf(deployment_dir: Path, config_file: Path, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "process",
        str(deployment_dir),
        "--config", str(config_file),
        "--output-dir", str(tmp_path / "output"),
    ])
    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "output"
    assert (output_dir / "18223_dat.nc").exists()
    assert (output_dir / "18223_dia.nc").exists()


def test_cli_process_no_qc(deployment_dir: Path, config_file: Path, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "process",
        str(deployment_dir),
        "--config", str(config_file),
        "--output-dir", str(tmp_path / "output"),
        "--no-qc",
    ])
    assert result.exit_code == 0, result.output


def test_cli_process_no_plots(deployment_dir: Path, config_file: Path, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "process",
        str(deployment_dir),
        "--config", str(config_file),
        "--output-dir", str(tmp_path / "output"),
        "--no-plots",
    ])
    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "output"
    png_files = list(output_dir.glob("*.png"))
    assert len(png_files) == 0


def test_cli_process_with_plots(deployment_dir: Path, config_file: Path, tmp_path: Path):
    config = yaml.safe_load(config_file.read_text())
    config["plots_enabled"] = True
    config_file.write_text(yaml.dump(config))
    runner = CliRunner()
    result = runner.invoke(main, [
        "process",
        str(deployment_dir),
        "--config", str(config_file),
        "--output-dir", str(tmp_path / "output"),
    ])
    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "output"
    png_files = list(output_dir.glob("*.png"))
    assert len(png_files) >= 1
