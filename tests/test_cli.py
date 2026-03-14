from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from aqdp.cli import _resolve_config, main


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = {
        "project": "TEST",
        "pi": "Test PI",
        "mooring": "Test Mooring",
        "latitude": 69.5,
        "longitude": -12.3,
        "bottom_depth": 250.0,
        "qc_enabled": False,
        "plots_enabled": False,
        "output_dir": str(tmp_path / "output"),
        "plots_dir": str(tmp_path / "fig"),
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    return p


def test_cli_process_creates_netcdf(
    deployment_dir: Path, config_file: Path, tmp_path: Path
):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "process",
            str(deployment_dir),
            "--config",
            str(config_file),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "output"
    assert (output_dir / "18223_dat.nc").exists()
    assert (output_dir / "18223_dia.nc").exists()


def test_cli_process_no_qc(deployment_dir: Path, config_file: Path, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "process",
            str(deployment_dir),
            "--config",
            str(config_file),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-qc",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_process_no_plots(deployment_dir: Path, config_file: Path, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "process",
            str(deployment_dir),
            "--config",
            str(config_file),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.output
    fig_dir = tmp_path / "fig"
    png_files = list(fig_dir.glob("*.png"))
    assert len(png_files) == 0


def test_cli_init_creates_config(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output
        assert Path("config.yml").exists()


def test_cli_init_custom_output(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init", "--output", "custom.yml"])
        assert result.exit_code == 0, result.output
        assert Path("custom.yml").exists()


def test_cli_init_refuses_overwrite(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("config.yml").write_text("existing")
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 1


@pytest.mark.parametrize("filename", ["config.yml", "config.yaml"])
def test_resolve_config_auto_detects_single_config(tmp_path: Path, filename: str):
    cfg = tmp_path / filename
    cfg.write_text("project: test\n")
    result = _resolve_config(None, tmp_path)
    assert result == cfg


def test_resolve_config_errors_on_no_yml(tmp_path: Path):
    with pytest.raises(SystemExit) as exc_info:
        _resolve_config(None, tmp_path)
    assert exc_info.value.code == 1


def test_resolve_config_errors_on_multiple_yml(tmp_path: Path):
    (tmp_path / "a.yml").write_text("project: a\n")
    (tmp_path / "b.yaml").write_text("project: b\n")
    with pytest.raises(SystemExit) as exc_info:
        _resolve_config(None, tmp_path)
    assert exc_info.value.code == 1


def test_resolve_config_explicit_path_passthrough(tmp_path: Path):
    cfg = tmp_path / "my.yml"
    cfg.write_text("project: test\n")
    result = _resolve_config(cfg, tmp_path)
    assert result == cfg


def test_resolve_config_explicit_path_missing(tmp_path: Path):
    with pytest.raises(SystemExit) as exc_info:
        _resolve_config(tmp_path / "nonexistent.yml", tmp_path)
    assert exc_info.value.code == 1


def test_cli_process_with_plots(
    deployment_dir: Path, config_file: Path, tmp_path: Path
):
    config = yaml.safe_load(config_file.read_text())
    config["plots_enabled"] = True
    config_file.write_text(yaml.dump(config))
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "process",
            str(deployment_dir),
            "--config",
            str(config_file),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 0, result.output
    fig_dir = tmp_path / "fig"
    png_files = list(fig_dir.glob("*.png"))
    assert len(png_files) >= 1
