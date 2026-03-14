from pathlib import Path

import pytest
import yaml

from aqdp.config import AqdpConfigError, ProcessingConfig, read_config


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = {
        "project": "MOTIVE",
        "pi": "Jane Smith",
        "mooring": "Mooring-A",
        "latitude": 69.5,
        "longitude": -12.3,
        "water_depth": 250.0,
        "instrument_depth": 50.0,
        "qc_enabled": True,
        "plots_enabled": True,
        "output_dir": "output/",
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    return p


def test_read_config_returns_processing_config(config_file: Path):
    config = read_config(config_file)
    assert isinstance(config, ProcessingConfig)


def test_read_config_metadata_fields(config_file: Path):
    config = read_config(config_file)
    assert config.project == "MOTIVE"
    assert config.pi == "Jane Smith"
    assert config.mooring == "Mooring-A"
    assert config.latitude == 69.5
    assert config.longitude == -12.3


def test_read_config_processing_options(config_file: Path):
    config = read_config(config_file)
    assert config.qc_enabled is True
    assert config.plots_enabled is True
    assert config.output_dir == Path("output/")


def test_read_config_optional_fields_default_none(tmp_path: Path):
    config = {
        "project": "TEST",
        "pi": "Test PI",
        "mooring": "Test Mooring",
        "qc_enabled": False,
        "plots_enabled": False,
        "output_dir": "out/",
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    result = read_config(p)
    assert result.latitude is None
    assert result.longitude is None
    assert result.water_depth is None
    assert result.instrument_depth is None


def test_read_config_missing_required_field_raises(tmp_path: Path):
    config = {"pi": "Test"}  # missing project, mooring, etc.
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    with pytest.raises(AqdpConfigError):
        read_config(p)
