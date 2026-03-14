from pathlib import Path

from aqdp.io import HeaderConfig, read_header


def test_read_header_returns_header_config(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert isinstance(header, HeaderConfig)


def test_read_header_serial_number(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.serial_number == "AQD18223"


def test_read_header_deployment_name(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.deployment_name == "18223"


def test_read_header_coordinate_system(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.coordinate_system == "ENU"


def test_read_header_measurement_interval(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.measurement_interval == 44


def test_read_header_blanking_distance(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.blanking_distance == 0.50


def test_read_header_n_beams(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.n_beams == 3


def test_read_header_head_frequency(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.head_frequency == 2000


def test_read_header_salinity(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.salinity == 35.0


def test_read_header_n_measurements(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.n_measurements == 778493


def test_read_header_transformation_matrix_shape(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.transformation_matrix.shape == (3, 3)


def test_read_header_n_pings_per_burst(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.n_pings_per_burst == 23


def test_read_header_diagnostics_interval(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.diagnostics_interval == 43604


def test_read_header_diagnostics_n_samples(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.diagnostics_n_samples == 20


def test_read_header_firmware_version(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.firmware_version == "3.42"


def test_read_header_head_serial_number(deployment_dir: Path):
    header = read_header(deployment_dir)
    assert header.head_serial_number == "A6L 12746"
