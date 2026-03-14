from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aqdp.io import HeaderConfig, read_dat, read_header


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


# --- .dat parsing tests ---


def test_read_dat_returns_dataset(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert isinstance(ds, xr.Dataset)


def test_read_dat_has_time_coordinate(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert "time" in ds.coords
    assert np.issubdtype(ds.time.dtype, np.datetime64)


def test_read_dat_has_velocity_variables(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert "u" in ds
    assert "v" in ds
    assert "w" in ds


def test_read_dat_velocity_units(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert ds.u.attrs["units"] == "m/s"
    assert ds.u.attrs["standard_name"] == "eastward_sea_water_velocity"


def test_read_dat_has_all_variables(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    expected = {
        "u", "v", "w",
        "amplitude_beam1", "amplitude_beam2", "amplitude_beam3",
        "battery_voltage", "sound_speed", "sound_speed_used",
        "heading", "pitch", "roll",
        "pressure", "depth", "temperature",
        "analog_input_1", "analog_input_2",
        "speed", "direction",
        "magnetometer_x", "magnetometer_y", "magnetometer_z",
        "burst_counter", "ensemble_counter",
        "error_code", "status_code",
    }
    assert expected.issubset(set(ds.data_vars))


def test_read_dat_first_timestamp(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    first_time = ds.time.values[0]
    expected = np.datetime64("2024-11-12T16:08:50", "ns")
    assert first_time == expected


def test_read_dat_first_velocity_values(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert float(ds.u.values[0]) == pytest.approx(-0.026)
    assert float(ds.v.values[0]) == pytest.approx(0.099)
    assert float(ds.w.values[0]) == pytest.approx(-0.171)


def test_read_dat_length(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert ds.sizes["time"] == 1000


def test_read_dat_stores_header_attrs(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    assert ds.attrs["serial_number"] == "AQD18223"
    assert ds.attrs["coordinate_system"] == "ENU"
