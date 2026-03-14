from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aqdp.config import ProcessingConfig
from aqdp.io import HeaderConfig, read_dat, read_dia, read_header, read_log, to_netcdf


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
        "u",
        "v",
        "w",
        "amplitude_beam1",
        "amplitude_beam2",
        "amplitude_beam3",
        "battery_voltage",
        "sound_speed",
        "sound_speed_used",
        "heading",
        "pitch",
        "roll",
        "pressure",
        "depth",
        "temperature",
        "analog_input_1",
        "analog_input_2",
        "speed",
        "direction",
        "magnetometer_x",
        "magnetometer_y",
        "magnetometer_z",
        "burst_counter",
        "ensemble_counter",
        "error_code",
        "status_code",
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


# --- .dia parsing tests ---


def test_read_dia_returns_dataset(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dia(deployment_dir, header)
    assert isinstance(ds, xr.Dataset)


def test_read_dia_has_velocity_variables(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dia(deployment_dir, header)
    assert "u" in ds
    assert "v" in ds
    assert "w" in ds


def test_read_dia_no_magnetometer(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dia(deployment_dir, header)
    assert "magnetometer_x" not in ds
    assert "burst_counter" not in ds


def test_read_dia_has_speed_direction(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dia(deployment_dir, header)
    assert "speed" in ds
    assert "direction" in ds


def test_read_dia_first_timestamp(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dia(deployment_dir, header)
    first_time = ds.time.values[0]
    expected = np.datetime64("2024-11-12T16:08:52", "ns")
    assert first_time == expected


def test_read_dia_first_velocity_values(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds = read_dia(deployment_dir, header)
    assert float(ds.u.values[0]) == pytest.approx(-0.110)
    assert float(ds.v.values[0]) == pytest.approx(-0.083)
    assert float(ds.w.values[0]) == pytest.approx(-0.242)


# --- .ssl log parsing tests ---


def test_read_log_returns_dataset(deployment_dir: Path):
    ds = read_log(deployment_dir)
    assert isinstance(ds, xr.Dataset)


def test_read_log_has_expected_variables(deployment_dir: Path):
    ds = read_log(deployment_dir)
    assert "error_code" in ds
    assert "status_code" in ds
    assert "level" in ds
    assert "description" in ds
    assert "time" in ds.coords


def test_read_log_first_entry(deployment_dir: Path):
    ds = read_log(deployment_dir)
    assert str(ds.description.values[0]) == "First measurement"
    assert str(ds.level.values[0]) == "Info"


# --- NetCDF output tests ---


@pytest.fixture
def sample_config(tmp_path: Path) -> ProcessingConfig:
    return ProcessingConfig(
        project="TEST",
        pi="Test PI",
        mooring="Test Mooring",
        latitude=69.5,
        longitude=-12.3,
        water_depth=250.0,
        instrument_depth=50.0,
        qc_enabled=False,
        plots_enabled=False,
        output_dir=tmp_path,
    )


def test_to_netcdf_creates_file(deployment_dir: Path, tmp_path: Path, sample_config):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    output = tmp_path / "test.nc"
    to_netcdf(ds, output, sample_config)
    assert output.exists()


def test_to_netcdf_cf_conventions(deployment_dir: Path, tmp_path: Path, sample_config):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    output = tmp_path / "test.nc"
    to_netcdf(ds, output, sample_config)
    result = xr.open_dataset(output)
    assert result.attrs["Conventions"] == "CF-1.6"
    assert result.attrs["project"] == "TEST"
    assert result.attrs["serial_number"] == "AQD18223"
    result.close()


def test_to_netcdf_roundtrip_velocity(
    deployment_dir: Path, tmp_path: Path, sample_config
):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    output = tmp_path / "test.nc"
    to_netcdf(ds, output, sample_config)
    result = xr.open_dataset(output)
    np.testing.assert_allclose(result.u.values[:5], ds.u.values[:5])
    result.close()


def test_to_netcdf_time_encoding(deployment_dir: Path, tmp_path: Path, sample_config):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    output = tmp_path / "test.nc"
    to_netcdf(ds, output, sample_config)
    import netCDF4

    nc = netCDF4.Dataset(output)
    assert "seconds since" in nc.variables["time"].units
    nc.close()


def test_to_netcdf_has_history(deployment_dir: Path, tmp_path: Path, sample_config):
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    output = tmp_path / "test.nc"
    to_netcdf(ds, output, sample_config)
    result = xr.open_dataset(output)
    assert "aqdp" in result.attrs["history"]
    result.close()
