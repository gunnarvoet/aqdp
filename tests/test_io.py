import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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
        bottom_depth=250.0,
        qc_enabled=False,
        plots_enabled=False,
        output_dir=tmp_path,
        plots_dir=tmp_path,
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


# --- Clock drift correction tests ---


def _make_header(deployment_time: datetime) -> HeaderConfig:
    """Create a minimal HeaderConfig for testing."""
    return HeaderConfig(
        serial_number="TEST",
        deployment_name="test",
        coordinate_system="ENU",
        measurement_interval=60,
        blanking_distance=0.5,
        n_beams=3,
        head_frequency=2000,
        salinity=35.0,
        deployment_time=deployment_time,
        transformation_matrix=np.eye(3),
        comments="",
        n_measurements=10,
        transmit_pulse_length=1.0,
        sampling_rate="1 Hz",
        average_interval=60,
        compass_update_rate=1,
        firmware_version="1.0",
        software_version="1.0",
        pressure_sensor_calibration=[0, 0, 0, 0],
        n_pings_per_burst=1,
        diagnostics_interval=720,
        diagnostics_n_samples=20,
        head_serial_number="H001",
    )


def _make_config(
    time_instrument: datetime | None = None,
    time_utc: datetime | None = None,
) -> ProcessingConfig:
    """Create a minimal ProcessingConfig for testing."""
    return ProcessingConfig(
        project="TEST",
        pi="Test",
        mooring="Test",
        latitude=None,
        longitude=None,
        bottom_depth=None,
        qc_enabled=False,
        plots_enabled=False,
        output_dir=Path("out/"),
        plots_dir=Path("fig/"),
        time_instrument=time_instrument,
        time_utc=time_utc,
    )


def _make_dataset(times: list[str]) -> xr.Dataset:
    """Create a minimal dataset with a time coordinate."""
    time_arr = pd.to_datetime(times)
    return xr.Dataset(
        {"u": ("time", np.ones(len(times)))},
        coords={"time": time_arr.values},
    )


def test_apply_clock_drift_linear_correction():
    """Verify linear interpolation: zero at start, full drift at instrument time."""
    from aqdp.io import _apply_clock_drift

    deploy_time = datetime(2025, 1, 1, 0, 0, 0)
    header = _make_header(deploy_time)
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 0, 0, 10),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset([
        "2025-01-01T00:00:00",
        "2025-01-01T12:00:05",
        "2025-01-02T00:00:10",
    ])

    result = _apply_clock_drift(ds, header, config)

    expected = pd.to_datetime([
        "2025-01-01T00:00:00",
        "2025-01-01T12:00:00",
        "2025-01-02T00:00:00",
    ]).values
    np.testing.assert_array_equal(result.time.values, expected)


def test_apply_clock_drift_adds_global_attributes():
    from aqdp.io import _apply_clock_drift

    deploy_time = datetime(2025, 1, 1, 0, 0, 0)
    header = _make_header(deploy_time)
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 0, 0, 10),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T00:00:10"])

    result = _apply_clock_drift(ds, header, config)

    assert result.attrs["clock_drift_applied"] is True
    assert result.attrs["clock_drift_instrument_time"] == "2025-01-02 00:00:10"
    assert result.attrs["clock_drift_utc_time"] == "2025-01-02 00:00:00"


def test_apply_clock_drift_noop_when_no_drift_fields():
    from aqdp.io import _apply_clock_drift

    header = _make_header(datetime(2025, 1, 1))
    config = _make_config()
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T00:00:00"])

    result = _apply_clock_drift(ds, header, config)

    np.testing.assert_array_equal(result.time.values, ds.time.values)
    assert "clock_drift_applied" not in result.attrs


def test_apply_clock_drift_noop_when_config_none():
    from aqdp.io import _apply_clock_drift

    header = _make_header(datetime(2025, 1, 1))
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T00:00:00"])
    original_times = ds.time.values.copy()

    result = _apply_clock_drift(ds, header, None)

    np.testing.assert_array_equal(result.time.values, original_times)


def test_apply_clock_drift_single_record_noop():
    from aqdp.io import _apply_clock_drift

    header = _make_header(datetime(2025, 1, 1))
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 0, 0, 10),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T12:00:00"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _apply_clock_drift(ds, header, config)
        assert any("Cannot apply" in str(warning.message) for warning in w)

    np.testing.assert_array_equal(result.time.values, ds.time.values)


def test_apply_clock_drift_zero_total_span_noop():
    """total_span is zero when deployment_time == time_instrument."""
    from aqdp.io import _apply_clock_drift

    deploy_time = datetime(2025, 1, 1, 0, 0, 0)
    header = _make_header(deploy_time)
    config = _make_config(
        time_instrument=datetime(2025, 1, 1, 0, 0, 0),
        time_utc=datetime(2025, 1, 1, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-01T12:00:00"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _apply_clock_drift(ds, header, config)
        assert any("total_span is zero" in str(warning.message) for warning in w)

    np.testing.assert_array_equal(result.time.values, ds.time.values)


def test_apply_clock_drift_large_drift_warns():
    from aqdp.io import _apply_clock_drift

    header = _make_header(datetime(2025, 1, 1))
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 2, 0, 0),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T02:00:00"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _apply_clock_drift(ds, header, config)
        assert any("exceeds 1 hour" in str(warning.message) for warning in w)
