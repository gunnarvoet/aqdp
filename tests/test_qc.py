import numpy as np
import pandas as pd
import xarray as xr

from aqdp.qc import flag_by_range, flag_by_status


def _make_dataset(error_codes, status_codes, u_values):
    """Helper to create a minimal dataset for QC testing."""
    n = len(error_codes)
    time = pd.date_range("2024-01-01", periods=n, freq="s").values
    return xr.Dataset(
        {
            "u": ("time", u_values),
            "v": ("time", np.zeros(n)),
            "w": ("time", np.zeros(n)),
            "temperature": ("time", np.full(n, 20.0)),
            "pressure": ("time", np.full(n, 100.0)),
            "error_code": ("time", error_codes),
            "status_code": ("time", status_codes),
        },
        coords={"time": time},
    )


def test_flag_by_status_returns_new_dataset():
    ds = _make_dataset(["00000000", "00000000"], ["00100101", "00100101"], [0.1, 0.2])
    result = flag_by_status(ds)
    assert result is not ds
    assert "qc_flag" in result


def test_flag_by_status_no_errors_all_good():
    ds = _make_dataset(["00000000", "00000000"], ["00100101", "00100101"], [0.1, 0.2])
    result = flag_by_status(ds)
    assert (result.qc_flag.values == 0).all()


def test_flag_by_status_nonzero_error_flagged():
    ds = _make_dataset(["00000001", "00000000"], ["00100101", "00100101"], [0.1, 0.2])
    result = flag_by_status(ds)
    assert result.qc_flag.values[0] != 0
    assert result.qc_flag.values[1] == 0


def test_flag_by_range_returns_new_dataset():
    ds = _make_dataset(["00000000"], ["00100101"], [0.5])
    result = flag_by_range(ds)
    assert result is not ds
    assert "qc_flag" in result


def test_flag_by_range_normal_values_pass():
    ds = _make_dataset(["00000000"], ["00100101"], [0.5])
    result = flag_by_range(ds)
    assert (result.qc_flag.values == 0).all()


def test_flag_by_range_extreme_velocity_flagged():
    ds = _make_dataset(["00000000"], ["00100101"], [10.0])
    result = flag_by_range(ds)
    assert result.qc_flag.values[0] != 0


def test_flag_by_range_extreme_temperature_flagged():
    ds = _make_dataset(["00000000"], ["00100101"], [0.5])
    ds["temperature"].values[0] = 50.0  # way too hot
    result = flag_by_range(ds)
    assert result.qc_flag.values[0] != 0


def test_flag_by_range_preserves_original():
    ds = _make_dataset(["00000000"], ["00100101"], [10.0])
    flag_by_range(ds)
    assert "qc_flag" not in ds
