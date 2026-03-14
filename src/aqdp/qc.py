"""Quality control flag functions for Aquadopp data."""

from __future__ import annotations

import numpy as np
import xarray as xr

# QC flag values
_FLAG_GOOD = 0
_FLAG_ERROR = 1
_FLAG_OUT_OF_RANGE = 2

_FLAG_ATTRS = {
    "long_name": "Quality control flag",
    "flag_values": [0, 1, 2],
    "flag_meanings": "good error_code_nonzero out_of_range",
}

# Physical range limits
_VELOCITY_MAX = 5.0  # m/s
_TEMP_MIN = -2.0  # degrees C
_TEMP_MAX = 35.0  # degrees C


def flag_by_status(ds: xr.Dataset) -> xr.Dataset:
    """Flag data points where error_code is nonzero.

    Returns a new Dataset with a ``qc_flag`` variable added.
    """
    ds = ds.copy(deep=True)
    n = ds.sizes["time"]
    qc_flag = np.zeros(n, dtype=np.int8)

    if "error_code" in ds:
        error_codes = ds.error_code.values
        for i in range(n):
            if str(error_codes[i]) != "00000000":
                qc_flag[i] = _FLAG_ERROR

    ds["qc_flag"] = ("time", qc_flag, _FLAG_ATTRS)
    return ds


def flag_by_range(ds: xr.Dataset) -> xr.Dataset:
    """Flag data points with physically unreasonable values.

    Returns a new Dataset with a ``qc_flag`` variable added or updated.
    """
    ds = ds.copy(deep=True)
    n = ds.sizes["time"]

    if "qc_flag" in ds:
        qc_flag = ds.qc_flag.values.copy()
    else:
        qc_flag = np.zeros(n, dtype=np.int8)

    # Check velocity variables
    for var in ("u", "v", "w"):
        if var in ds:
            bad = np.abs(ds[var].values) > _VELOCITY_MAX
            qc_flag[bad] = _FLAG_OUT_OF_RANGE

    # Check temperature
    if "temperature" in ds:
        temp = ds.temperature.values
        bad = (temp < _TEMP_MIN) | (temp > _TEMP_MAX)
        qc_flag[bad] = _FLAG_OUT_OF_RANGE

    # Check pressure
    if "pressure" in ds:
        bad = ds.pressure.values < 0
        qc_flag[bad] = _FLAG_OUT_OF_RANGE

    ds["qc_flag"] = ("time", qc_flag, _FLAG_ATTRS)
    return ds
