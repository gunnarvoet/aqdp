"""I/O functions for reading Aquadopp data files and writing NetCDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class AqdpParsingError(Exception):
    """Raised when a data file cannot be parsed."""


@dataclass
class HeaderConfig:
    serial_number: str
    deployment_name: str
    coordinate_system: str
    measurement_interval: int
    blanking_distance: float
    n_beams: int
    head_frequency: int
    salinity: float
    deployment_time: datetime
    transformation_matrix: np.ndarray
    comments: str
    n_measurements: int
    transmit_pulse_length: float
    sampling_rate: str
    average_interval: int
    compass_update_rate: int
    firmware_version: str
    software_version: str
    pressure_sensor_calibration: list[int]
    n_pings_per_burst: int
    diagnostics_interval: int
    diagnostics_n_samples: int
    head_serial_number: str


def _find_file(directory: Path, subdirectory: str, extension: str) -> Path:
    """Find a single file with the given extension in a subdirectory."""
    search_dir = directory / subdirectory
    files = list(search_dir.glob(f"*{extension}"))
    if not files:
        raise AqdpParsingError(f"No {extension} file found in {search_dir}")
    return files[0]


def _parse_int_value(text: str) -> int:
    """Extract integer from value string like '44 sec' or '2000 kHz'."""
    match = re.match(r"(-?\d+)", text.strip())
    if match:
        return int(match.group(1))
    raise AqdpParsingError(f"Cannot parse integer from: {text!r}")


def _parse_float_value(text: str) -> float:
    """Extract float from value string like '0.50 m' or '35.0 ppt'."""
    match = re.match(r"(-?\d+\.?\d*)", text.strip())
    if match:
        return float(match.group(1))
    raise AqdpParsingError(f"Cannot parse float from: {text!r}")


def read_header(path: Path) -> HeaderConfig:
    """Parse a .hdr file from a deployment directory.

    Parameters
    ----------
    path : Path
        Deployment root directory containing a ``raw/`` subdirectory.

    Returns
    -------
    HeaderConfig
        Parsed header metadata.
    """
    hdr_path = _find_file(path, "raw", ".hdr")
    text = hdr_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Track which section we're in
    section = "preamble"
    section_fields: dict[str, dict[str, str]] = {
        "preamble": {},
        "user_setup": {},
        "hardware": {},
        "head": {},
        "data_format": {},
    }

    # Multi-line values we need to accumulate
    transformation_matrix_lines: list[str] = []
    collecting_matrix = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Section detection
        if stripped == "User setup":
            section = "user_setup"
            i += 1
            continue
        elif stripped == "Hardware configuration":
            section = "hardware"
            i += 1
            continue
        elif stripped == "Head configuration":
            section = "head"
            i += 1
            continue
        elif stripped == "Data file format":
            section = "data_format"
            i += 1
            continue

        # Skip separator lines and empty lines
        if stripped.startswith("-----") or stripped == "" or stripped.startswith("["):
            i += 1
            continue

        # Check if this is a continuation line (starts with spaces, no label)
        if collecting_matrix and line.startswith("                  "):
            transformation_matrix_lines.append(stripped)
            if len(transformation_matrix_lines) == 3:
                collecting_matrix = False
            i += 1
            continue

        # Parse key-value pairs
        # The format is: label (variable width) followed by value
        # Split on 2+ consecutive spaces
        parts = re.split(r"  {2,}", stripped, maxsplit=1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()

            # Check for transformation matrix start
            if key == "Transformation matrix":
                transformation_matrix_lines = [value]
                collecting_matrix = True
                i += 1
                continue

            section_fields[section][key] = value

        i += 1

    # Extract values from the appropriate sections
    preamble = section_fields["preamble"]
    user = section_fields["user_setup"]
    hardware = section_fields["hardware"]
    head = section_fields["head"]

    # Parse transformation matrix
    matrix_rows = []
    for mline in transformation_matrix_lines:
        row = [float(x) for x in mline.split()]
        matrix_rows.append(row)
    transformation_matrix = np.array(matrix_rows)

    # Parse deployment time
    dt_str = user.get("Deployment time", "")
    deployment_time = datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")

    # Parse pressure sensor calibration
    cal_str = head.get("Pressure sensor calibration", "0 0 0 0")
    pressure_cal = [int(x) for x in cal_str.split()]

    return HeaderConfig(
        serial_number=hardware.get("Serial number", ""),
        deployment_name=user.get("Deployment name", ""),
        coordinate_system=user.get("Coordinate system", ""),
        measurement_interval=_parse_int_value(user.get("Measurement interval", "0")),
        blanking_distance=_parse_float_value(user.get("Blanking distance", "0")),
        n_beams=_parse_int_value(head.get("Number of beams", "0")),
        head_frequency=_parse_int_value(head.get("Head frequency", "0")),
        salinity=_parse_float_value(user.get("Salinity", "0")),
        deployment_time=deployment_time,
        transformation_matrix=transformation_matrix,
        comments=user.get("Comments", ""),
        n_measurements=_parse_int_value(preamble.get("Number of measurements", "0")),
        transmit_pulse_length=_parse_float_value(
            user.get("Transmit pulse length", "0")
        ),
        sampling_rate=user.get("Sampling rate", ""),
        average_interval=_parse_int_value(user.get("Average interval", "0")),
        compass_update_rate=_parse_int_value(user.get("Compass update rate", "0")),
        firmware_version=hardware.get("Firmware version", ""),
        software_version=user.get("Software version", ""),
        pressure_sensor_calibration=pressure_cal,
        n_pings_per_burst=_parse_int_value(user.get("Number of pings per burst", "0")),
        diagnostics_interval=_parse_int_value(user.get("Diagnostics - Interval", "0")),
        diagnostics_n_samples=_parse_int_value(
            user.get("Diagnostics - Number of samples", "0")
        ),
        head_serial_number=head.get("Serial number", ""),
    )


# Variable attributes for CF compliance
DAT_VAR_ATTRS = {
    "u": {
        "units": "m/s",
        "standard_name": "eastward_sea_water_velocity",
        "long_name": "Eastward velocity",
    },
    "v": {
        "units": "m/s",
        "standard_name": "northward_sea_water_velocity",
        "long_name": "Northward velocity",
    },
    "w": {
        "units": "m/s",
        "standard_name": "upward_sea_water_velocity",
        "long_name": "Upward velocity",
    },
    "amplitude_beam1": {"units": "counts", "long_name": "Amplitude beam 1"},
    "amplitude_beam2": {"units": "counts", "long_name": "Amplitude beam 2"},
    "amplitude_beam3": {"units": "counts", "long_name": "Amplitude beam 3"},
    "battery_voltage": {"units": "V", "long_name": "Battery voltage"},
    "sound_speed": {
        "units": "m/s",
        "standard_name": "speed_of_sound_in_sea_water",
        "long_name": "Measured sound speed",
    },
    "sound_speed_used": {
        "units": "m/s",
        "long_name": "Sound speed used in calculations",
    },
    "heading": {"units": "degrees", "long_name": "Heading"},
    "pitch": {
        "units": "degrees",
        "standard_name": "platform_pitch",
        "long_name": "Pitch",
    },
    "roll": {"units": "degrees", "standard_name": "platform_roll", "long_name": "Roll"},
    "pressure": {
        "units": "dbar",
        "standard_name": "sea_water_pressure",
        "long_name": "Pressure",
    },
    "depth": {
        "units": "m",
        "standard_name": "depth",
        "long_name": "Depth",
        "positive": "down",
    },
    "temperature": {
        "units": "degrees_C",
        "standard_name": "sea_water_temperature",
        "long_name": "Temperature",
    },
    "analog_input_1": {"long_name": "Analog input 1"},
    "analog_input_2": {"long_name": "Analog input 2"},
    "speed": {"units": "m/s", "long_name": "Current speed"},
    "direction": {"units": "degrees", "long_name": "Current direction"},
    "magnetometer_x": {"units": "counts", "long_name": "Magnetometer X"},
    "magnetometer_y": {"units": "counts", "long_name": "Magnetometer Y"},
    "magnetometer_z": {"units": "counts", "long_name": "Magnetometer Z"},
    "burst_counter": {"long_name": "Burst counter"},
    "ensemble_counter": {"long_name": "Ensemble counter"},
    "error_code": {"long_name": "Error code"},
    "status_code": {"long_name": "Status code"},
}

# Column mapping for .dat files (0-indexed)
_DAT_COLUMNS = {
    6: "burst_counter",
    7: "ensemble_counter",
    8: "error_code",
    9: "status_code",
    10: "u",
    11: "v",
    12: "w",
    13: "amplitude_beam1",
    14: "amplitude_beam2",
    15: "amplitude_beam3",
    16: "battery_voltage",
    17: "sound_speed",
    18: "sound_speed_used",
    19: "heading",
    20: "pitch",
    21: "roll",
    22: "pressure",
    23: "depth",
    24: "temperature",
    25: "analog_input_1",
    26: "analog_input_2",
    27: "speed",
    28: "direction",
    29: "magnetometer_x",
    30: "magnetometer_y",
    31: "magnetometer_z",
}


def _header_to_attrs(header: HeaderConfig) -> dict:
    """Convert HeaderConfig fields to dataset attributes."""
    return {
        "serial_number": header.serial_number,
        "deployment_name": header.deployment_name,
        "coordinate_system": header.coordinate_system,
        "head_frequency": header.head_frequency,
        "firmware_version": header.firmware_version,
        "deployment_time": str(header.deployment_time),
        "comments": header.comments,
        "measurement_interval": header.measurement_interval,
        "blanking_distance": header.blanking_distance,
        "n_beams": header.n_beams,
        "salinity": header.salinity,
    }


def read_dat(path: Path, header: HeaderConfig) -> xr.Dataset:
    """Read a .dat measurement file into an xarray Dataset.

    Parameters
    ----------
    path : Path
        Deployment root directory containing a ``raw/`` subdirectory.
    header : HeaderConfig
        Parsed header metadata.

    Returns
    -------
    xr.Dataset
        Dataset with all measurement variables and CF attributes.
    """
    dat_path = _find_file(path, "raw", ".dat")
    df = pd.read_csv(
        dat_path,
        sep=r"\s+",
        header=None,
        dtype={8: str, 9: str},
    )

    # Construct time coordinate from columns 0-5
    time = pd.to_datetime(
        df[[2, 0, 1, 3, 4, 5]].rename(
            columns={
                2: "year",
                0: "month",
                1: "day",
                3: "hour",
                4: "minute",
                5: "second",
            }
        )
    )

    # Build data variables
    data_vars = {}
    for col_idx, var_name in _DAT_COLUMNS.items():
        attrs = DAT_VAR_ATTRS.get(var_name, {})
        data_vars[var_name] = ("time", df[col_idx].values, attrs)

    ds = xr.Dataset(data_vars, coords={"time": time.values})
    ds.attrs.update(_header_to_attrs(header))
    return ds


# Column mapping for .dia files (0-indexed)
_DIA_COLUMNS = {
    6: "error_code",
    7: "status_code",
    8: "u",
    9: "v",
    10: "w",
    11: "amplitude_beam1",
    12: "amplitude_beam2",
    13: "amplitude_beam3",
    14: "battery_voltage",
    15: "sound_speed",
    16: "sound_speed_used",
    17: "heading",
    18: "pitch",
    19: "roll",
    20: "pressure",
    21: "depth",
    22: "temperature",
    23: "analog_input_1",
    24: "analog_input_2",
    25: "speed",
    26: "direction",
}


def read_dia(path: Path, header: HeaderConfig) -> xr.Dataset:
    """Read a .dia diagnostics file into an xarray Dataset.

    Parameters
    ----------
    path : Path
        Deployment root directory containing a ``raw/`` subdirectory.
    header : HeaderConfig
        Parsed header metadata.

    Returns
    -------
    xr.Dataset
        Dataset with diagnostics variables and CF attributes.
    """
    dia_path = _find_file(path, "raw", ".dia")
    df = pd.read_csv(
        dia_path,
        sep=r"\s+",
        header=None,
        dtype={6: str, 7: str},
    )

    # Construct time coordinate from columns 0-5
    time = pd.to_datetime(
        df[[2, 0, 1, 3, 4, 5]].rename(
            columns={
                2: "year",
                0: "month",
                1: "day",
                3: "hour",
                4: "minute",
                5: "second",
            }
        )
    )

    # Build data variables
    data_vars = {}
    for col_idx, var_name in _DIA_COLUMNS.items():
        attrs = DAT_VAR_ATTRS.get(var_name, {})
        data_vars[var_name] = ("time", df[col_idx].values, attrs)

    ds = xr.Dataset(data_vars, coords={"time": time.values})
    ds.attrs.update(_header_to_attrs(header))
    return ds


_SSL_LINE_RE = re.compile(
    r"^(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+(\d+)\s+(\d+)\s+(\w+)\s+(.+)$"
)


def read_log(path: Path) -> xr.Dataset:
    """Read a .ssl log file into an xarray Dataset.

    Parameters
    ----------
    path : Path
        Deployment root directory containing a ``log/`` subdirectory.

    Returns
    -------
    xr.Dataset
        Dataset with log entries.
    """
    ssl_path = _find_file(path, "log", ".ssl")
    text = ssl_path.read_text(encoding="utf-8", errors="replace")
    data_lines = text.splitlines()[1:]  # skip header

    times = []
    error_codes = []
    status_codes = []
    levels = []
    descriptions = []

    for line in data_lines:
        match = _SSL_LINE_RE.match(line.strip())
        if not match:
            continue
        dt_str, ecode, scode, level, desc = match.groups()
        times.append(pd.to_datetime(dt_str, format="%m/%d/%Y %H:%M:%S.%f"))
        error_codes.append(ecode)
        status_codes.append(scode)
        levels.append(level)
        descriptions.append(desc)

    time_arr = pd.DatetimeIndex(times)

    return xr.Dataset(
        {
            "error_code": ("time", error_codes),
            "status_code": ("time", status_codes),
            "level": ("time", levels),
            "description": ("time", descriptions),
        },
        coords={"time": time_arr.values},
    )


def to_netcdf(ds: xr.Dataset, output: Path, config) -> None:
    """Write a dataset to a CF-1.6 compliant NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to write.
    output : Path
        Output file path.
    config : ProcessingConfig
        Processing configuration with metadata.
    """
    from datetime import datetime as dt

    # Add global attributes from config
    ds = ds.copy()
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["history"] = f"Created {dt.now().isoformat()} by aqdp v0.1.0"
    serial = ds.attrs.get("serial_number", "unknown")
    ds.attrs["source"] = f"Nortek Aquadopp {serial}"

    # Add config metadata
    ds.attrs["project"] = config.project
    ds.attrs["pi"] = config.pi
    ds.attrs["mooring"] = config.mooring
    if config.latitude is not None:
        ds.attrs["latitude"] = config.latitude
    if config.longitude is not None:
        ds.attrs["longitude"] = config.longitude
    if config.bottom_depth is not None:
        ds.attrs["bottom_depth"] = config.bottom_depth

    # Time encoding
    encoding = {
        "time": {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "calendar": "standard",
        }
    }

    ds.to_netcdf(output, encoding=encoding)
