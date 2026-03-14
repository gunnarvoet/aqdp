"""I/O functions for reading Aquadopp data files and writing NetCDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


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
