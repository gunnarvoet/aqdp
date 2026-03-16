"""YAML configuration parsing for aqdp processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


class AqdpConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""


_REQUIRED_FIELDS = {
    "project",
    "pi",
    "mooring",
    "qc_enabled",
    "plots_enabled",
    "output_dir",
    "plots_dir",
}


@dataclass
class ProcessingConfig:
    project: str
    pi: str
    mooring: str
    latitude: float | None
    longitude: float | None
    bottom_depth: float | None
    qc_enabled: bool
    plots_enabled: bool
    output_dir: Path
    plots_dir: Path
    time_instrument: datetime | None = None
    time_utc: datetime | None = None


_STARTER_TEMPLATE = """\
project: "PROJECT_NAME"
pi: "INVESTIGATOR_NAME"
mooring: "MOORING_ID"

# latitude: 0.0
# longitude: 0.0
# bottom_depth: 0.0

# Clock drift correction
# time_instrument: "YYYY-MM-DD HH:MM:SS"
# time_utc: "YYYY-MM-DD HH:MM:SS"

qc_enabled: true
plots_enabled: true
output_dir: "proc/"
plots_dir: "fig/"
"""


def generate_config(path: Path) -> None:
    """Write a starter YAML config file with sensible defaults.

    Parameters
    ----------
    path : Path
        Destination file path.

    Raises
    ------
    FileExistsError
        If *path* already exists.
    """
    if path.exists():
        raise FileExistsError(f"File already exists: {path}")
    path.write_text(_STARTER_TEMPLATE)


def read_config(path: Path) -> ProcessingConfig:
    """Read a YAML config file and return a ProcessingConfig.

    Parameters
    ----------
    path : Path
        Path to the YAML config file.

    Returns
    -------
    ProcessingConfig
        Parsed configuration.

    Raises
    ------
    AqdpConfigError
        If required fields are missing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise AqdpConfigError(
            f"Missing required config fields: {', '.join(sorted(missing))}"
        )

    time_instrument_raw = data.get("time_instrument")
    time_utc_raw = data.get("time_utc")

    if (time_instrument_raw is None) != (time_utc_raw is None):
        raise AqdpConfigError(
            "Both time_instrument and time_utc must be provided together"
        )

    def _parse_datetime(value):
        """Parse a datetime from YAML — may be str or datetime (YAML auto-parses)."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    time_instrument = _parse_datetime(time_instrument_raw)
    time_utc = _parse_datetime(time_utc_raw)

    return ProcessingConfig(
        project=data["project"],
        pi=data["pi"],
        mooring=data["mooring"],
        latitude=data.get("latitude"),
        longitude=data.get("longitude"),
        bottom_depth=data.get("bottom_depth"),
        qc_enabled=data["qc_enabled"],
        plots_enabled=data["plots_enabled"],
        output_dir=Path(data["output_dir"]),
        plots_dir=Path(data["plots_dir"]),
        time_instrument=time_instrument,
        time_utc=time_utc,
    )
