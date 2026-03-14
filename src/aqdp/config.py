"""YAML configuration parsing for aqdp processing."""

from __future__ import annotations

from dataclasses import dataclass
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
    water_depth: float | None
    instrument_depth: float | None
    qc_enabled: bool
    plots_enabled: bool
    output_dir: Path
    plots_dir: Path


_STARTER_TEMPLATE = """\
project: "PROJECT_NAME"
pi: "INVESTIGATOR_NAME"
mooring: "MOORING_ID"

# latitude: 0.0
# longitude: 0.0
# water_depth: 0.0
# instrument_depth: 0.0

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

    return ProcessingConfig(
        project=data["project"],
        pi=data["pi"],
        mooring=data["mooring"],
        latitude=data.get("latitude"),
        longitude=data.get("longitude"),
        water_depth=data.get("water_depth"),
        instrument_depth=data.get("instrument_depth"),
        qc_enabled=data["qc_enabled"],
        plots_enabled=data["plots_enabled"],
        output_dir=Path(data["output_dir"]),
        plots_dir=Path(data["plots_dir"]),
    )
