"""YAML configuration parsing for aqdp processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class AqdpConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""


_REQUIRED_FIELDS = {"project", "pi", "mooring", "qc_enabled", "plots_enabled", "output_dir"}


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
        raise AqdpConfigError(f"Missing required config fields: {', '.join(sorted(missing))}")

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
    )
