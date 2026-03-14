"""aqdp — Nortek Aquadopp data processing package."""

from aqdp.config import AqdpConfigError, ProcessingConfig, read_config
from aqdp.io import (
    AqdpParsingError,
    HeaderConfig,
    read_dat,
    read_dia,
    read_header,
    read_log,
    to_netcdf,
)
from aqdp.plot import plot_diagnostics, plot_pressure, plot_velocity
from aqdp.qc import flag_by_range, flag_by_status

__all__ = [
    "AqdpConfigError",
    "AqdpParsingError",
    "HeaderConfig",
    "ProcessingConfig",
    "flag_by_range",
    "flag_by_status",
    "plot_diagnostics",
    "plot_pressure",
    "plot_velocity",
    "read_config",
    "read_dat",
    "read_dia",
    "read_header",
    "read_log",
    "to_netcdf",
]
