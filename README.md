# aqdp

Python package for converting Nortek Aquadopp text exports (.hdr, .dat, .dia, .ssl) into CF-1.6 compliant NetCDF files.

## Installation

Requires Python 3.12+.

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Generate a starter config file
aqdp init

# Edit config.yml with your project details, then process
aqdp process /path/to/deployment --config config.yml
```

This reads the Aquadopp text files from the deployment directory, applies quality control, generates plots, and writes NetCDF output.

## CLI Reference

```
aqdp init [OPTIONS]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--output PATH` | No | Output path (default: `config.yml`) |

```
aqdp process DEPLOYMENT_DIR --config PATH [OPTIONS]
```

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `DEPLOYMENT_DIR`  | Yes      | Path to the deployment directory |
| `--config PATH`   | Yes      | Path to YAML configuration file |
| `--output-dir PATH` | No    | Override the output directory from config |
| `--no-qc`         | No       | Skip quality control steps |
| `--no-plots`      | No       | Skip plot generation |

## Python API

```python
from pathlib import Path
from aqdp import read_config, read_header, read_dat, flag_by_status, flag_by_range, to_netcdf

config = read_config(Path("config.yaml"))
header = read_header(Path("deployment/"))
ds = read_dat(Path("deployment/"), header)
ds = flag_by_status(ds)
ds = flag_by_range(ds)
to_netcdf(ds, Path("output/"), config)
```

### Public Functions

| Function | Description |
|----------|-------------|
| `generate_config(path)` | Write a starter YAML config file with sensible defaults |
| `read_config(path)` | Parse a YAML configuration file into a `ProcessingConfig` |
| `read_header(path)` | Parse an Aquadopp .hdr file into a `HeaderConfig` |
| `read_dat(path, header)` | Read a .dat measurement file into an `xr.Dataset` |
| `read_dia(path, header)` | Read a .dia diagnostics file into an `xr.Dataset` |
| `read_log(path)` | Read a .ssl log file into an `xr.Dataset` |
| `to_netcdf(ds, output, config)` | Write an `xr.Dataset` to a CF-1.6 NetCDF file |
| `flag_by_status(ds)` | Flag records where `error_code` is nonzero |
| `flag_by_range(ds)` | Flag records with physically unreasonable values |
| `plot_velocity(ds)` | Generate a velocity time-series figure |
| `plot_pressure(ds)` | Generate a pressure time-series figure |
| `plot_diagnostics(ds)` | Generate a diagnostics figure |

### Exceptions

- `AqdpConfigError` — Invalid or missing configuration fields
- `AqdpParsingError` — Data file cannot be parsed

## Configuration

Example YAML configuration file:

```yaml
project: "MOTIVE"
pi: "Jane Smith"
mooring: "Mooring-A"
latitude: 69.5
longitude: -12.3
water_depth: 250.0
instrument_depth: 50.0
qc_enabled: true
plots_enabled: true
output_dir: "output/"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project` | str | Yes | Project name |
| `pi` | str | Yes | Principal investigator |
| `mooring` | str | Yes | Mooring identifier |
| `latitude` | float | No | Latitude coordinate |
| `longitude` | float | No | Longitude coordinate |
| `water_depth` | float | No | Water depth in metres |
| `instrument_depth` | float | No | Instrument depth in metres |
| `qc_enabled` | bool | Yes | Enable QC processing |
| `plots_enabled` | bool | Yes | Enable plot generation |
| `output_dir` | str | Yes | Output directory path |

## Input Data Layout

```
deployment_dir/
├── raw/
│   ├── *.hdr          # Header file with instrument metadata
│   ├── *.dat          # Measurement data (whitespace-separated)
│   └── *.dia          # Diagnostics data (optional)
└── log/
    └── *.ssl          # Instrument log file (optional)
```

## Output

NetCDF files are written to the configured output directory:

- `{deployment_name}_dat.nc` — Measurement data
- `{deployment_name}_dia.nc` — Diagnostics data

Where `deployment_name` comes from the .hdr file (e.g., `18223`).

Files follow **CF-1.6** conventions with global attributes for project metadata, instrument info, and time encoded as seconds since 1970-01-01T00:00:00Z. When plots are enabled, PNG figures are also written to the output directory.

## Data Variables

### Measurement (.dat) — 26 variables

| Variable | Units | CF Standard Name |
|----------|-------|------------------|
| `u` | m/s | `eastward_sea_water_velocity` |
| `v` | m/s | `northward_sea_water_velocity` |
| `w` | m/s | `upward_sea_water_velocity` |
| `amplitude_beam1` | counts | — |
| `amplitude_beam2` | counts | — |
| `amplitude_beam3` | counts | — |
| `battery_voltage` | V | — |
| `sound_speed` | m/s | `speed_of_sound_in_sea_water` |
| `sound_speed_used` | m/s | — |
| `heading` | degrees | — |
| `pitch` | degrees | `platform_pitch` |
| `roll` | degrees | `platform_roll` |
| `pressure` | dbar | `sea_water_pressure` |
| `depth` | m | `depth` |
| `temperature` | degrees_C | `sea_water_temperature` |
| `analog_input_1` | — | — |
| `analog_input_2` | — | — |
| `speed` | m/s | — |
| `direction` | degrees | — |
| `magnetometer_x` | counts | — |
| `magnetometer_y` | counts | — |
| `magnetometer_z` | counts | — |
| `burst_counter` | — | — |
| `ensemble_counter` | — | — |
| `error_code` | — | — |
| `status_code` | — | — |

### Diagnostics (.dia) — 20 variables

Same as the measurement dataset but without `magnetometer_x/y/z`, `burst_counter`, and `ensemble_counter`.

### Log (.ssl)

| Variable | Description |
|----------|-------------|
| `time` | Timestamp |
| `error_code` | Error code string |
| `status_code` | Status code string |
| `level` | Log level (Info, Error, etc.) |
| `description` | Log message text |

## Quality Control

Two QC functions add a `qc_flag` variable to the dataset:

**`flag_by_status`** — Flags records where `error_code` is nonzero.

**`flag_by_range`** — Flags records with physically unreasonable values:

| Variable | Valid Range |
|----------|------------|
| `u`, `v`, `w` | \|value\| ≤ 5.0 m/s |
| `temperature` | −2.0 to 35.0 °C |
| `pressure` | ≥ 0 dbar |

### Flag Values

| Value | Meaning |
|-------|---------|
| 0 | Good |
| 1 | Nonzero error code |
| 2 | Out of range |

## Development

Install with development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check
```
