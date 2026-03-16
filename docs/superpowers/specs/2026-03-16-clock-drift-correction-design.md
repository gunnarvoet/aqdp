# Clock Drift Correction

## Problem

Instrument clocks drift over time relative to UTC. The Aquadopp records
timestamps using its internal clock, so the time vector in `.dat` and `.dia`
files accumulates an error that grows linearly over the deployment. We need a
way to specify the drift and correct the timestamps during processing.

## User input

The user records the instrument clock reading and the true UTC time at
instrument recovery. These two values are provided in the YAML config file as
optional fields:

```yaml
# Clock drift correction (optional)
time_instrument: "2025-03-10 14:00:05"
time_utc: "2025-03-10 14:00:00"
```

- Both fields are optional. If neither is present, no correction is applied.
- If only one is provided, `AqdpConfigError` is raised.
- Both are parsed as Python `datetime` objects in `ProcessingConfig`.

## Drift model

Linear drift: zero correction at deployment start, full measured drift at the
recovery time point (`time_instrument`).

```
drift = time_utc - time_instrument
elapsed = time - deployment_start
total_span = time_instrument - deployment_start
correction = drift * (elapsed / total_span)
corrected_time = time + correction
```

`deployment_start` comes from `header.deployment_time` (parsed from the `.hdr`
file). `time_instrument` is used as the interpolation endpoint (not the last
record) because that is when the drift was actually measured. Records after
`time_instrument` (if any) are extrapolated.

**Edge cases:**
- If the dataset has fewer than 2 records or `total_span` is zero, skip the
  correction and log a warning.
- If `|drift|` exceeds 1 hour, log a warning — large drifts may indicate
  swapped `time_instrument`/`time_utc` values or a typo.

## Architecture

### Config changes (`config.py`)

Two new optional fields on `ProcessingConfig`:

```python
time_instrument: datetime | None  # default None
time_utc: datetime | None         # default None
```

Validation: if exactly one is set, raise `AqdpConfigError`.

`read_config` parses the YAML strings with `datetime.fromisoformat()`.

The starter template adds commented-out placeholder lines.

### Correction logic (`io.py`)

A private helper `_apply_clock_drift(ds, header, config)` implements the linear
correction:

1. Compute `drift = config.time_utc - config.time_instrument` (Python
   `timedelta`). Convert to `np.timedelta64` for vectorized arithmetic.
2. Get `deployment_start` from `header.deployment_time`. Convert to
   `np.datetime64` for compatibility with the time coordinate.
3. Convert `config.time_instrument` to `np.datetime64` for the interpolation
   endpoint.
4. Guard: if dataset has fewer than 2 records or `total_span` is zero, return
   unchanged. Warn if `|drift|` exceeds 1 hour.
5. Compute fractional elapsed time for each record and apply proportional
   correction to the time coordinate.
6. Add global attributes to the dataset:
   - `clock_drift_instrument_time` — instrument reading at recovery
   - `clock_drift_utc_time` — true UTC at recovery
   - `clock_drift_applied` — `True`

Called at the end of `read_dat` and `read_dia` when config is provided and
contains drift fields. No-op otherwise. Because correction happens inside the
read functions, it is applied before QC and plotting in the pipeline.

### Function signature changes (`io.py`)

```python
def read_dat(path: Path, header: HeaderConfig, config: ProcessingConfig | None = None) -> xr.Dataset:
def read_dia(path: Path, header: HeaderConfig, config: ProcessingConfig | None = None) -> xr.Dataset:
```

Backward-compatible — existing calls without `config` are unchanged.

### CLI changes (`cli.py`)

Pass `cfg` to `read_dat` and `read_dia`. Log when drift correction is applied.

### Starter template (`config.py`)

Add commented-out lines to `_STARTER_TEMPLATE`:

```yaml
# Clock drift correction
# time_instrument: "YYYY-MM-DD HH:MM:SS"
# time_utc: "YYYY-MM-DD HH:MM:SS"
```

### README updates

- **Configuration section**: Add `time_instrument` and `time_utc` to the
  example YAML and config field table.
- **Python API section**: Update `read_dat` and `read_dia` signatures in the
  public functions table to show the optional `config` parameter.
- **New section**: Add a "Clock Drift Correction" section (after Configuration,
  before Input Data Layout) explaining the feature, the linear drift model, and
  usage.

## Traceability

Corrected datasets include global attributes recording the drift parameters.
This appears in the NetCDF output for downstream consumers to verify.

## Modules NOT changed

- `qc.py` — operates on the dataset after correction
- `plot.py` — operates on the dataset after correction
- `to_netcdf` — writes whatever timestamps are in the dataset
- `__init__.py` — no new public API needed

## Tests

1. **Config parsing** — both fields present, both absent, only one present
   (error)
2. **Correction math** — synthetic dataset with known timestamps, verify linear
   interpolation
3. **No-op when absent** — `config=None` or no drift fields returns unchanged
   timestamps
4. **Global attributes** — verify `clock_drift_*` attributes after correction
5. **Edge cases** — single-record dataset (no-op), zero total_span (no-op),
   large drift warning
6. **Integration** — end-to-end with drift config, verify NetCDF output
