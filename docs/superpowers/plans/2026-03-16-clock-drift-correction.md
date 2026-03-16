# Clock Drift Correction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional linear clock drift correction to the processing pipeline, controlled via two YAML config fields.

**Architecture:** Two new optional `datetime` fields on `ProcessingConfig` (`time_instrument`, `time_utc`). A private helper `_apply_clock_drift` in `io.py` performs linear interpolation. Called inside `read_dat`/`read_dia` when config is provided with drift fields. No-op otherwise.

**Tech Stack:** Python 3.12+, xarray, numpy, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-clock-drift-correction-design.md`

---

## Chunk 1: Config Changes

### Task 1: Add clock drift fields to ProcessingConfig and read_config

**Files:**
- Modify: `src/aqdp/config.py:1-112`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config parsing with drift fields**

Add to `tests/test_config.py`:

```python
from datetime import datetime


def test_read_config_drift_fields_parsed(tmp_path: Path):
    config = {
        "project": "TEST",
        "pi": "Test PI",
        "mooring": "Test Mooring",
        "qc_enabled": False,
        "plots_enabled": False,
        "output_dir": "out/",
        "plots_dir": "fig/",
        "time_instrument": "2025-03-10 14:00:05",
        "time_utc": "2025-03-10 14:00:00",
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    result = read_config(p)
    assert result.time_instrument == datetime(2025, 3, 10, 14, 0, 5)
    assert result.time_utc == datetime(2025, 3, 10, 14, 0, 0)


def test_read_config_drift_fields_default_none(tmp_path: Path):
    config = {
        "project": "TEST",
        "pi": "Test PI",
        "mooring": "Test Mooring",
        "qc_enabled": False,
        "plots_enabled": False,
        "output_dir": "out/",
        "plots_dir": "fig/",
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    result = read_config(p)
    assert result.time_instrument is None
    assert result.time_utc is None


def test_read_config_drift_only_one_field_raises(tmp_path: Path):
    config = {
        "project": "TEST",
        "pi": "Test PI",
        "mooring": "Test Mooring",
        "qc_enabled": False,
        "plots_enabled": False,
        "output_dir": "out/",
        "plots_dir": "fig/",
        "time_instrument": "2025-03-10 14:00:05",
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(config))
    with pytest.raises(AqdpConfigError):
        read_config(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v -k "drift"`
Expected: FAIL — `ProcessingConfig` does not accept `time_instrument`/`time_utc`

- [ ] **Step 3: Implement config changes**

In `src/aqdp/config.py`:

1. Add `from datetime import datetime` import.

2. Add two fields to `ProcessingConfig` (after `plots_dir`):

```python
time_instrument: datetime | None = None
time_utc: datetime | None = None
```

3. In `read_config`, after the required-fields check, add drift field parsing and validation:

```python
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
```

4. Pass them to the `ProcessingConfig` constructor:

```python
return ProcessingConfig(
    ...existing fields...,
    time_instrument=time_instrument,
    time_utc=time_utc,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: ALL PASS (including existing tests — they must still work with the new `None` defaults)

- [ ] **Step 5: Update starter template**

In `src/aqdp/config.py`, add to `_STARTER_TEMPLATE` (after the `bottom_depth` comment block, before the blank line before `qc_enabled`):

```yaml

# Clock drift correction
# time_instrument: "YYYY-MM-DD HH:MM:SS"
# time_utc: "YYYY-MM-DD HH:MM:SS"
```

- [ ] **Step 6: Fix test_generate_config_roundtrips**

The existing `test_generate_config_roundtrips` test constructs a `ProcessingConfig` from the template. It needs to verify the new fields default to `None`:

```python
assert config.time_instrument is None
assert config.time_utc is None
```

Add these two asserts to the end of the existing test.

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/aqdp/config.py tests/test_config.py
git commit -m "Add time_instrument and time_utc config fields for clock drift"
```

---

## Chunk 2: Clock Drift Correction Logic

### Task 2: Implement _apply_clock_drift helper

**Files:**
- Modify: `src/aqdp/io.py:323-442`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write failing tests for drift correction**

Add to `tests/test_io.py`. These tests use a synthetic dataset to verify the correction math independently of file I/O:

```python
import warnings
from datetime import datetime, timedelta

import pandas as pd

from aqdp.config import ProcessingConfig
from aqdp.io import _apply_clock_drift, HeaderConfig


def _make_header(deployment_time: datetime) -> HeaderConfig:
    """Create a minimal HeaderConfig for testing."""
    return HeaderConfig(
        serial_number="TEST",
        deployment_name="test",
        coordinate_system="ENU",
        measurement_interval=60,
        blanking_distance=0.5,
        n_beams=3,
        head_frequency=2000,
        salinity=35.0,
        deployment_time=deployment_time,
        transformation_matrix=np.eye(3),
        comments="",
        n_measurements=10,
        transmit_pulse_length=1.0,
        sampling_rate="1 Hz",
        average_interval=60,
        compass_update_rate=1,
        firmware_version="1.0",
        software_version="1.0",
        pressure_sensor_calibration=[0, 0, 0, 0],
        n_pings_per_burst=1,
        diagnostics_interval=720,
        diagnostics_n_samples=20,
        head_serial_number="H001",
    )


def _make_config(
    time_instrument: datetime | None = None,
    time_utc: datetime | None = None,
) -> ProcessingConfig:
    """Create a minimal ProcessingConfig for testing."""
    return ProcessingConfig(
        project="TEST",
        pi="Test",
        mooring="Test",
        latitude=None,
        longitude=None,
        bottom_depth=None,
        qc_enabled=False,
        plots_enabled=False,
        output_dir=Path("out/"),
        plots_dir=Path("fig/"),
        time_instrument=time_instrument,
        time_utc=time_utc,
    )


def _make_dataset(times: list[str]) -> xr.Dataset:
    """Create a minimal dataset with a time coordinate."""
    time_arr = pd.to_datetime(times)
    return xr.Dataset(
        {"u": ("time", np.ones(len(times)))},
        coords={"time": time_arr.values},
    )


def test_apply_clock_drift_linear_correction():
    """Verify linear interpolation: zero at start, full drift at instrument time."""
    deploy_time = datetime(2025, 1, 1, 0, 0, 0)
    header = _make_header(deploy_time)
    # Instrument clock is 10 seconds ahead at recovery
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 0, 0, 10),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset([
        "2025-01-01T00:00:00",  # deployment start
        "2025-01-01T12:00:05",  # midpoint (in instrument time)
        "2025-01-02T00:00:10",  # recovery (in instrument time)
    ])

    result = _apply_clock_drift(ds, header, config)

    expected = pd.to_datetime([
        "2025-01-01T00:00:00",   # zero correction at start
        "2025-01-01T12:00:00",   # -5s correction at midpoint
        "2025-01-02T00:00:00",   # -10s correction at recovery
    ]).values
    np.testing.assert_array_equal(result.time.values, expected)


def test_apply_clock_drift_adds_global_attributes():
    deploy_time = datetime(2025, 1, 1, 0, 0, 0)
    header = _make_header(deploy_time)
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 0, 0, 10),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T00:00:10"])

    result = _apply_clock_drift(ds, header, config)

    assert result.attrs["clock_drift_applied"] is True
    assert result.attrs["clock_drift_instrument_time"] == "2025-01-02 00:00:10"
    assert result.attrs["clock_drift_utc_time"] == "2025-01-02 00:00:00"


def test_apply_clock_drift_noop_when_no_drift_fields():
    header = _make_header(datetime(2025, 1, 1))
    config = _make_config()  # no drift fields
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T00:00:00"])

    result = _apply_clock_drift(ds, header, config)

    np.testing.assert_array_equal(result.time.values, ds.time.values)
    assert "clock_drift_applied" not in result.attrs


def test_apply_clock_drift_noop_when_config_none():
    header = _make_header(datetime(2025, 1, 1))
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T00:00:00"])
    original_times = ds.time.values.copy()

    result = _apply_clock_drift(ds, header, None)

    np.testing.assert_array_equal(result.time.values, original_times)


def test_apply_clock_drift_single_record_noop():
    header = _make_header(datetime(2025, 1, 1))
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 0, 0, 10),
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T12:00:00"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _apply_clock_drift(ds, header, config)
        assert any("Cannot apply" in str(warning.message) for warning in w)

    np.testing.assert_array_equal(result.time.values, ds.time.values)


def test_apply_clock_drift_zero_total_span_noop():
    """total_span is zero when deployment_time == time_instrument."""
    deploy_time = datetime(2025, 1, 1, 0, 0, 0)
    header = _make_header(deploy_time)
    config = _make_config(
        time_instrument=datetime(2025, 1, 1, 0, 0, 0),  # same as deploy
        time_utc=datetime(2025, 1, 1, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-01T12:00:00"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _apply_clock_drift(ds, header, config)
        assert any("total_span is zero" in str(warning.message) for warning in w)

    np.testing.assert_array_equal(result.time.values, ds.time.values)


def test_apply_clock_drift_large_drift_warns():
    header = _make_header(datetime(2025, 1, 1))
    config = _make_config(
        time_instrument=datetime(2025, 1, 2, 2, 0, 0),  # 2 hours off
        time_utc=datetime(2025, 1, 2, 0, 0, 0),
    )
    ds = _make_dataset(["2025-01-01T00:00:00", "2025-01-02T02:00:00"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _apply_clock_drift(ds, header, config)
        assert any("exceeds 1 hour" in str(warning.message) for warning in w)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io.py -v -k "clock_drift"`
Expected: FAIL — `_apply_clock_drift` does not exist

- [ ] **Step 3: Implement _apply_clock_drift**

Add to `src/aqdp/io.py`, before `read_dat` (around line 322):

```python
import warnings


def _apply_clock_drift(
    ds: xr.Dataset,
    header: HeaderConfig,
    config: ProcessingConfig | None,
) -> xr.Dataset:
    """Apply linear clock drift correction to the time coordinate.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``time`` coordinate.
    header : HeaderConfig
        Parsed header metadata (provides deployment_time).
    config : ProcessingConfig or None
        Processing config. If None or drift fields are absent, returns
        *ds* unchanged.

    Returns
    -------
    xr.Dataset
        Dataset with corrected time coordinate (or unchanged if no
        drift info).
    """
    if config is None or config.time_instrument is None:
        return ds

    drift = config.time_utc - config.time_instrument
    if abs(drift) > timedelta(hours=1):
        warnings.warn(
            f"Clock drift ({drift}) exceeds 1 hour — check "
            f"time_instrument/time_utc for errors",
            stacklevel=2,
        )

    if ds.sizes["time"] < 2:
        warnings.warn(
            "Cannot apply clock drift correction to dataset with fewer "
            "than 2 records",
            stacklevel=2,
        )
        return ds

    deploy_start = np.datetime64(header.deployment_time)
    instrument_end = np.datetime64(config.time_instrument)
    total_span = instrument_end - deploy_start

    if total_span == np.timedelta64(0):
        warnings.warn(
            "Cannot apply clock drift correction: total_span is zero",
            stacklevel=2,
        )
        return ds

    drift_ns = np.timedelta64(drift)
    elapsed = ds.time.values - deploy_start
    fraction = elapsed / total_span
    correction = fraction * drift_ns

    ds = ds.assign_coords(time=ds.time.values + correction)
    ds.attrs["clock_drift_applied"] = True
    ds.attrs["clock_drift_instrument_time"] = str(config.time_instrument)
    ds.attrs["clock_drift_utc_time"] = str(config.time_utc)
    return ds
```

Also add `from datetime import timedelta` to the imports at the top of `io.py` (there's already `from datetime import datetime`; change to `from datetime import datetime, timedelta`).

Add the `ProcessingConfig` import: `from aqdp.config import ProcessingConfig`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_io.py -v -k "clock_drift"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/aqdp/io.py tests/test_io.py
git commit -m "Add _apply_clock_drift helper with linear correction"
```

### Task 3: Wire drift correction into read_dat and read_dia

**Files:**
- Modify: `src/aqdp/io.py` (signatures and return of `read_dat` and `read_dia`)
- Test: `tests/test_io.py`

Note: Line numbers below reference the file *before* Task 2's insertion. After Task 2, find these functions by name — they will have shifted down by ~60 lines.

- [ ] **Step 1: Write failing tests for read_dat/read_dia with config**

Add to `tests/test_io.py`:

```python
def test_read_dat_with_drift_config_corrects_time(deployment_dir: Path):
    header = read_header(deployment_dir)
    # Instrument clock 10s ahead at a time after the last record
    last_time = datetime(2024, 11, 20, 0, 0, 10)
    config = _make_config(
        time_instrument=last_time,
        time_utc=datetime(2024, 11, 20, 0, 0, 0),
    )
    ds_corrected = read_dat(deployment_dir, header, config)
    ds_raw = read_dat(deployment_dir, header)

    # Corrected times should differ from raw
    assert not np.array_equal(ds_corrected.time.values, ds_raw.time.values)
    assert ds_corrected.attrs["clock_drift_applied"] is True


def test_read_dat_without_config_unchanged(deployment_dir: Path):
    header = read_header(deployment_dir)
    ds1 = read_dat(deployment_dir, header)
    ds2 = read_dat(deployment_dir, header, None)

    np.testing.assert_array_equal(ds1.time.values, ds2.time.values)


def test_read_dia_with_drift_config_corrects_time(deployment_dir: Path):
    header = read_header(deployment_dir)
    last_time = datetime(2024, 11, 20, 0, 0, 10)
    config = _make_config(
        time_instrument=last_time,
        time_utc=datetime(2024, 11, 20, 0, 0, 0),
    )
    ds_corrected = read_dia(deployment_dir, header, config)
    ds_raw = read_dia(deployment_dir, header)

    assert not np.array_equal(ds_corrected.time.values, ds_raw.time.values)
    assert ds_corrected.attrs["clock_drift_applied"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io.py -v -k "read_dat_with_drift or read_dat_without or read_dia_with_drift"`
Expected: FAIL — `read_dat()` and `read_dia()` don't accept `config` parameter

- [ ] **Step 3: Update read_dat and read_dia signatures**

In `src/aqdp/io.py`:

Update `read_dat` signature (line 323):
```python
def read_dat(path: Path, header: HeaderConfig, config: ProcessingConfig | None = None) -> xr.Dataset:
```

Before the `return ds` at end of `read_dat` (line 368), add:
```python
    ds = _apply_clock_drift(ds, header, config)
    return ds
```
(Replace the existing `return ds`.)

Update `read_dia` signature (line 397):
```python
def read_dia(path: Path, header: HeaderConfig, config: ProcessingConfig | None = None) -> xr.Dataset:
```

Before the `return ds` at end of `read_dia` (line 442), add:
```python
    ds = _apply_clock_drift(ds, header, config)
    return ds
```
(Replace the existing `return ds`.)

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS (existing tests unaffected because `config` defaults to `None`)

- [ ] **Step 5: Commit**

```bash
git add src/aqdp/io.py tests/test_io.py
git commit -m "Wire clock drift correction into read_dat and read_dia"
```

---

## Chunk 3: CLI, Template, README, Integration

### Task 4: Update CLI to pass config to read functions

**Files:**
- Modify: `src/aqdp/cli.py:119,125`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Update CLI calls**

In `src/aqdp/cli.py`, change line 119:
```python
ds_dat = read_dat(deployment_dir, header, cfg)
```

Change line 125:
```python
ds_dia = read_dia(deployment_dir, header, cfg)
```

After line 120 (`click.echo(f"  Read {ds_dat.sizes['time']} measurement records")`), add (note: the `ProcessingConfig` is `cfg` in this file, not `config` which is the path):
```python
    if ds_dat.attrs.get("clock_drift_applied"):
        drift = cfg.time_utc - cfg.time_instrument
        click.echo(f"  Applied clock drift correction (drift: {drift.total_seconds():.1f}s)")
```

- [ ] **Step 2: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/aqdp/cli.py
git commit -m "Pass config to read_dat/read_dia in CLI for clock drift"
```

### Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add drift fields to config example**

In the Configuration section example YAML (around line 88-99), add after `bottom_depth: 250.0`:
```yaml
# time_instrument: "2025-03-10 14:00:05"
# time_utc: "2025-03-10 14:00:00"
```

- [ ] **Step 2: Add drift fields to config table**

Add two rows to the configuration field table (after `bottom_depth`):
```
| `time_instrument` | str | No | Instrument clock reading at recovery (ISO format) |
| `time_utc` | str | No | True UTC time at recovery (ISO format) |
```

- [ ] **Step 3: Update Python API function signatures**

Update the public functions table entries for `read_dat` and `read_dia`:
```
| `read_dat(path, header, config=None)` | Read a .dat measurement file into an `xr.Dataset` |
| `read_dia(path, header, config=None)` | Read a .dia diagnostics file into an `xr.Dataset` |
```

- [ ] **Step 4: Update Python API example**

Update the Python API code example (around line 50-59) to show config being passed:
```python
config = read_config(Path("config.yaml"))
header = read_header(Path("deployment/"))
ds = read_dat(Path("deployment/"), header, config)
```

- [ ] **Step 5: Add Clock Drift Correction section**

Insert a new section after the Configuration section (before "Input Data Layout"):

```markdown
## Clock Drift Correction

Instrument clocks drift over time. To correct for this, record the instrument's
clock reading and the true UTC time at recovery, then add them to the config:

\```yaml
time_instrument: "2025-03-10 14:00:05"
time_utc: "2025-03-10 14:00:00"
\```

A linear correction is applied: zero at deployment start, scaling to the full
measured drift at the recovery time. The correction is applied automatically
during data reading when both fields are present.

Corrected datasets include global attributes (`clock_drift_applied`,
`clock_drift_instrument_time`, `clock_drift_utc_time`) for traceability.
```

(Remove the backslashes before the triple backticks — they are escapes for this plan document only.)

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Document clock drift correction in README"
```

### Task 6: Integration test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_integration.py`:

```python
from datetime import datetime


def test_full_pipeline_with_clock_drift(deployment_dir: Path, tmp_path: Path):
    """Read .dat with drift correction -> QC -> write NetCDF -> read back."""
    header = read_header(deployment_dir)

    config = ProcessingConfig(
        project="TEST",
        pi="Test",
        mooring="Test",
        latitude=69.5,
        longitude=-12.3,
        bottom_depth=250.0,
        qc_enabled=True,
        plots_enabled=False,
        output_dir=tmp_path,
        plots_dir=tmp_path,
        time_instrument=datetime(2024, 11, 20, 0, 0, 10),
        time_utc=datetime(2024, 11, 20, 0, 0, 0),
    )

    ds = read_dat(deployment_dir, header, config)
    assert ds.attrs["clock_drift_applied"] is True

    ds = flag_by_status(ds)
    ds = flag_by_range(ds)

    output = tmp_path / "test_drift_dat.nc"
    to_netcdf(ds, output, config)

    result = xr.open_dataset(output)
    assert result.attrs["clock_drift_applied"] == True
    assert result.attrs["clock_drift_instrument_time"] == "2024-11-20 00:00:10"
    assert result.attrs["clock_drift_utc_time"] == "2024-11-20 00:00:00"
    assert result.sizes["time"] == ds.sizes["time"]
    result.close()
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "Add integration test for clock drift correction"
```
