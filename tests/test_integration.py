from pathlib import Path

import numpy as np
import xarray as xr

from aqdp import (
    ProcessingConfig,
    flag_by_range,
    flag_by_status,
    read_dat,
    read_header,
    to_netcdf,
)


def test_full_pipeline_dat_roundtrip(deployment_dir: Path, tmp_path: Path):
    """Read .dat -> QC -> write NetCDF -> read back -> verify."""
    header = read_header(deployment_dir)
    ds = read_dat(deployment_dir, header)
    ds = flag_by_status(ds)
    ds = flag_by_range(ds)

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
    )
    output = tmp_path / "test_dat.nc"
    to_netcdf(ds, output, config)

    result = xr.open_dataset(output)
    assert result.attrs["Conventions"] == "CF-1.6"
    assert result.attrs["project"] == "TEST"
    assert "qc_flag" in result
    np.testing.assert_allclose(result.u.values[:10], ds.u.values[:10])
    assert result.sizes["time"] == ds.sizes["time"]
    result.close()
