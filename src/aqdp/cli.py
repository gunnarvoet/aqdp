"""CLI entry point for aqdp."""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")


@click.group()
def main():
    """aqdp — Nortek Aquadopp data processing."""


@main.command()
@click.argument("deployment_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory",
)
@click.option("--no-qc", is_flag=True, help="Skip QC steps")
@click.option("--no-plots", is_flag=True, help="Skip plot generation")
def process(deployment_dir, config, output_dir, no_qc, no_plots):
    """Process an Aquadopp deployment directory."""
    from aqdp.config import read_config
    from aqdp.io import (
        AqdpParsingError,
        read_dat,
        read_dia,
        read_header,
        read_log,
        to_netcdf,
    )
    from aqdp.plot import plot_diagnostics, plot_pressure, plot_velocity
    from aqdp.qc import flag_by_range, flag_by_status

    cfg = read_config(config)

    # CLI overrides
    if output_dir is not None:
        cfg.output_dir = output_dir
    if no_qc:
        cfg.qc_enabled = False
    if no_plots:
        cfg.plots_enabled = False

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse header
    header = read_header(deployment_dir)
    name = header.deployment_name
    click.echo(f"Processing deployment {name} (SN {header.serial_number})")

    # Read measurement data
    ds_dat = read_dat(deployment_dir, header)
    click.echo(f"  Read {ds_dat.sizes['time']} measurement records")

    # Read diagnostics (optional)
    ds_dia = None
    try:
        ds_dia = read_dia(deployment_dir, header)
        click.echo(f"  Read {ds_dia.sizes['time']} diagnostics records")
    except AqdpParsingError:
        click.echo("  Warning: No .dia file found, skipping diagnostics")

    # Read log (optional)
    try:
        ds_log = read_log(deployment_dir)
        click.echo(f"  Read {ds_log.sizes['time']} log entries")
    except AqdpParsingError:
        click.echo("  Warning: No .ssl file found, skipping log")

    # QC
    if cfg.qc_enabled:
        ds_dat = flag_by_status(ds_dat)
        ds_dat = flag_by_range(ds_dat)
        click.echo("  Applied QC flags to measurements")
        if ds_dia is not None:
            ds_dia = flag_by_status(ds_dia)
            ds_dia = flag_by_range(ds_dia)
            click.echo("  Applied QC flags to diagnostics")

    # Plots
    if cfg.plots_enabled:
        import matplotlib.pyplot as plt

        fig = plot_velocity(ds_dat)
        fig.savefig(cfg.output_dir / f"{name}_velocity.png", dpi=150)
        plt.close(fig)

        fig = plot_pressure(ds_dat)
        fig.savefig(cfg.output_dir / f"{name}_pressure.png", dpi=150)
        plt.close(fig)

        if ds_dia is not None:
            fig = plot_diagnostics(ds_dia)
            fig.savefig(cfg.output_dir / f"{name}_diagnostics.png", dpi=150)
            plt.close(fig)

        click.echo(f"  Saved plots to {cfg.output_dir}")

    # Write NetCDF
    to_netcdf(ds_dat, cfg.output_dir / f"{name}_dat.nc", cfg)
    click.echo(f"  Wrote {cfg.output_dir / f'{name}_dat.nc'}")

    if ds_dia is not None:
        to_netcdf(ds_dia, cfg.output_dir / f"{name}_dia.nc", cfg)
        click.echo(f"  Wrote {cfg.output_dir / f'{name}_dia.nc'}")

    click.echo("Done.")
