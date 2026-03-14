"""Quick-look plotting functions for Aquadopp data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr


def plot_velocity(ds: xr.Dataset) -> plt.Figure:
    """Plot u, v, w velocity time series in 3 panels."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, var, label in zip(
        axes, ("u", "v", "w"), ("East (u)", "North (v)", "Up (w)")
    ):
        ax.plot(ds.time.values, ds[var].values, linewidth=0.5)
        ax.set_ylabel(f"{label} [m/s]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Velocity")
    fig.tight_layout()
    return fig


def plot_pressure(ds: xr.Dataset) -> plt.Figure:
    """Plot pressure time series."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(ds.time.values, ds.pressure.values, linewidth=0.5)
    ax.set_ylabel("Pressure [dbar]")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Pressure")
    fig.tight_layout()
    return fig


def plot_diagnostics(ds: xr.Dataset) -> plt.Figure:
    """Plot battery voltage, pitch/roll, and temperature in 3 panels."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ds.time.values, ds.battery_voltage.values, linewidth=0.5)
    axes[0].set_ylabel("Battery [V]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ds.time.values, ds["pitch"].values, linewidth=0.5, label="Pitch")
    axes[1].plot(ds.time.values, ds["roll"].values, linewidth=0.5, label="Roll")
    axes[1].set_ylabel("Tilt [degrees]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ds.time.values, ds.temperature.values, linewidth=0.5)
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Diagnostics")
    fig.tight_layout()
    return fig
