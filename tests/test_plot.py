import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqdp.plot import plot_diagnostics, plot_pressure, plot_velocity


@pytest.fixture
def sample_dataset():
    n = 100
    time = pd.date_range("2024-11-12", periods=n, freq="h").values
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "u": ("time", rng.normal(0, 0.1, n)),
            "v": ("time", rng.normal(0, 0.1, n)),
            "w": ("time", rng.normal(0, 0.05, n)),
            "pressure": ("time", np.full(n, 100.0) + rng.normal(0, 0.5, n)),
            "battery_voltage": ("time", np.linspace(12.0, 11.0, n)),
            "pitch": ("time", rng.normal(0, 2, n)),
            "roll": ("time", rng.normal(0, 2, n)),
            "temperature": ("time", np.full(n, 5.0) + rng.normal(0, 0.1, n)),
        },
        coords={"time": time},
    )


def test_plot_velocity_returns_figure(sample_dataset):
    fig = plot_velocity(sample_dataset)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pressure_returns_figure(sample_dataset):
    fig = plot_pressure(sample_dataset)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_diagnostics_returns_figure(sample_dataset):
    fig = plot_diagnostics(sample_dataset)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_velocity_can_save(sample_dataset, tmp_path):
    fig = plot_velocity(sample_dataset)
    output = tmp_path / "vel.png"
    fig.savefig(output)
    assert output.exists()
    plt.close(fig)
