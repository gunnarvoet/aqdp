from pathlib import Path

import pytest


@pytest.fixture
def deployment_dir() -> Path:
    return Path(__file__).parent / "data" / "SN18223"


@pytest.fixture
def raw_dir(deployment_dir: Path) -> Path:
    return deployment_dir / "raw"


@pytest.fixture
def log_dir(deployment_dir: Path) -> Path:
    return deployment_dir / "log"
