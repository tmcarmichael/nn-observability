"""Shared test configuration and fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hypothesis import HealthCheck, settings

# Hypothesis profiles: dev runs fast, CI runs thorough.
settings.register_profile("ci", max_examples=200)
settings.register_profile(
    "dev",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)


@pytest.fixture()
def results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


@pytest.fixture(autouse=True)
def _torch_deterministic() -> None:
    """Seed torch for reproducible test runs."""
    torch.manual_seed(42)
