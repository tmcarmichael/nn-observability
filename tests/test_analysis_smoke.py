"""Smoke tests for CPU analysis scripts.

Verifies each script runs without crashing against the committed
results JSONs. Does not check output values (the statistical tests
are their own validation). Catches broken imports, missing data
files, and schema changes after refactors.

Run: uv run pytest tests/test_analysis_smoke.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
REPO_ROOT = ANALYSIS_DIR.parent

SCRIPTS = [
    "meta_regression.py",
    "ancova_family.py",
    "permutation_test.py",
    "selectivity.py",
    "pearson_vs_spearman.py",
    "loocv_scaling.py",
    "funnel_plot.py",
    "exclusive_catch_rates.py",
    "held_out_split.py",
]


@pytest.fixture(params=SCRIPTS, ids=lambda s: s.replace(".py", ""))
def script_path(request):
    return ANALYSIS_DIR / request.param


def test_script_runs(script_path):
    """Script should exit 0 against committed data."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script_path.name} failed (exit {result.returncode}):\nstderr: {result.stderr[-500:]}"
    )
