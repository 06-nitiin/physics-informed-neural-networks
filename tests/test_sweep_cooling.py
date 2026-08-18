"""Tests for the controlled Newton-cooling experiment sweep."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.sweep_cooling import CoolingExperimentCase, run_sweep, write_results


def test_short_controlled_sweep_records_finite_metrics(tmp_path: Path) -> None:
    """A one-case sweep should record a complete, finite experiment row."""
    case = CoolingExperimentCase("test_case", (8, 8), "tanh")

    rows = run_sweep(epochs=3, cases=(case,))
    output_path = tmp_path / "sweep.csv"
    write_results(rows, output_path)

    assert len(rows) == 1
    assert rows[0]["case"] == "test_case"
    assert rows[0]["activation"] == "tanh"
    assert np.isfinite(rows[0]["final_loss"])
    assert np.isfinite(rows[0]["rmse"])
    assert output_path.exists()
    assert "test_case" in output_path.read_text()
