from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.train_cooling import CoolingTrainingConfig, train_cooling


def test_short_cooling_run_returns_finite_outputs() -> None:
    config = CoolingTrainingConfig(
        epochs=5,
        collocation_points=12,
        hidden_layers=(8, 8),
        print_every=0,
    )

    run = train_cooling(config)

    assert len(run.history.total) == config.epochs
    assert np.isfinite(run.rmse)
    assert np.all(np.isfinite(run.pinn_temperature))
    assert np.all(np.isfinite(run.exact_temperature))
