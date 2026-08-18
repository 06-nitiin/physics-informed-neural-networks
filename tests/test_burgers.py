from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.train_burgers import BurgersTrainingConfig, train_burgers
from problems.burgers import BurgersConfig, burgers_initial_profile, burgers_reference


def test_reference_matches_initial_and_boundary_conditions() -> None:
    config = BurgersConfig()
    x, time, solution = burgers_reference(config, nx=31, nt=31)

    assert solution.shape == (len(time), len(x))
    assert np.allclose(solution[0], burgers_initial_profile(x))
    assert np.allclose(solution[:, 0], 0.0)
    assert np.allclose(solution[:, -1], 0.0)
    assert np.all(np.isfinite(solution))


def test_short_burgers_run_returns_finite_outputs() -> None:
    config = BurgersTrainingConfig(
        epochs=3,
        collocation_x=8,
        collocation_t=8,
        initial_condition_points=8,
        boundary_condition_points=8,
        hidden_layers=(8, 8),
        reference_nx=21,
        reference_nt=21,
        print_every=0,
    )

    run = train_burgers(config)

    assert run.pinn_solution.shape == run.reference_solution.shape
    assert len(run.history.total) == config.epochs
    assert np.isfinite(run.rmse)
    assert np.all(np.isfinite(run.pinn_solution))
