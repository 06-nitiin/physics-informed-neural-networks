from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.train_oscillator import OscillatorTrainingConfig, train_oscillator
from problems.odes import OscillatorConfig, oscillator_reference

def test_reference_starts_at_configured_initial_state() -> None:

    config = OscillatorConfig()
    time = np.linspace(0.0, config.final_time, 25)

    reference = oscillator_reference(time, config)

    assert reference.shape == (len(time), 2)
    assert np.allclose(
        reference[0],
        [config.initial_position, config.initial_velocity],
    )

def test_short_oscillator_run_returns_finite_outputs() -> None:

    config = OscillatorTrainingConfig(
        epochs=5,
        collocation_points=16,
        hidden_layers=(8, 8),
        print_every=0,
    )

    run = train_oscillator(config)

    assert run.pinn_state.shape == run.reference_state.shape
    assert run.pinn_state.shape[1] == 2
    assert len(run.history.total) == config.epochs
    assert np.isfinite(run.rmse)
    assert np.all(np.isfinite(run.pinn_state))
