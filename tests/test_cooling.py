from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problems.odes import CoolingConfig, cooling_exact, cooling_to_temperature

def test_analytical_solution_matches_initial_temperature() -> None:
    config = CoolingConfig()

    temperature_at_zero = cooling_exact(np.array([0.0]), config)

    assert np.isclose(temperature_at_zero[0], config.initial_temperature)


def test_analytical_solution_cools_toward_ambient_temperature() -> None:
    config = CoolingConfig()

    temperature_at_final_time = cooling_exact(np.array([config.final_time]), config)

    assert temperature_at_final_time[0] < config.initial_temperature
    assert temperature_at_final_time[0] > config.ambient_temperature


def test_normalized_temperature_conversion() -> None:
    config = CoolingConfig()

    temperatures = cooling_to_temperature(np.array([1.0, 0.0]), config)

    assert np.allclose(
        temperatures,
        [config.initial_temperature, config.ambient_temperature],
    )
