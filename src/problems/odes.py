from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import Tensor, nn

from pinn.core import derivative


@dataclass(frozen=True)
class CoolingConfig:

    ambient_temperature: float = 20.0
    initial_temperature: float = 90.0
    cooling_rate: float = 0.35
    final_time: float = 10.0


def cooling_residual(model: nn.Module, points: Tensor, config: CoolingConfig) -> Tensor:
    """The network predicts ``y = (T - T_ambient) / (T_initial - T_ambient)``.
    In normalized form, Newton's law becomes ``dy/dt + k*y = 0``.
    """
    normalized_temperature = model(points)
    temperature_time_derivative = derivative(normalized_temperature, points)
    return temperature_time_derivative + config.cooling_rate * normalized_temperature


def cooling_initial_condition(model: nn.Module) -> Tensor:
    time_zero = torch.zeros((1, 1), dtype=torch.float32)
    return model(time_zero) - 1.0


def cooling_to_temperature(normalized_temperature: np.ndarray, config: CoolingConfig) -> np.ndarray:
    temperature_range = config.initial_temperature - config.ambient_temperature
    return config.ambient_temperature + temperature_range * normalized_temperature


def cooling_exact(time: np.ndarray, config: CoolingConfig) -> np.ndarray:
    temperature_range = config.initial_temperature - config.ambient_temperature
    return config.ambient_temperature + temperature_range * np.exp(-config.cooling_rate * time)


@dataclass(frozen=True)
class OscillatorConfig:

    damping_ratio: float = 0.15
    natural_frequency: float = 2.0
    initial_position: float = 1.0
    initial_velocity: float = 0.0
    final_time: float = 12.0


def oscillator_residual(
    model: nn.Module,
    points: Tensor,
    config: OscillatorConfig,
) -> Tensor:
    
    prediction = model(points)
    position = prediction[:, 0:1]
    velocity = prediction[:, 1:2]
    position_time_derivative = derivative(prediction, points, component=0)
    velocity_time_derivative = derivative(prediction, points, component=1)

    position_residual = position_time_derivative - velocity
    velocity_residual = (
        velocity_time_derivative
        + 2.0 * config.damping_ratio * velocity
        + config.natural_frequency**2 * position
    )
    return torch.cat((position_residual, velocity_residual), dim=1)


def oscillator_initial_condition(model: nn.Module, config: OscillatorConfig) -> Tensor:
    time_zero = torch.zeros((1, 1), dtype=torch.float32)
    expected_initial_state = torch.tensor(
        [[config.initial_position, config.initial_velocity]],
        dtype=torch.float32,
    )
    return model(time_zero) - expected_initial_state


def oscillator_reference(time: np.ndarray, config: OscillatorConfig) -> np.ndarray:
    def right_hand_side(_: float, state: np.ndarray) -> list[float]:
        position, velocity = state
        acceleration = (
            -2.0 * config.damping_ratio * velocity
            - config.natural_frequency**2 * position
        )
        return [velocity, acceleration]

    solution = solve_ivp(
        right_hand_side,
        t_span=(float(time[0]), float(time[-1])),
        y0=[config.initial_position, config.initial_velocity],
        t_eval=time,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Oscillator reference solver failed: {solution.message}")
    return solution.y.T
