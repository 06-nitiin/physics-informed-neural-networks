from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from pinn.core import derivative, second_derivative


@dataclass(frozen=True)
class BurgersConfig:

    viscosity: float = 0.01 / np.pi
    x_min: float = -1.0
    x_max: float = 1.0
    final_time: float = 1.0


def burgers_initial_profile(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def burgers_residual(model: nn.Module, points: Tensor, config: BurgersConfig) -> Tensor:
    prediction = model(points)
    u = prediction[:, 0:1]
    u_x = derivative(prediction, points, component=0)[:, 0:1]
    u_t = derivative(prediction, points, component=0)[:, 1:2]
    u_xx = second_derivative(prediction, points, component=0, coordinate=0)
    return u_t + u * u_x - config.viscosity * u_xx


def burgers_conditions(
    model: nn.Module,
    config: BurgersConfig,
    initial_points: int = 64,
    boundary_points: int = 64,
) -> Tensor:
    x = torch.linspace(config.x_min, config.x_max, initial_points).reshape(-1, 1)
    initial_coordinates = torch.cat((x, torch.zeros_like(x)), dim=1)
    expected_initial_profile = -torch.sin(torch.pi * x)
    initial_error = model(initial_coordinates) - expected_initial_profile

    time = torch.linspace(0.0, config.final_time, boundary_points).reshape(-1, 1)
    left_boundary = torch.cat((torch.full_like(time, config.x_min), time), dim=1)
    right_boundary = torch.cat((torch.full_like(time, config.x_max), time), dim=1)
    boundary_error = torch.cat((model(left_boundary), model(right_boundary)), dim=0)

    return torch.cat((initial_error, boundary_error), dim=0)


def burgers_reference(
    config: BurgersConfig,
    nx: int = 121,
    nt: int = 201,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nx < 3 or nt < 2:
        raise ValueError("Burgers reference requires nx >= 3 and nt >= 2.")

    x = np.linspace(config.x_min, config.x_max, nx)
    dx = x[1] - x[0]
    requested_step = config.final_time / (nt - 1)
    advection_step = 0.2 * dx
    diffusion_step = 0.2 * dx**2 / config.viscosity
    dt = min(requested_step, advection_step, diffusion_step)
    steps = int(np.ceil(config.final_time / dt))
    dt = config.final_time / steps

    solution = burgers_initial_profile(x).astype(float)
    snapshots = [solution.copy()]
    time_values = [0.0]

    for step in range(steps):
        previous = solution.copy()
        speed = previous[1:-1]
        backward_difference = (previous[1:-1] - previous[:-2]) / dx
        forward_difference = (previous[2:] - previous[1:-1]) / dx
        upwind_advection = np.where(
            speed >= 0.0,
            speed * backward_difference,
            speed * forward_difference,
        )
        diffusion = config.viscosity * (
            previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
        ) / dx**2
        solution[1:-1] = previous[1:-1] - dt * upwind_advection + dt * diffusion
        solution[0] = 0.0
        solution[-1] = 0.0
        snapshots.append(solution.copy())
        time_values.append((step + 1) * dt)

    return x, np.asarray(time_values), np.asarray(snapshots)
