from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
