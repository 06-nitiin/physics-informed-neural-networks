from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_layers: Sequence[int] = (64, 64, 64)
    activation: str = "tanh"


class MLP(nn.Module):

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        activations: dict[str, type[nn.Module]] = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if config.activation not in activations:
            raise ValueError(f"Unknown activation: {config.activation}")
        layers: list[nn.Module] = []
        dimensions = [config.input_dim, *config.hidden_layers, config.output_dim]
        for left, right in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(left, right), activations[config.activation]()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        return self.network(coordinates)


def derivative(output: Tensor, inputs: Tensor, component: int = 0) -> Tensor:
    selected = output[:, component : component + 1]
    return torch.autograd.grad(
        selected,
        inputs,
        grad_outputs=torch.ones_like(selected),
        create_graph=True,
        retain_graph=True,
    )[0]


def second_derivative(output: Tensor, inputs: Tensor, component: int, coordinate: int) -> Tensor:
    first = derivative(output, inputs, component)[:, coordinate : coordinate + 1]
    return torch.autograd.grad(
        first,
        inputs,
        grad_outputs=torch.ones_like(first),
        create_graph=True,
        retain_graph=True,
    )[0][:, coordinate : coordinate + 1]


@dataclass
class TrainingHistory:
    total: list[float]
    physics: list[float]
    condition: list[float]


class PINNTrainer:

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def fit(
        self,
        collocation: Tensor,
        residual_fn: Callable[[nn.Module, Tensor], Tensor],
        condition_fn: Callable[[nn.Module], Tensor],
        epochs: int,
        print_every: int = 500,
    ) -> TrainingHistory:
        history = TrainingHistory([], [], [])
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad(set_to_none=True)
            residual = residual_fn(self.model, collocation)
            condition = condition_fn(self.model)
            physics_loss = torch.mean(residual.square())
            condition_loss = torch.mean(condition.square())
            total_loss = physics_loss + condition_loss
            total_loss.backward()
            self.optimizer.step()
            history.total.append(float(total_loss.detach()))
            history.physics.append(float(physics_loss.detach()))
            history.condition.append(float(condition_loss.detach()))
            if print_every and (epoch == 1 or epoch % print_every == 0):
                print(f"epoch={epoch:>6} total={total_loss.item():.3e} "
                      f"physics={physics_loss.item():.3e} condition={condition_loss.item():.3e}")
        return history
