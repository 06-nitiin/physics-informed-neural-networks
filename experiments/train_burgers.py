from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pinn.core import MLP, MLPConfig, PINNTrainer, TrainingHistory
from problems.burgers import (
    BurgersConfig,
    burgers_conditions,
    burgers_reference,
    burgers_residual,
)
from visualization.plots import save_burgers_results


@dataclass(frozen=True)
class BurgersTrainingConfig:

    epochs: int = 3_000
    learning_rate: float = 2e-3
    collocation_x: int = 48
    collocation_t: int = 48
    initial_condition_points: int = 64
    boundary_condition_points: int = 64
    hidden_layers: tuple[int, ...] = (48, 48, 48)
    reference_nx: int = 81
    reference_nt: int = 121
    seed: int = 7
    print_every: int = 500


@dataclass
class BurgersRun:

    x: np.ndarray
    time: np.ndarray
    pinn_solution: np.ndarray
    reference_solution: np.ndarray
    rmse: float
    history: TrainingHistory


def train_burgers(config: BurgersTrainingConfig) -> BurgersRun:
    """Train the PINN from the PDE residual and initial/boundary conditions."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    physical_config = BurgersConfig()
    model = MLP(MLPConfig(input_dim=2, output_dim=1, hidden_layers=config.hidden_layers))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = PINNTrainer(model, optimizer)

    x_collocation = torch.linspace(
        physical_config.x_min,
        physical_config.x_max,
        config.collocation_x,
    )
    t_collocation = torch.linspace(0.0, physical_config.final_time, config.collocation_t)
    x_grid, t_grid = torch.meshgrid(x_collocation, t_collocation, indexing="ij")
    collocation = torch.stack((x_grid.ravel(), t_grid.ravel()), dim=1)
    collocation.requires_grad_(True)

    history = trainer.fit(
        collocation=collocation,
        residual_fn=lambda network, points: burgers_residual(network, points, physical_config),
        condition_fn=lambda network: burgers_conditions(
            network,
            physical_config,
            initial_points=config.initial_condition_points,
            boundary_points=config.boundary_condition_points,
        ),
        epochs=config.epochs,
        print_every=config.print_every,
    )

    x_reference, time_reference, reference_solution = burgers_reference(
        physical_config,
        nx=config.reference_nx,
        nt=config.reference_nt,
    )
    x_evaluation, t_evaluation = np.meshgrid(
        x_reference,
        time_reference,
        indexing="ij",
    )
    evaluation_coordinates = np.column_stack(
        (x_evaluation.ravel(), t_evaluation.ravel())
    )
    with torch.no_grad():
        pinn_solution = model(
            torch.tensor(evaluation_coordinates, dtype=torch.float32)
        ).numpy().reshape(len(x_reference), len(time_reference))

    reference_by_x_then_time = reference_solution.T
    rmse = float(
        np.sqrt(np.mean((pinn_solution - reference_by_x_then_time) ** 2))
    )

    return BurgersRun(
        x=x_reference,
        time=time_reference,
        pinn_solution=pinn_solution.T,
        reference_solution=reference_solution,
        rmse=rmse,
        history=history,
    )


def main() -> None:
    """Run the Burgers experiment from the command line and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    arguments = parser.parse_args()

    config = BurgersTrainingConfig(epochs=arguments.epochs)
    run = train_burgers(config)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = arguments.output_dir / "burgers_results.png"
    save_burgers_results(
        x=run.x,
        time=run.time,
        pinn_solution=run.pinn_solution,
        reference_solution=run.reference_solution,
        output_path=plot_path,
    )

    metrics = {
        "training_config": asdict(config),
        "rmse": run.rmse,
        "final_loss": run.history.total[-1],
    }
    metrics_path = arguments.output_dir / "burgers_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Burgers PINN RMSE: {run.rmse:.6f}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
