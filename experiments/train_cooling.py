"""Train and evaluate a physics-informed neural network for Newton's cooling."""
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
from problems.odes import (
    CoolingConfig,
    cooling_exact,
    cooling_initial_condition,
    cooling_residual,
    cooling_to_temperature,
)
from visualization.plots import save_cooling_results


@dataclass(frozen=True)
class CoolingTrainingConfig:
    """Training choices kept separate from the physical cooling parameters."""

    epochs: int = 1_500
    learning_rate: float = 1e-3
    collocation_points: int = 96
    hidden_layers: tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    seed: int = 7
    print_every: int = 250


@dataclass
class CoolingRun:
    """Outputs needed to evaluate and visualise one reproducible training run."""

    time: np.ndarray
    pinn_temperature: np.ndarray
    exact_temperature: np.ndarray
    rmse: float
    history: TrainingHistory


def train_cooling(config: CoolingTrainingConfig) -> CoolingRun:
    """Train a PINN from the cooling physics residual and initial condition."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    physical_config = CoolingConfig()
    model = MLP(
        MLPConfig(
            input_dim=1,
            output_dim=1,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = PINNTrainer(model, optimizer)

    collocation = torch.linspace(
        0.0,
        physical_config.final_time,
        config.collocation_points,
    ).reshape(-1, 1)
    collocation.requires_grad_(True)

    history = trainer.fit(
        collocation=collocation,
        residual_fn=lambda network, points: cooling_residual(network, points, physical_config),
        condition_fn=cooling_initial_condition,
        epochs=config.epochs,
        print_every=config.print_every,
    )

    time = np.linspace(0.0, physical_config.final_time, 200)
    model_inputs = torch.tensor(time, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        normalized_prediction = model(model_inputs).numpy().ravel()

    pinn_temperature = cooling_to_temperature(normalized_prediction, physical_config)
    exact_temperature = cooling_exact(time, physical_config)
    rmse = float(np.sqrt(np.mean((pinn_temperature - exact_temperature) ** 2)))

    return CoolingRun(
        time=time,
        pinn_temperature=pinn_temperature,
        exact_temperature=exact_temperature,
        rmse=rmse,
        history=history,
    )


def main() -> None:
    """Run the experiment from the command line and save reproducible outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1_500)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    arguments = parser.parse_args()

    config = CoolingTrainingConfig(epochs=arguments.epochs)
    run = train_cooling(config)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    save_cooling_results(
        time=run.time,
        pinn_temperature=run.pinn_temperature,
        exact_temperature=run.exact_temperature,
        total_loss=run.history.total,
        output_path=arguments.output_dir / "cooling_results.png",
    )

    metrics = {
        "training_config": asdict(config),
        "rmse": run.rmse,
        "final_loss": run.history.total[-1],
    }
    metrics_path = arguments.output_dir / "cooling_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Cooling PINN RMSE: {run.rmse:.6f}")
    print(f"Saved plot: {arguments.output_dir / 'cooling_results.png'}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
