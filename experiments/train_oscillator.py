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
    OscillatorConfig,
    oscillator_initial_condition,
    oscillator_reference,
    oscillator_residual,
)
from visualization.plots import save_oscillator_results


@dataclass(frozen=True)
class OscillatorTrainingConfig:

    epochs: int = 2_000
    learning_rate: float = 1e-3
    collocation_points: int = 128
    hidden_layers: tuple[int, ...] = (48, 48)
    seed: int = 7
    print_every: int = 250


@dataclass
class OscillatorRun:

    time: np.ndarray
    pinn_state: np.ndarray
    reference_state: np.ndarray
    rmse: float
    history: TrainingHistory


def train_oscillator(config: OscillatorTrainingConfig) -> OscillatorRun:

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    physical_config = OscillatorConfig()
    model = MLP(MLPConfig(input_dim=1, output_dim=2, hidden_layers=config.hidden_layers))
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
        residual_fn=lambda network, points: oscillator_residual(network, points, physical_config),
        condition_fn=lambda network: oscillator_initial_condition(network, physical_config),
        epochs=config.epochs,
        print_every=config.print_every,
    )

    time = np.linspace(0.0, physical_config.final_time, 300)
    model_inputs = torch.tensor(time, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        pinn_state = model(model_inputs).numpy()

    reference_state = oscillator_reference(time, physical_config)
    rmse = float(np.sqrt(np.mean((pinn_state - reference_state) ** 2)))

    return OscillatorRun(
        time=time,
        pinn_state=pinn_state,
        reference_state=reference_state,
        rmse=rmse,
        history=history,
    )


def main() -> None:
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    arguments = parser.parse_args()

    config = OscillatorTrainingConfig(epochs=arguments.epochs)
    run = train_oscillator(config)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = arguments.output_dir / "oscillator_results.png"
    save_oscillator_results(
        time=run.time,
        pinn_state=run.pinn_state,
        reference_state=run.reference_state,
        total_loss=run.history.total,
        output_path=plot_path,
    )

    metrics = {
        "training_config": asdict(config),
        "rmse": run.rmse,
        "final_loss": run.history.total[-1],
    }
    metrics_path = arguments.output_dir / "oscillator_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Oscillator PINN RMSE: {run.rmse:.6f}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
