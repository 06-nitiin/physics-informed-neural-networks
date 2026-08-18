from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problems.burgers import BurgersConfig, burgers_reference
from problems.odes import CoolingConfig, cooling_exact, cooling_to_temperature
from visualization.plots import save_burgers_results, save_cooling_results


def load_deepxde() -> Any:
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    try:
        import deepxde as dde
    except ImportError as error:
        raise SystemExit(
            "DeepXDE is optional. Activate .venv, then run `pip install deepxde` "
            "before using this comparison script."
        ) from error
    return dde


def run_cooling_deepxde(iterations: int, output_dir: Path) -> dict[str, float]:
    dde = load_deepxde()
    dde.config.set_random_seed(7)
    physical = CoolingConfig()
    geometry = dde.geometry.TimeDomain(0.0, physical.final_time)

    def pde(time, normalized_temperature):
        derivative_time = dde.grad.jacobian(normalized_temperature, time, i=0, j=0)
        return derivative_time + physical.cooling_rate * normalized_temperature

    initial_condition = dde.icbc.DirichletBC(
        geometry,
        lambda coordinates: np.ones((len(coordinates), 1)),
        lambda coordinates, on_boundary: on_boundary and np.isclose(coordinates[0], 0.0),
    )
    data = dde.data.PDE(
        geometry,
        pde,
        [initial_condition],
        num_domain=96,
        num_boundary=2,
        num_test=200,
    )
    network = dde.nn.FNN([1, 32, 32, 1], "tanh", "Glorot normal")
    model = dde.Model(data, network)
    model.compile("adam", lr=1e-3)
    loss_history, _ = model.train(iterations=iterations, display_every=max(1, iterations // 4))

    time = np.linspace(0.0, physical.final_time, 200).reshape(-1, 1)
    normalized_prediction = model.predict(time).ravel()
    pinn_temperature = cooling_to_temperature(normalized_prediction, physical)
    exact_temperature = cooling_exact(time.ravel(), physical)
    rmse = float(np.sqrt(np.mean((pinn_temperature - exact_temperature) ** 2)))

    save_cooling_results(
        time=time.ravel(),
        pinn_temperature=pinn_temperature,
        exact_temperature=exact_temperature,
        total_loss=[float(np.sum(value)) for value in loss_history.loss_train],
        output_path=output_dir / "deepxde_cooling_results.png",
    )
    return {"problem": "cooling", "iterations": iterations, "rmse": rmse}


def run_burgers_deepxde(iterations: int, output_dir: Path) -> dict[str, float]:
    dde = load_deepxde()
    dde.config.set_random_seed(7)
    physical = BurgersConfig()
    space = dde.geometry.Interval(physical.x_min, physical.x_max)
    time_domain = dde.geometry.TimeDomain(0.0, physical.final_time)
    space_time = dde.geometry.GeometryXTime(space, time_domain)

    def pde(coordinates, solution):
        solution_x = dde.grad.jacobian(solution, coordinates, i=0, j=0)
        solution_t = dde.grad.jacobian(solution, coordinates, i=0, j=1)
        solution_xx = dde.grad.hessian(solution, coordinates, i=0, j=0)
        return solution_t + solution * solution_x - physical.viscosity * solution_xx

    boundary_condition = dde.icbc.DirichletBC(
        space_time,
        lambda coordinates: 0.0,
        lambda _, on_boundary: on_boundary,
    )
    initial_condition = dde.icbc.IC(
        space_time,
        lambda coordinates: -np.sin(np.pi * coordinates[:, 0:1]),
        lambda _, on_initial: on_initial,
    )
    data = dde.data.TimePDE(
        space_time,
        pde,
        [boundary_condition, initial_condition],
        num_domain=2304,
        num_boundary=128,
        num_initial=128,
        num_test=1024,
    )
    network = dde.nn.FNN([2, 48, 48, 48, 1], "tanh", "Glorot normal")
    model = dde.Model(data, network)
    model.compile("adam", lr=2e-3)
    model.train(iterations=iterations, display_every=max(1, iterations // 4))

    x, time, reference = burgers_reference(physical, nx=81, nt=121)
    x_grid, time_grid = np.meshgrid(x, time, indexing="ij")
    coordinates = np.column_stack((x_grid.ravel(), time_grid.ravel()))
    pinn_by_x_then_time = model.predict(coordinates).reshape(len(x), len(time))
    pinn_solution = pinn_by_x_then_time.T
    rmse = float(np.sqrt(np.mean((pinn_solution - reference) ** 2)))

    save_burgers_results(
        x=x,
        time=time,
        pinn_solution=pinn_solution,
        reference_solution=reference,
        output_path=output_dir / "deepxde_burgers_results.png",
    )
    return {"problem": "burgers", "iterations": iterations, "rmse": rmse}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("cooling", "burgers"), default="cooling")
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    if arguments.problem == "cooling":
        metrics = run_cooling_deepxde(arguments.iterations, arguments.output_dir)
    else:
        metrics = run_burgers_deepxde(arguments.iterations, arguments.output_dir)

    metrics_path = arguments.output_dir / f"deepxde_{arguments.problem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"DeepXDE {metrics['problem']} RMSE: {metrics['rmse']:.6f}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
