from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.train_cooling import CoolingTrainingConfig, train_cooling


@dataclass(frozen=True)
class CoolingExperimentCase:

    name: str
    hidden_layers: tuple[int, ...]
    activation: str


DEFAULT_CASES = (
    CoolingExperimentCase("narrow_tanh", (16, 16), "tanh"),
    CoolingExperimentCase("baseline_tanh", (32, 32), "tanh"),
    CoolingExperimentCase("wide_tanh", (64, 64), "tanh"),
    CoolingExperimentCase("baseline_silu", (32, 32), "silu"),
)


def run_case(case: CoolingExperimentCase, epochs: int) -> dict[str, Any]:
    training_config = CoolingTrainingConfig(
        epochs=epochs,
        hidden_layers=case.hidden_layers,
        activation=case.activation,
        seed=7,
        print_every=0,
    )

    start_time = time.perf_counter()
    run = train_cooling(training_config)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "case": case.name,
        "hidden_layers": str(case.hidden_layers),
        "activation": case.activation,
        "epochs": epochs,
        "final_loss": run.history.total[-1],
        "rmse": run.rmse,
        "seconds": elapsed_seconds,
    }


def run_sweep(epochs: int, cases: tuple[CoolingExperimentCase, ...] = DEFAULT_CASES) -> list[dict[str, Any]]:
    # A short warm-up avoids attributing one-time framework initialisation to
    # the first architecture's measured runtime.
    train_cooling(
        CoolingTrainingConfig(
            epochs=1,
            collocation_points=8,
            hidden_layers=(8, 8),
            print_every=0,
        )
    )
    return [run_case(case, epochs) for case in cases]


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "cooling_sweep.csv",
    )
    arguments = parser.parse_args()

    rows = run_sweep(arguments.epochs)
    write_results(rows, arguments.output)

    print(f"Saved {len(rows)} measured experiment rows to: {arguments.output}")
    for row in rows:
        print(
            f"{row['case']}: RMSE={row['rmse']:.6f}, "
            f"loss={row['final_loss']:.3e}, seconds={row['seconds']:.3f}"
        )


if __name__ == "__main__":
    main()
