from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_cooling_results(
    time: np.ndarray,
    pinn_temperature: np.ndarray,
    exact_temperature: np.ndarray,
    total_loss: Sequence[float],
    output_path: str | Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(time, exact_temperature, label="Analytical reference", linewidth=2.4)
    axes[0].plot(time, pinn_temperature, "--", label="PINN prediction", linewidth=2.4)
    axes[0].set_title("Newton's law of cooling")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Temperature (degrees C)")
    axes[0].legend()

    axes[1].semilogy(total_loss, color="tab:red", linewidth=2)
    axes[1].set_title("PINN training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Total loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
