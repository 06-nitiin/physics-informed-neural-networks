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


def save_oscillator_results(
    time: np.ndarray,
    pinn_state: np.ndarray,
    reference_state: np.ndarray,
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

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = ("Position x(t)", "Velocity v(t)")

    for index, label in enumerate(labels):
        axes[index].plot(time, reference_state[:, index], label="Numerical reference", linewidth=2.2)
        axes[index].plot(time, pinn_state[:, index], "--", label="PINN prediction", linewidth=2.2)
        axes[index].set_title(label)
        axes[index].set_xlabel("Time")
        axes[index].legend()

    axes[2].semilogy(total_loss, color="tab:red", linewidth=2)
    axes[2].set_title("PINN training loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Total loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_burgers_results(
    x: np.ndarray,
    time: np.ndarray,
    pinn_solution: np.ndarray,
    reference_solution: np.ndarray,
    output_path: str | Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.family": "DejaVu Sans",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    extent = [x.min(), x.max(), time.min(), time.max()]
    error = pinn_solution - reference_solution
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    panels = (
        (pinn_solution, "PINN solution", "coolwarm"),
        (reference_solution, "Finite-difference reference", "coolwarm"),
        (error, "PINN error", "seismic"),
    )
    for axis, (values, title, color_map) in zip(axes, panels):
        image = axis.imshow(
            values,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=color_map,
        )
        axis.set_title(title)
        axis.set_xlabel("Position x")
        axis.set_ylabel("Time t")
        figure.colorbar(image, ax=axis, shrink=0.88)

    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)