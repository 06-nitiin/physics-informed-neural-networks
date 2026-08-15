from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pinn.core import derivative, second_derivative


def test_derivative_of_cubic() -> None:
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = x**3

    actual = derivative(y, x)
    expected = 3 * x**2

    assert torch.allclose(actual, expected)


def test_second_derivative_of_cubic() -> None:
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = x**3

    actual = second_derivative(y, x, component=0, coordinate=0)
    expected = 6 * x

    assert torch.allclose(actual, expected)
