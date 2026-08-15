# Physics-Informed Neural Networks: From ODEs to Nonlinear PDEs

This repository is an original, from-scratch exploration of **physics-informed neural networks (PINNs)** using Python and PyTorch. Rather than fitting a neural network to a table of labelled solutions, the model learns a differentiable function whose derivatives satisfy a governing differential equation while also satisfying initial and boundary conditions.

> **Central idea:** coordinates enter a neural network, automatic differentiation computes derivatives of its prediction, and the differential-equation residual becomes part of the optimisation loss.

## What is included

| Stage | Problem | Main learning objective | Reference |
|---|---|---|---|
| 1 | PINN fundamentals | Build an MLP, derivatives, residuals, conditions, and a transparent trainer | Analytical solution |
| 2 | Newton's law of cooling | Learn a first-order ODE without solution labels | Closed form |
| 3 | Coupled oscillator | Extend the framework to multiple outputs and residuals | `scipy.integrate.solve_ivp` |
| 4 | Viscous Burgers' equation | Compute first and second derivatives for a nonlinear PDE | Explicit finite differences |
| 5 | Experiments | Measure the effect of architecture and training choices | Saved metrics and plots |
| 6 | DeepXDE comparison | Optional framework implementation after understanding the mechanics | DeepXDE |

## How a PINN works

For Newton cooling, the network predicts \(T_\theta(t)\). Automatic differentiation gives \(dT_\theta/dt\), so the physics residual is

\[
r_\theta(t)=\frac{dT_\theta}{dt}+k(T_\theta-T_\mathrm{environment}).
\]

The training objective combines the residual with the initial-condition error:

\[
\mathcal{L}=\operatorname{mean}(r_\theta^2)+\operatorname{mean}((T_\theta(0)-T_0)^2).
\]

For Burgers' equation, the network predicts \(u_\theta(x,t)\) and the residual is \(u_t+u u_x-\nu u_{xx}\). The implementation in `src/pinn/core.py` deliberately exposes these derivatives instead of hiding them behind a high-level library.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The base project does not require DeepXDE. To run the optional comparison, install it separately with `pip install deepxde` and use the documented comparison notebook.

## Running the project

From the repository root:

```bash
PYTHONPATH=src python experiments/run_all.py --epochs 1500
```

The command writes `results/cooling.png`, `results/burgers.png`, and `results/metrics.json`. Reduce `--epochs` for a quick smoke test. The Burgers' experiment is intentionally more expensive because it differentiates twice through a two-dimensional network.

The notebooks in `notebooks/` are designed as guided explanations. They call the same reusable modules as the experiment script, so the notebooks are not a separate, unverified implementation.

## Repository map

| Path | Purpose |
|---|---|
| `src/pinn/core.py` | MLP, first/second derivatives, training history, and transparent trainer |
| `src/problems/odes.py` | Cooling and coupled oscillator equations and references |
| `src/problems/burgers.py` | Burgers' residual, conditions, and finite-difference reference |
| `src/visualization/plots.py` | Consistent comparison and loss plots |
| `experiments/run_all.py` | Reproducible end-to-end demonstrations |
| `experiments/` | Controlled configuration studies and optional framework comparison |
| `notebooks/` | Educational, step-by-step walkthroughs |
| `docs/` | Mathematical notes and interpretation guidance |
| `results/` | Generated artifacts; large generated files are ignored by Git |

## Experimental discipline

The repository is structured so results are generated rather than claimed in advance. Every experiment should record its seed, model configuration, epoch count, final loss, error metric, and wall-clock time. When comparing settings, change one factor at a time and keep the collocation points and evaluation grid fixed. The included smoke-test outputs are examples of the workflow, not universal benchmarks; hardware and library versions affect timing and optimisation.

Useful follow-up studies include width/depth, `tanh` versus `silu`, collocation density, learning rate, Adam versus L-BFGS, and separate weights for physics and condition losses. A good report should explain not only which configuration wins, but why sampling, smooth activations, or loss balance might affect optimisation.

## Quality checks

```bash
python -m compileall src experiments
python -m pytest -q
```

## Future work

Natural extensions include adaptive collocation, the heat and wave equations, inverse problems that discover unknown coefficients, Navier–Stokes equations, and hybrid Adam/L-BFGS optimisation. These are useful only after checking that the simpler residuals and reference comparisons are correct.

## Suggested GitHub workflow

This repository is intentionally left local. After reviewing the code, create your own repository and commit it in logical increments, for example: scaffolding; core PINN engine; ODE examples; Burgers' equation; experiments and figures; and final documentation. Do not commit `.venv/` or generated result files unless you deliberately want selected figures in the repository.
