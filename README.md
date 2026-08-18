# Physics-Informed Neural Networks: From ODEs to Nonlinear PDEs

This repository is an original, from-scratch learning project on **Physics-Informed Neural Networks (PINNs)** using Python and PyTorch. It progresses from a first-order ODE to a coupled system and then to the nonlinear viscous Burgers' equation.

> A PINN represents an unknown solution with a neural network, differentiates that network with respect to physical coordinates, and penalises violations of the governing equation together with known initial or boundary conditions.

The approach follows the central idea of physics-informed learning: neural networks can encode differential-equation constraints as prior information while remaining differentiable with respect to their input coordinates [1].

## Why this project?

Many introductory machine-learning projects learn from labelled input-output pairs. Here, the primary training signal is instead the **physics residual**. The analytical or numerical reference solutions are used after training to evaluate whether the PINN worked; they are not used as the main labelled training data.

This project is designed for learning and portfolio use. It deliberately exposes the mechanics that a high-level library would otherwise hide:

```text
Coordinates → MLP → Automatic differentiation → Physics residual
        ↘ known initial / boundary conditions ↗
                         ↓
                  Total loss → Optimiser
```

## Problems solved

| Stage | Mathematical problem | PINN input → output | Reference used only for evaluation |
|---|---|---|---|
| 1 | Newton's law of cooling | `t → T(t)` | Analytical exponential solution |
| 2 | Damped harmonic oscillator | `t → [x(t), v(t)]` | SciPy `solve_ivp` numerical solution |
| 3 | Viscous Burgers' equation | `(x, t) → u(x, t)` | Finite-difference numerical solution |

### Newton's law of cooling

\[
\frac{dT}{dt}=-k(T-T_{\mathrm{ambient}}).
\]

The model learns a normalized temperature state, which improves numerical conditioning. The training loss combines the residual \(dy/dt+ky\) with the normalized initial condition \(y(0)=1\).

### Coupled damped oscillator

\[
\frac{dx}{dt}=v,
\qquad
\frac{dv}{dt}=-2\zeta v-\omega^2x.
\]

The neural network has two outputs. This demonstrates that PINNs can learn coupled variables and enforce multiple differential-equation residuals simultaneously. `scipy.integrate.solve_ivp` solves initial-value ODE systems independently for the evaluation reference [2].

### Viscous Burgers' equation

\[
u_t+u u_x=\nu u_{xx}.
\]

The Burgers PINN uses automatic differentiation to calculate \(u_t\), \(u_x\), and \(u_{xx}\). It is constrained by the initial profile \(u(x,0)=-\sin(\pi x)\) and zero values at both spatial boundaries. A finite-difference solver is used only after training to generate the reference and error heatmaps.

## Repository structure

```text
physics-informed-neural-networks/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── pinn/
│   │   └── core.py                 # MLP, derivatives, trainer
│   ├── problems/
│   │   ├── odes.py                 # Cooling and coupled oscillator
│   │   └── burgers.py              # Burgers residual, conditions, reference solver
│   └── visualization/
│       └── plots.py                # Cooling, oscillator, and PDE plots
├── experiments/
│   ├── train_cooling.py
│   ├── train_oscillator.py
│   ├── train_burgers.py
│   ├── sweep_cooling.py
│   └── deepxde_comparison.py       # Optional comparison only
├── notebooks/
│   ├── 01_pinn_fundamentals.ipynb
│   ├── 02_newtons_cooling.ipynb
│   ├── 03_coupled_odes.ipynb
│   ├── 04_burgers_equation.ipynb
│   └── 05_deepxde_comparison.ipynb
├── docs/
│   └── mathematical_notes.md
└── tests/
```

## How the manual PINN works

`src/pinn/core.py` contains the reusable implementation. `MLP` is a configurable fully connected network. The `derivative` and `second_derivative` helpers use PyTorch automatic differentiation, which computes derivatives through the computational graph [3]. Each problem module defines its own residual and condition functions; the generic `PINNTrainer` minimises:

\[
\mathcal{L}=\operatorname{mean}(r_\theta^2)+\operatorname{mean}(c_\theta^2).
\]

Here, \(r_\theta\) is the physics residual and \(c_\theta\) is the initial- or boundary-condition error. Collocation points are coordinates where the equation is enforced; they are not labelled solution values.

## Installation

```bash
git clone https://github.com/06-nitiin/physics-informed-neural-networks.git
cd physics-informed-neural-networks

python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the experiments

Start with the automated checks:

```bash
python -m pytest -q
```

Run the ODE and PDE examples from the repository root:

```bash
# Newton cooling
python experiments/train_cooling.py --epochs 1500

# Coupled oscillator
python experiments/train_oscillator.py --epochs 2000

# Burgers' equation: start with a quick check, then run the baseline
python experiments/train_burgers.py --epochs 500
python experiments/train_burgers.py --epochs 3000
```

Each script writes plots and JSON metrics to `results/`. Those generated artifacts are ignored by Git by default. If you want selected figures to appear in the GitHub repository, review them first and deliberately add only the images you want to preserve.

## Controlled experiments

The cooling experiment sweep compares network width and activation while holding the physical problem, seed, learning rate, collocation count, and epoch count fixed.

```bash
python experiments/sweep_cooling.py --epochs 500
cat results/cooling_sweep.csv
```

| Variable | Configurations in the initial sweep |
|---|---|
| Hidden-layer width | `(16, 16)`, `(32, 32)`, `(64, 64)` |
| Activation | `tanh` and `silu` with the same baseline width |
| Recorded metrics | Final loss, RMSE, wall-clock runtime, seed, and epoch count |

The CSV contains **measured local results**, not universal claims. A lower training loss does not always imply a lower reference-solution error, so both loss and RMSE should be interpreted together.

## Results and interpretation

The cooling and oscillator scripts save prediction-versus-reference plots alongside training losses. The Burgers script saves a three-panel heatmap:

1. The learned PINN solution.
2. The independently generated finite-difference reference.
3. The pointwise difference between them.

The baseline Burgers PINN is intentionally simple. It can reproduce the broad structure of the solution, but error can remain concentrated near its steep transition region. This is an expected limitation worth reporting honestly; a low average residual is not a guarantee of low error at every coordinate.

## Manual PyTorch versus DeepXDE

[DeepXDE](https://deepxde.readthedocs.io/) is a scientific-machine-learning library that supports PINN workflows and multiple tensor backends, including PyTorch [4]. The optional comparison in this repository keeps the manual implementation as the primary solution while showing what a framework abstracts.

| Criterion | Manual PyTorch implementation | Optional DeepXDE comparison |
|---|---|---|
| Code mechanics | MLP, derivatives, loss construction, and optimiser loop are explicit | Geometry, conditions, sampling, and model workflow are expressed through framework objects |
| Learning value | Best for understanding what a PINN is doing | Best for learning framework conventions and rapidly prototyping standard cases |
| Flexibility | Direct control over tensors, losses, and diagnostics | High-level abstractions plus framework-supported options |
| Use in this repository | Primary implementation | Secondary comparison only |

Install the optional dependency only when you want to run the comparison:

```bash
pip install deepxde
DDE_BACKEND=pytorch python experiments/deepxde_comparison.py --problem cooling --iterations 1000
DDE_BACKEND=pytorch python experiments/deepxde_comparison.py --problem burgers --iterations 1000
```

DeepXDE's Burgers demo uses the same core ingredients: a space-time geometry, PDE callback, condition objects, a neural network, and a model-training workflow [5].

## Educational notebooks and notes

The notebooks explain the project progression without duplicating the source implementation:

```text
01_pinn_fundamentals.ipynb
02_newtons_cooling.ipynb
03_coupled_odes.ipynb
04_burgers_equation.ipynb
05_deepxde_comparison.ipynb
```

For a concise mathematical reference and source links, see [`docs/mathematical_notes.md`](docs/mathematical_notes.md).

## Reproducibility and limitations

The scripts define random seeds and expose architecture, sampling, learning-rate, and epoch choices as configuration values. Results can still vary with operating system, hardware, package versions, numerical precision, and optimisation details.

This is a **working learning and portfolio project**, not a production fluid-dynamics solver. It uses baseline architectures and moderate training budgets so that the mechanics remain understandable. The project does not claim state-of-the-art PDE accuracy.

## Future work

Realistic extensions include adaptive collocation, loss weighting, Adam-to-L-BFGS optimisation, Fourier features, domain decomposition, inverse problems, parameter discovery, the heat and wave equations, and more difficult fluid-dynamics systems such as Navier–Stokes.

## References

[1]: https://arxiv.org/abs/1711.10561 "Raissi, Perdikaris, and Karniadakis, Physics Informed Deep Learning (Part I)"
[2]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html "SciPy documentation: scipy.integrate.solve_ivp"
[3]: https://docs.pytorch.org/docs/stable/autograd.html "PyTorch documentation: torch.autograd"
[4]: https://deepxde.readthedocs.io/en/latest/ "DeepXDE documentation"
[5]: https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.html "DeepXDE Burgers equation demo"
