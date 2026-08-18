# Mathematical notes for the PINN project

## 1. What turns a neural network into a PINN?

A standard neural network learns a mapping from inputs to outputs. In this project, the network represents an unknown physical solution, such as temperature \(T(t)\), oscillator state \([x(t),v(t)]\), or field \(u(x,t)\). A Physics-Informed Neural Network (PINN) becomes different from ordinary supervised fitting when derivatives of the network output are substituted into a governing differential equation and violations of that equation are included in the loss. PINNs are designed to encode physical laws as prior information while remaining differentiable with respect to input coordinates [1].

PyTorch provides automatic differentiation through `torch.autograd`, which allows the project to calculate derivatives of the network output with respect to time or position-and-time coordinates [2]. The reusable helpers in `src/pinn/core.py` expose that mechanism explicitly rather than hiding it behind a high-level PINN framework.

## 2. Residual and condition loss

Suppose a network \(u_\theta\) approximates a solution to a differential equation \(F=0\). The physics residual is:

\[
r_\theta=F\left(u_\theta,\frac{\partial u_\theta}{\partial t},\frac{\partial u_\theta}{\partial x},\ldots\right).
\]

The training objective in this repository is the mean squared residual combined with errors at known initial or boundary conditions:

\[
\mathcal{L}=\operatorname{mean}(r_\theta^2)+\operatorname{mean}(c_\theta^2).
\]

The collocation points are coordinates where the residual is evaluated. They are not labelled solution values in the main physics-informed training loop. Initial and boundary conditions are necessary because a low PDE residual alone can permit many mathematically valid but physically irrelevant functions.

## 3. Newton's law of cooling

The first example solves:

\[
\frac{dT}{dt}=-k(T-T_{\mathrm{ambient}}).
\]

To improve conditioning, the model predicts the normalized variable:

\[
y=\frac{T-T_{\mathrm{ambient}}}{T_0-T_{\mathrm{ambient}}}.
\]

The normalized residual becomes \(dy/dt+ky\), and the initial condition is \(y(0)=1\). The analytical exponential curve is calculated separately by `cooling_exact` only to evaluate the trained PINN with RMSE. It is not supplied as labelled data to the primary training loop.

## 4. Coupled damped oscillator

The oscillator network has one input and two outputs:

\[
t\longmapsto[x(t),v(t)].
\]

It enforces two coupled residuals:

\[
r_x=\frac{dx}{dt}-v,
\]

\[
r_v=\frac{dv}{dt}+2\zeta v+\omega^2x.
\]

The project uses `scipy.integrate.solve_ivp` as an independent evaluation reference. SciPy defines `solve_ivp` as a solver for an initial-value problem of the form \(dy/dt=f(t,y)\), with a specified initial state [3]. This reference is not part of PINN training.

## 5. Viscous Burgers' equation

The most difficult example solves:

\[
u_t+u u_x=\nu u_{xx}.
\]

The network input is \((x,t)\), while the output is \(u(x,t)\). Automatic differentiation calculates \(u_t\), \(u_x\), and \(u_{xx}\). The loss combines the PDE residual with the initial profile \(u(x,0)=-\sin(\pi x)\) and zero boundary values at \(x=-1\) and \(x=1\).

A finite-difference solver provides a reference heatmap and RMSE after training. The baseline PINN can learn the broad solution but may retain concentrated error near the steep transition region. This is a limitation to report honestly rather than hide; a small average loss does not guarantee uniform pointwise accuracy.

## 6. How to interpret the controlled experiments

The cooling sweep fixes the physical problem, seed, collocation count, learning rate, and epochs while changing width or activation. The output CSV reports final loss, RMSE, and runtime. The results are evidence for the specified configuration and hardware only. They should not be presented as a general ranking of all architectures or activations.

A useful habit is to distinguish **training fit** from **solution accuracy**. Final loss measures the optimisation objective, while RMSE measures disagreement with an independently computed reference. Both should be reported.

## 7. Reproducibility and limitations

The experiment scripts set NumPy and PyTorch seeds, record configuration values, and generate outputs from code. Even with a fixed seed, results can vary with hardware, library versions, precision, and optimisation details. This repository is a working learning and portfolio project, not a production fluid-dynamics solver. Natural next steps include loss weighting, adaptive collocation, L-BFGS refinement, Fourier features, domain decomposition, inverse problems, and more challenging PDEs.

## References

[1]: https://arxiv.org/abs/1711.10561 "Raissi, Perdikaris, and Karniadakis, Physics Informed Deep Learning (Part I)"
[2]: https://docs.pytorch.org/docs/stable/autograd.html "PyTorch documentation: torch.autograd"
[3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html "SciPy documentation: scipy.integrate.solve_ivp"
