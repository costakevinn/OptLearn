# OptLearn: Numerical Optimization Framework in Python

OptLearn is a Python framework for **numerical optimization** and **gradient-based learning**, designed to solve **continuous optimization problems** commonly arising in **Machine Learning**, **Neural Networks**, and **Scientific Computing**.

It provides a full workflow for **first-order optimization**, **convergence analysis**, and **model parameter estimation**, with reproducible outputs and visual diagnostics.

---

## Optimization Model

OptLearn minimizes objective functions of the form:

[
\min_{\theta \in \mathbb{R}^n} f(\theta)
]

using **numerical gradients** computed via central finite differences:

[
\frac{\partial f}{\partial \theta_i}
\approx
\frac{f(\theta_i + \varepsilon) - f(\theta_i - \varepsilon)}{2\varepsilon}
]

This formulation allows optimization of **arbitrary black-box objectives**, including loss functions that do not admit closed-form gradients.

---

## Supported Optimizers

OptLearn implements several widely used **gradient-based optimizers** under a unified API:

* **Stochastic Gradient Descent (SGD)**
* **Momentum**
* **RMSProp**
* **Adam**

Each optimizer tracks:

* Objective value ( f(\theta) )
* Gradient norm ( |\nabla f(\theta)| )
* Parameter trajectory across iterations

Convergence behavior is automatically logged and visualized.

---

## Optimization Benchmarks

The framework includes standard benchmark functions used to evaluate optimizer performance:

* **Quadratic Function** (convex)
* **Rosenbrock Function** (non-convex, ill-conditioned)
* **Himmelblau Function** (multi-modal)

These examples highlight different optimization challenges such as curvature, local minima, and stability.

**Convergence example:**

| Convergence                          |
| ------------------------------------ |
| ![](plots/Quadratic_convergence.png) |

---

## Example: Neural Network Optimization

Neural network training is treated as a **pure optimization problem**.

### Model

[
y = a \sin(x) + b x + \epsilon
]

A fully connected neural network (R → R) is trained by minimizing a **weighted least-squares loss** with L2 regularization:

[
\chi^2(\theta) =
\sum_i \left( \frac{y_i - \hat y_i(\theta)}{\sigma_i} \right)^2

* \lambda |\theta|_2^2
  ]

This example demonstrates the application of general-purpose optimizers to **nonlinear function approximation** under noisy observations.

### Results

| Convergence                   | Trained Model          |
| ----------------------------- | ---------------------- |
| ![](plots/NN_convergence.png) | ![](plots/nn_plot.png) |

* Left: optimizer convergence (loss and gradient norm)
* Right: learned function compared to noisy data
* Illustrates stability, smoothness, and generalization

---

## Features

* Modular optimizer architecture
* Numerical gradient computation
* Convergence tracking and visualization
* L2 regularization support
* Reproducible experiments
* Automatic saving of:

  * Optimization histories (`.txt`)
  * Convergence plots (`.png`)
  * Model-fit visualizations (`.png`)

---

## Usage

```bash
python3 main.py
```

Outputs:

* `results/` — optimization logs
* `plots/` — convergence and model-fit figures

---

## Project Structure

```
OptLearn/
├── optimizers/      # SGD, Momentum, RMSProp, Adam
├── examples/        # Benchmark and neural network examples
├── utils.py         # Gradients, plotting, logging
├── optimize.py      # Core optimization loop
├── main.py          # Execute experiments
├── results/         # Optimization histories
└── plots/           # Convergence and fit plots
```

---

## License

MIT License — see `LICENSE` for details.
