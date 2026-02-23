# 🚀 OptLearn — Numerical Optimization Framework in Python

OptLearn is a Python framework for numerical optimization and gradient-based learning, designed to solve continuous optimization problems commonly found in Machine Learning, Neural Networks, and Scientific Computing.

The framework provides a structured workflow for first-order optimization, convergence analysis, and parameter estimation, with reproducible experiments and visual diagnostics.

**Author:** Kevin Mota da Costa

**Portfolio:** [https://costakevinn.github.io](https://costakevinn.github.io)

**LinkedIn:** [https://linkedin.com/in/costakevinnn](https://linkedin.com/in/costakevinnn)

---

## 🎯 Project Purpose

OptLearn was developed to explore optimization as the core engine behind machine learning systems.

Instead of relying on automatic differentiation frameworks, this project:

* Implements numerical gradient estimation
* Benchmarks multiple first-order optimizers
* Analyzes convergence behavior
* Studies stability under difficult objective landscapes

This reflects a systems-level understanding of how optimization drives learning.

---

## 🧠 Optimization Formulation

The framework minimizes objective functions of the form:

minimize f(θ), where θ ∈ Rⁿ

Gradients are computed numerically using central finite differences:

∂f/∂θᵢ ≈ [ f(θᵢ + ε) − f(θᵢ − ε) ] / (2ε)

This enables optimization of arbitrary black-box objectives, including functions without closed-form gradients.

---

## ⚙ Supported Optimizers

OptLearn implements several widely used gradient-based optimizers under a unified API:

* Stochastic Gradient Descent (SGD)
* Momentum
* RMSProp
* Adam

Each optimizer tracks:

* Objective value f(θ)
* Gradient norm ||∇f(θ)||
* Parameter trajectory across iterations

Convergence behavior is automatically logged and visualized.

---

## 📊 Optimization Benchmarks

The framework includes classical benchmark functions:

* Quadratic function (convex)
* Rosenbrock function (non-convex, ill-conditioned)
* Himmelblau function (multi-modal)

These functions highlight challenges such as curvature sensitivity, local minima, and convergence stability.

### Example: Convergence Behavior

| Convergence                          |
| ------------------------------------ |
| ![](plots/Quadratic_convergence.png) |

---

## 🧪 Neural Network Optimization Example

Neural network training is treated explicitly as an optimization problem.

### Model

y = a sin(x) + b x + noise

A fully connected neural network (R → R) is trained by minimizing a weighted least-squares loss with L2 regularization.

Loss function:

χ²(θ) = Σ [ (yᵢ − ŷᵢ(θ)) / σᵢ ]² + λ ||θ||²

This example demonstrates how general-purpose optimizers behave when applied to nonlinear function approximation under noisy observations.

---

### Results

| Convergence                   | Trained Model          |
| ----------------------------- | ---------------------- |
| ![](plots/NN_convergence.png) | ![](plots/nn_plot.png) |

* Left: optimizer convergence (loss and gradient norm)
* Right: learned function vs noisy data
* Illustrates stability and generalization behavior

---

## 🔬 Capabilities Demonstrated

* Numerical gradient computation
* First-order optimization mechanics
* Convergence diagnostics
* Regularization handling
* Non-convex optimization behavior
* Reproducible experimentation

---

## 🛠 Features

* Modular optimizer architecture
* Central finite difference gradients
* Convergence tracking
* L2 regularization support
* Automatic output generation:

  * Optimization histories (.txt)
  * Convergence plots (.png)
  * Model-fit visualizations (.png)

---

## ▶ Usage

```bash
python3 main.py
```

Outputs:

* `results/` → optimization logs
* `plots/` → convergence and model-fit figures

---

## 📁 Project Structure

```
OptLearn/
├── optimizers/      # SGD, Momentum, RMSProp, Adam
├── examples/        # Benchmark and neural network cases
├── utils.py         # Gradients, plotting, logging
├── optimize.py      # Core optimization loop
├── main.py          # Execute experiments
├── results/         # Optimization histories
└── plots/           # Convergence and fit plots
```

---

## 🛠 Tech Stack

### Programming

Python

### Scientific Computing

* NumPy

### Optimization

* SGD
* Momentum
* RMSProp
* Adam
* Finite difference gradients

### Visualization

* Matplotlib

---

## 🌐 Portfolio

This project is part of my Machine Learning portfolio:
👉 [https://costakevinn.github.io](https://costakevinn.github.io)

---

## License

MIT License — see `LICENSE` for details.
