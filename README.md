# OptLearn — Numerical Optimization Framework in Python

OptLearn is a **numerical optimization framework** implemented in Python, designed for solving **continuous optimization problems** using **gradient-based methods**.  
The project provides a unified optimization interface and multiple optimizers commonly used in **machine learning**, **statistical inference**, and **scientific computing**.

Although neural network training is included as an example, the core of OptLearn is a **general-purpose optimizer**, applicable to arbitrary objective functions.

---

## Core Idea

Given an objective function:

\[
\min_{\mathbf{x}} f(\mathbf{x})
\]

OptLearn performs iterative optimization using **numerical gradients** and modular **update rules**, allowing different optimizers to be swapped without changing the problem definition.

---

## Supported Optimizers

### Stochastic Gradient Descent (SGD)

\[
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
\]

### Momentum

\[
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta)\nabla f(\mathbf{x}_t)
\]
\[
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_t
\]

### RMSProp

\[
\mathbf{s}_t = \beta \mathbf{s}_{t-1} + (1 - \beta)(\nabla f(\mathbf{x}_t))^2
\]
\[
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \nabla f(\mathbf{x}_t)
\]

### Adam

\[
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\nabla f(\mathbf{x}_t)
\]
\[
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)(\nabla f(\mathbf{x}_t))^2
\]

---

## Neural Network Optimization Example

The framework includes a neural network regression example to demonstrate scalability to high-dimensional parameter spaces.

### Convergence

![Neural Network Convergence](plots/NN_convergence.png)

### Trained Network Fit

![Neural Network Fit](plots/nn_plot.png)

---

## Project Structure

```
OptLearn/
├── examples/
├── optimizers/
├── plots/
├── results/
├── utils.py
├── optimize.py
├── main.py
└── README.md
```

---

## How to Run

```bash
python main.py
```

---

## License

MIT License
