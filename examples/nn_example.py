# examples/nn_example.py
import numpy as np
import matplotlib.pyplot as plt

from optimize import optimize
from utils import save_history, print_summary, plot_convergence


# ============================================================
# Ground-truth function
# ============================================================
def real_function(x, coeffs):
    # y = a*sin(x) + b*x
    a, b = coeffs
    return a * np.sin(x) + b * x


# ============================================================
# Gaussian noise (Box–Muller)
# ============================================================
def gaussian_noise(n, std=1.0):
    # Always return exactly n samples (even if n is odd)
    m = (n + 1) // 2
    u1 = np.random.rand(m)
    u2 = np.random.rand(m)

    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)

    z = np.concatenate([z0, z1])[:n]
    return z * std


# ============================================================
# Activation
# ============================================================
def activation(x):
    return np.tanh(x)


# ============================================================
# Neural network (R -> R)
# ============================================================
layers = [1, 16, 16, 1]


def unpack_params(params):
    # Flat params -> list of (W, b) per layer
    weights, biases = [], []
    idx = 0

    for i in range(len(layers) - 1):
        in_dim = layers[i]
        out_dim = layers[i + 1]

        w_size = in_dim * out_dim
        b_size = out_dim

        W = params[idx : idx + w_size].reshape(in_dim, out_dim)
        idx += w_size

        b = params[idx : idx + b_size].reshape(1, out_dim)
        idx += b_size

        weights.append(W)
        biases.append(b)

    return weights, biases


def forward_nn(x, weights, biases):
    # x: (N, 1) -> y: (N, 1)
    a = x
    for W, b in zip(weights[:-1], biases[:-1]):
        a = activation(a @ W + b)
    y = a @ weights[-1] + biases[-1]  # linear output
    return y


# ============================================================
# Training data
# ============================================================
np.random.seed(10)

n_samples = 10
coeffs = (1.0, 0.5)

x_train = np.linspace(0.0, 5.0, n_samples) + gaussian_noise(n_samples, std=0.5)
y_true_train = real_function(x_train, coeffs)

# Noisy observations
y_train = y_true_train + gaussian_noise(n_samples, std=0.2)

# Uncertainty (avoid zeros)
dy = np.abs(gaussian_noise(n_samples, std=0.5)) + 1e-3


# ============================================================
# Loss (only at train points)
# ============================================================
l2_lambda = 5e-3


def loss(params, c=None):
    # Signature includes c=None because optimize calls func(params, c)
    weights, biases = unpack_params(params)

    x_in = x_train.reshape(-1, 1)  # (N, 1)
    y_pred = forward_nn(x_in, weights, biases).reshape(-1)  # (N,)

    chi2 = float(np.sum(((y_train - y_pred) / dy) ** 2))
    l2 = float(sum(np.sum(W**2) for W in weights)) * l2_lambda

    return chi2 + l2


# ============================================================
# Runner
# ============================================================
def run_nn_example():
    print("\n==== Running Neural Network example ====")

    # Total number of parameters
    n_params = sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))

    # Train
    params_opt, history_detailed = optimize(
        loss,
        n_params=n_params,
        optimizer="adam",
        lr=0.01,
        max_iters=500,
        tol=1e-5,
        verbose=True,
    )

    weights_opt, biases_opt = unpack_params(params_opt)

    # Dense curve (NN prediction; no interpolation)
    t_plot = np.linspace(np.min(x_train), np.max(x_train), 400)
    y_plot = forward_nn(t_plot.reshape(-1, 1), weights_opt, biases_opt).reshape(-1)

    # Dense ground-truth (just for reference in the plot)
    y_true_plot = real_function(t_plot, coeffs)

    # Predictions at train points
    y_pred_train = forward_nn(x_train.reshape(-1, 1), weights_opt, biases_opt).reshape(-1)

    # Save optimization artifacts
    save_history(history_detailed, filename="results/NN_history.txt")
    print_summary(history_detailed, last_n=5)
    plot_convergence(history_detailed, filename="plots/NN_convergence.png", title="Neural Network Convergence")

    plt.figure(figsize=(8, 4.5))

    # Ground-truth curve (reference)
    plt.plot(t_plot, y_true_plot, "--", lw=2, label="Real Function")

    # NN fit (continuous)
    plt.plot(t_plot, y_plot, "-",c='blue', lw=2, label="NN Fit")

    # Observations with error bars
    plt.errorbar(
        x_train,
        y_train,
        yerr=dy,
        color = 'red',
        fmt=".",
        capsize=3,
        label="Train data",
    )

    # NN predictions at train points (optional, but nice)
    plt.plot(x_train, y_pred_train,"x",c = 'green', label="NN Predictions (train)")

    plt.title("Neural Network Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/nn_plot.png")
    plt.close()


    print("\nSaved: NN_history.txt, NN_convergence.png, nn_plot.png")
    print("==== Neural Network example completed ====\n")
