import os
import numpy as np

from optimize import optimize
from utils import numerical_grad, save_history, print_summary, plot_convergence


# ============================================================
# Example objective functions
# ============================================================
def quadratic(x, c=None):
    """Quadratic function with optional constant vector c."""
    if c is None:
        c = [3, -2]
    return (x[0] - c[0])**2 + (x[1] - c[1])**2


def rosenbrock(x, c=None):
    """Classic Rosenbrock function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def himmelblau(x, c=None):
    """Himmelblau function."""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


# ============================================================
# Generic runner
# ============================================================
def run_example(
    func,
    name,
    n_params=2,
    optimizer="adam",
    lr=0.01,
    max_iters=1000,
    tol=1e-6,
    c=None,
):
    print(f"\n==== Running {name} example ====")

    # Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Run optimization
    x_opt, history_detailed = optimize(
        func,
        n_params=n_params,
        optimizer=optimizer,
        lr=lr,
        max_iters=max_iters,
        tol=tol,
        verbose=True,
        c=c,
    )

    # Final gradient
    grad_final = numerical_grad(func, x_opt, c)
    grad_norm_final = np.linalg.norm(grad_final)
    print(f"Gradient at final x: {grad_final}, norm: {grad_norm_final:.6f}")

    # Save results
    save_history(history_detailed, filename=f"results/{name}_history.txt")
    print_summary(history_detailed, last_n=5)

    plot_convergence(
        history_detailed,
        filename=f"plots/{name}_convergence.png",
        title=f"{name} Convergence ({optimizer}, lr={lr})",
    )

    print(f"{name} optimized x: {x_opt}")
    print(f"{name} final loss: {history_detailed[-1]['f_val']:.6e}")
    print(f"==== {name} example completed ====\n")


# ============================================================
# Specific runners (different optimizer per example)
# ============================================================
def run_quadratic_example():
    # Quadratic is easy → SGD works great with higher LR
    c = [1, 2]
    run_example(
        quadratic,
        name="Quadratic",
        n_params=2,
        optimizer="sgd",
        lr=0.2,
        max_iters=500,
        tol=1e-8,
        c=c,
    )


def run_rosenbrock_example():
    # Rosenbrock is narrow/curved valley → Momentum helps a lot
    run_example(
        rosenbrock,
        name="Rosenbrock",
        n_params=2,
        optimizer="momentum",
        lr=0.005,
        max_iters=3000,
        tol=1e-6,
    )


def run_himmelblau_example():
    # Himmelblau has multiple basins → RMSProp stabilizes steps
    run_example(
        himmelblau,
        name="Himmelblau",
        n_params=2,
        optimizer="rmsprop",
        lr=0.01,
        max_iters=2000,
        tol=1e-6,
    )
