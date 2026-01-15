import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Numerical gradient
# -------------------------------
def numerical_grad(func, params, c=None, eps=1e-5):
    grads = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grads[i] = (func(params_plus, c) - func(params_minus, c)) / (2 * eps)
    return grads

# -------------------------------
# Save optimization history
# -------------------------------
def save_history(history_detailed, filename="optimization_history.txt"):
    if not history_detailed:
        print("No history to save.")
        return
    n_params = len(history_detailed[0]["x"])
    header = "iter\tf_val\tgrad_norm\t" + "\t".join([f"x{i}" for i in range(n_params)])
    with open(filename, "w") as f:
        f.write(header + "\n")
        for h in history_detailed:
            line = [str(h["iter"]), f"{h['f_val']:.8f}", f"{h['grad_norm']:.8f}"]
            line += [f"{v:.8f}" for v in h["x"]]
            f.write("\t".join(line) + "\n")
    print(f"History saved to {filename}")

# -------------------------------
# Print summary
# -------------------------------
def print_summary(history_detailed, last_n=5):
    print("\n--- Optimization Summary (last {} iterations) ---".format(last_n))
    for h in history_detailed[-last_n:]:
        print(f"Iter {h['iter']:>4}: f_val={h['f_val']:.6f}, grad_norm={h['grad_norm']:.6f}, x={h['x']}")
    print("--- End Summary ---\n")

# -------------------------------
# Plot convergence
# -------------------------------
def plot_convergence(history_detailed, filename=None, title="Convergence"):
    f_vals = [h["f_val"] for h in history_detailed]
    grad_norms = [h["grad_norm"] for h in history_detailed]
    iters = [h["iter"] for h in history_detailed]

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x)", color=color)
    ax1.plot(iters, f_vals, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("||grad||", color=color)
    ax2.plot(iters, grad_norms, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title)

    if filename:
        plt.savefig(filename)
        print(f"Convergence plot saved to {filename}")
    else:
        plt.show()
