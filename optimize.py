import numpy as np
from utils import numerical_grad
from optimizers.sgd import sgd_update, sgd_init
from optimizers.momentum import momentum_update, momentum_init
from optimizers.rmsprop import rms_update, rms_init
from optimizers.adam import adam_update, adam_init

# Map optimizer names to their init and update functions
OPTIMIZERS = {
    "sgd": (sgd_init, sgd_update),
    "momentum": (momentum_init, momentum_update),
    "rmsprop": (rms_init, rms_update),
    "adam": (adam_init, adam_update)
}

def optimize(f, x0=None, c=None, optimizer="sgd", lr=0.01,
             max_iters=1000, tol=1e-6, verbose=False, **kwargs):
    """
    Generic optimization loop for minimizing f(x, c) using any supported optimizer.

    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from {list(OPTIMIZERS.keys())}.")

    optimizer_init, optimizer_update = OPTIMIZERS[optimizer]

    # handle x0
    if x0 is None:
        if "n_params" in kwargs:
            n = kwargs.pop("n_params")
            x0 = np.random.randn(n)
        else:
            raise ValueError("x0 not provided and cannot infer size. Provide x0 or n_params.")

    x = np.array(x0, dtype=float)
    state = optimizer_init(x)
    history_detailed = []

    for i in range(1, max_iters + 1):
        grad = numerical_grad(f, x, c)
        optimizer_kwargs = kwargs.copy()
        optimizer_kwargs.pop("n_params", None)

        x, state = optimizer_update(x, grad, state, lr=lr, **optimizer_kwargs)

        f_val = f(x, c)
        grad_norm = np.linalg.norm(grad)
        history_detailed.append({
            "iter": i,
            "x": x.copy(),
            "f_val": f_val,
            "grad": grad.copy(),
            "grad_norm": grad_norm
        })

        if grad_norm < tol:
            if verbose:
                print(f"Converged in {i} iterations. f(x) = {f_val:.6f}")
            break

        if verbose and i % 100 == 0:
            print(f"Iter {i}: f(x) = {f_val:.6f}, ||grad|| = {grad_norm:.6f}")

    return x, history_detailed
