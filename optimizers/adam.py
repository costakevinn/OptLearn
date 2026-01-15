import numpy as np


def adam_init(params):
    """
    Initialize Adam optimizer state.
    """
    state = {}
    state["m"] = np.zeros_like(params)
    state["v"] = np.zeros_like(params)
    state["t"] = 0
    return state


def adam_update(params, grads, state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Initialize state if empty
    if not state:
        state = adam_init(params)

    state["t"] += 1
    t = state["t"]

    state["m"] = beta1 * state["m"] + (1 - beta1) * grads
    state["v"] = beta2 * state["v"] + (1 - beta2) * (grads ** 2)

    m_hat = state["m"] / (1 - beta1 ** t)
    v_hat = state["v"] / (1 - beta2 ** t)

    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, state

