import numpy as np

def momentum_init(params):
    """
    Initialize Momentum optimizer state.
    """
    state = {}
    state["v"] = np.zeros_like(params)
    return state

def momentum_update(params, grads, state, lr=0.01, beta=0.9, **kwargs):
    """
    Perform one Momentum update step.

    """
    # Initialize state if empty
    if not state:
        state = momentum_init(params)

    # update velocity
    state["v"] = beta * state["v"] + (1 - beta) * grads
    # update parameters
    params = params - lr * state["v"]

    return params, state
