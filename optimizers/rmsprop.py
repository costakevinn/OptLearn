import numpy as np

def rms_init(params):
    """
    Initialize RMSProp optimizer state.
    """
    state = {}
    state["v"] = np.zeros_like(params)
    return state

def rms_update(params, grads, state, lr=1e-3, beta=0.9, eps=1e-8, **kwargs):
    """
    Perform one RMSProp update step.
    
    """
    # Initialize state if empty
    if not state:
        state = rms_init(params)

    # update moving average of squared gradients
    state["v"] = beta * state["v"] + (1 - beta) * (grads ** 2)

    # update parameters
    params = params - lr * grads / (np.sqrt(state["v"]) + eps)

    return params, state
