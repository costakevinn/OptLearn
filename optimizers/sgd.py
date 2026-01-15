import numpy as np

def sgd_init(params):
    """
    Initialize SGD optimizer state.
    SGD does not have internal state, but we keep this for uniform API.
    """
    state = {}
    return state

def sgd_update(params, grads, state, lr=1e-2, **kwargs):
    """
    Perform one SGD update step.
    """
    # Initialize state if empty
    if not state:
        state = sgd_init(params)

    # update parameters
    params = params - lr * grads

    return params, state
