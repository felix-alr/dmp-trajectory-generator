"""
For reasons of efficiency, the functions simulate and plot_states have been created using ChatGPT model GPT-5.5 (Medium).
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from collections.abc import Callable
from numpy.typing import NDArray
from dmp import (DMPParam, DMPNDim, GaussianBasis)

StateVector = NDArray[np.float64]
ExecuteFunction = Callable[[float], StateVector]

def simulate(
    execute: ExecuteFunction,
    delta_t: float,
    steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Runs a step-by-step simulation.

    Parameters
    ----------
    execute:
        A function that takes delta_t and returns the new system state
        as a NumPy array with shape (n, 3).

    delta_t:
        Time step size.

    steps:
        Number of simulation steps.

    Returns
    -------
    times:
        NumPy array with shape (steps,).

    states:
        NumPy array with shape (steps, n, 3), where states[step, i]
        is [z1, z2, z3] for system element i at that step.
    """

    times = np.zeros(steps)
    states: NDArray[np.float64] | None = None

    for step in range(steps):
        t = step * delta_t
        state = execute(delta_t)

        if not isinstance(state, np.ndarray):
            raise TypeError(f"execute(delta_t) must return np.ndarray, got {type(state)}")

        if state.ndim != 2 or state.shape[1] != 3:
            raise ValueError(f"Expected state shape (n, 3), got {state.shape}")

        if states is None:
            n = state.shape[0]
            states = np.zeros((steps, n, 3), dtype=np.float64)
        elif state.shape != states.shape[1:]:
            raise ValueError(
                f"Expected state shape {states.shape[1:]}, got {state.shape}"
            )

        times[step] = t
        states[step, :, :] = state

    if states is None:
        states = np.zeros((steps, 0, 3), dtype=np.float64)

    return times, states


def plot_states(
    times: NDArray[np.float64],
    states: NDArray[np.float64],
) -> None:
    """
    Plots z1, z2, and z3 as functions of time.
    """
    for k in range(states.shape[1]):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

        labels = ["z1", "z2", "z3"]

        for i, ax in enumerate(axes):
            ax.plot(times, states[:, k, i], label=labels[i])
            ax.set_ylabel(labels[i])
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Time")

        fig.suptitle(f"System State Evolution (z1, z2, z3) for dimension {k}")
        plt.tight_layout()

    plt.show()

# alpha_y = 25 beta_y = alpha_y/4 (for critical damping)
# alpha_x not used yet, thus simply set to 1
# tao = 0.2
# g = 1
dmp_param = DMPParam(25, 6.25, 5, 0.02, np.array([5,10,100]))
dmp_basis = GaussianBasis(np.array([3, 1, 1]), np.array([100, 50, 10]))
dmp = DMPNDim(dmp_param, dmp_basis, np.reshape(np.array([[5000, 1000, 0], [-10000, -500, -5000], [500, 10000, -100]]), shape=(3,dmp_basis.get_size())))
steps = 100

times, states = simulate(dmp.execute, 1.1*dmp_param.tao/steps, steps)
plot_states(times, states)