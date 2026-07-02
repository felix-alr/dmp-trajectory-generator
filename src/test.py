"""
For reasons of efficiency, the functions simulate and plot_states have been created using ChatGPT model GPT-5.5 (Medium).
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from collections.abc import Callable
from numpy.typing import NDArray
from dmp import (DMPParam, DMP1Dim, GaussianBasis)

StateVector = NDArray[np.float64]
ExecuteFunction = Callable[[float], StateVector]


def simulate(
    execute: ExecuteFunction,
    delta_t: float,
    steps: int,
    print_steps: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Runs a step-by-step simulation.

    Parameters
    ----------
    execute:
        A function that takes delta_t and returns the new state vector
        as a NumPy array with shape (3,).

    delta_t:
        Time step size.

    steps:
        Number of simulation steps.

    print_steps:
        Whether to print every step.

    Returns
    -------
    times:
        NumPy array with shape (steps,).

    states:
        NumPy array with shape (steps, 3), where each row is [z1, z2, z3].
    """

    times = np.zeros(steps)
    states = np.zeros((steps, 3), dtype=np.float64)

    for step in range(steps):
        t = step * delta_t
        state = execute(delta_t)

        if not isinstance(state, np.ndarray):
            raise TypeError(f"execute(delta_t) must return np.ndarray, got {type(state)}")

        if state.shape != (3,):
            raise ValueError(f"Expected state shape (3,), got {state.shape}")

        times[step] = t
        states[step, :] = state

        if print_steps:
            z1, z2, z3 = state
            print(
                f"Step {step:03d} | "
                f"t = {t:.3f} | "
                f"z1 = {z1:.6f}, "
                f"z2 = {z2:.6f}, "
                f"z3 = {z3:.6f}"
            )

    return times, states


def plot_states(
    times: NDArray[np.float64],
    states: NDArray[np.float64],
) -> None:
    """
    Plots z1, z2, and z3 as functions of time.
    """

    plt.figure()

    plt.plot(times, states[:, 0], label="z1")
    #plt.plot(times, states[:, 1], label="z2")
    #plt.plot(times, states[:, 2], label="z3")

    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.title("3-State System Evolution")
    plt.legend()
    plt.grid(True)

    plt.show()

# alpha_y = 25 beta_y = alpha_y/4 (for critical damping)
# alpha_x not used yet, thus simply set to 1
# tao = 0.2
# g = 1
dmp_param = DMPParam(25, 6.25, 1, 0.02, 1)
dmp_basis = GaussianBasis(np.zeros(1), np.zeros(1))
dmp = DMP1Dim(dmp_basis, dmp_param)
steps = 100

times, states = simulate(dmp.execute, 2*dmp_param.tao/steps, steps, False)
plot_states(times, states)