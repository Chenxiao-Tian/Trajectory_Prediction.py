"""
Created on Tue Dec 19 13:50:24 2023

@author: ct347
"""

import numpy as np
from scipy.linalg import expm


def expected_number_of_jumps(a, b, t0, t1, Q):
    """
    Calculate the expected number of jumps in a Markov process.

    Args:
    - a, b: State indices.
    - t0, t1: Start and end times.
    - Q: Transition rate matrix.

    Returns:
    - Float: Expected number of jumps.
    """
    n_states = len(Q)
    T = t1 - t0
    m = np.max(np.abs(Q.diagonal()))
    tpm = expm(Q * T)
    p_ab = tpm[a - 1, b - 1]
    R = np.eye(n_states) + Q / m
    return m * T * (R.dot(tpm)[a - 1, b - 1]) / p_ab


def sample_path(a, b, t0, t1, Q):
    """
    Sample a path for a Markov process.

    Args:
    - a, b: State indices.
    - t0, t1: Start and end times.
    - Q: Transition rate matrix.

    Returns:
    - Numpy Array: Sampled path.
    """
    n_states = len(Q)
    T = t1 - t0
    m = np.max(np.abs(Q.diagonal()))
    tpm = expm(Q * T)
    p_ab = tpm[a - 1, b - 1]
    R = np.eye(n_states) + Q / m
    n_thresh = np.random.uniform()
    n_jumps = 0
    c_prob = np.exp(-m * T) * (a == b) / p_ab

    if c_prob > n_thresh:
        return np.array([[t0, t1], [a, b]])

    while c_prob < n_thresh:
        n_jumps += 1
        R_pow = np.linalg.matrix_power(R, n_jumps)
        c_prob += (
            np.exp(-m * T)
            * np.power(m * T, n_jumps)
            / np.math.factorial(n_jumps)
            * R_pow[a - 1, b - 1]
            / p_ab
        )

    path_times = np.sort(np.random.uniform(t0, t1, n_jumps))
    path_states = np.random.choice(n_states, n_jumps, replace=True) + 1
    path = np.column_stack(
        (
            np.concatenate([[t0], path_times, [t1]]),
            np.concatenate([[a], path_states, [b]]),
        )
    )

    return path


def main():
    Q = np.array([[-1, 1, 0], [0, -0.75, 0.75], [0.5, 0.5, -1]])
    path = sample_path(a=1, b=3, t0=0, t1=5, Q=Q)
    print("Sampled Path:\n", path)


def net():
    Q = np.array(
        [
            [-1, 1 / 2, 1 / 2, 0, 0, 0, 0],
            [1 / 3, -1, 1 / 3, 1 / 3, 0, 0, 0],
            [1 / 2, 1 / 2, -1, 0, 0, 0, 0],
            [0, 1 / 4, 0, -1, 1 / 4, 1 / 4, 1 / 4],
            [0, 0, 0, 1 / 2, -1, 1 / 2, 0],
            [0, 0, 0, 1 / 3, 1 / 3, -1, 1 / 3],
            [0, 0, 0, 1 / 2, 0, 1 / 2, -1],
        ]
    )

    noj = expected_number_of_jumps(a=1, b=7, t0=0, t1=3, Q=Q)
    path = sample_path(a=1, b=7, t0=0, t1=5, Q=Q)

    print("Expected Number of Jumps:\n", noj)
    print("Sampled Path:\n", path)


net()
