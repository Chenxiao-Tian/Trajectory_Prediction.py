from functools import reduce

import numpy as np

np.set_printoptions(suppress=True, linewidth=200)


def sample_path(a, b, N, R):
    """
    Generates a sample path using the transition matrix R.

    Args:
    - a: Starting state.
    - b: Ending state.
    - N: Number of steps.
    - R: Transition matrix.

    Returns:
    - Array: The sample path.
    """
    n_states = len(R)
    R_pow = np.zeros((n_states, n_states, N + 1))

    for i in range(N + 1):
        R_pow[:, :, i] = np.linalg.matrix_power(R, i)

    path = np.zeros((N + 1, 2))
    path[0, 0] = 0
    path[0, 1] = a
    path[1 : N + 1, 0] = range(1, N + 1)

    for j in range(1, N + 1):
        Rx = R[int(path[j - 1, 1]) - 1]
        Rn1 = R_pow[:, b - 1, N - j]
        Rn2 = R_pow[int(path[j - 1, 1]) - 1, b - 1, N - j + 1]
        state_probs = Rx * Rn1 / Rn2
        path[j, 1] = np.random.choice(n_states, 1, False, state_probs)[0] + 1

    return path.T[1]


def P_matrix(a, b, u, v, i, N, R):
    """
    Calculates the transition probability matrix.

    Args:
    - a, b, u, v: State indices.
    - i: Current step.
    - N: Total number of steps.
    - R: Transition matrix.

    Returns:
    - Float: Probability value.
    """
    return (
        R[u, v]
        * np.linalg.matrix_power(R, N - i)[v, b]
        / np.linalg.matrix_power(R, N - i + 1)[u, b]
    )


def Muv(a, b, u, v, N, R):
    """
    Computes Muv for given parameters.

    Args:
    - a, b, u, v: State indices.
    - N: Number of steps.
    - R: Transition matrix.

    Returns:
    - Float: Calculated Muv value.
    """
    a, b, u, v = a - 1, b - 1, u - 1, v - 1
    M = sum(
        np.linalg.matrix_power(R, i - 1)[a, u]
        * R[u, v]
        * np.linalg.matrix_power(R, N - i)[v, b]
        for i in range(1, N + 1)
    )
    return M


def path_prob(path, a, b, N, R):
    """
    Computes the probability of a given path.

    Args:
    - path: Array of states in the path.
    - a, b: State indices.
    - N: Number of steps.
    - R: Transition matrix.

    Returns:
    - Float: Probability of the path.
    """
    probabilities = [
        P_matrix(a - 1, b - 1, path[i - 1] - 1, path[i] - 1, i, N, R)
        for i in range(1, N + 1)
    ]
    return reduce(lambda x, y: x * y, probabilities)


def Icd(N, R):
    """
    Computes the Icd matrix.

    Args:
    - N: Number of steps.
    - R: Transition matrix.

    Returns:
    - Numpy Array: The Icd matrix.
    """
    len(R)
    return sum(
        np.linalg.matrix_power(R, i) * np.linalg.matrix_power(R, N - i)
        for i in range(N)
    )


def Da(a, b, u, N, R):
    """
    Computes Da for given parameters.

    Args:
    - a, b, u: State indices.
    - N: Number of steps.
    - R: Transition matrix.

    Returns:
    - Float: Calculated Da value.
    """
    a, b, u = a - 1, b - 1, u - 1
    return sum(
        np.linalg.matrix_power(R, i)[a, u] * np.linalg.matrix_power(R, N - i)[u, b]
        for i in range(N)
    )


def PNuv(a, b, u, v, K, N, R):
    """
    Calculates PNuv for given parameters.

    Args:
    - a, b, u, v: State indices.
    - K, N: Number of steps.
    - R: Transition matrix.

    Returns:
    - Float: Calculated PNuv value.
    """
    a, b, u, v = a - 1, b - 1, u - 1, v - 1
    n_states = len(R)
    U = np.zeros((n_states, n_states))
    U[u, v] = 1

    P = np.zeros((N + 1, N + 1, n_states, n_states))
    P[0, 0] = np.eye(n_states)
    P[0, 1] = R - R * U
    P[1, 1] = R * U

    for n in range(2, N + 1):
        P[0, n] = P[0, n - 1].dot(R - R * U)

    for n in range(1, N + 1):
        for k in range(1, n):
            P[k, n] = P[k - 1, n - 1].dot(R * U) + P[k, n - 1].dot(R - R * U)

    return P[K, N, a, b]


def main():
    T = np.array(
        [
            [0, 1 / 2, 1 / 2, 0, 0, 0, 0],
            [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0],
            [1 / 2, 1 / 2, 0, 0, 0, 0, 0],
            [0, 1 / 4, 0, 0, 1 / 4, 1 / 4, 1 / 4],
            [0, 0, 0, 1 / 2, 0, 1 / 2, 0],
            [0, 0, 0, 1 / 3, 1 / 3, 0, 1 / 3],
            [0, 0, 0, 1 / 2, 0, 1 / 2, 0],
        ]
    )

    N = 20
    Pab = np.linalg.matrix_power(T, N)

    _D = Da(a=1, b=7, u=2, N=N, R=T)
    _N = Muv(a=1, b=7, u=2, v=4, N=N, R=T)
    P = PNuv(a=1, b=7, u=2, v=4, K=3, N=5, R=T)

    print("Transition Matrix:\n", T)
    print("Pab:\n", Pab)
    print("Da:\n", _D)
    print("Muv:\n", _N)
    print("PNuv:\n", P)
    print("Sample Path:\n", sample_path(a=1, b=7, N=3, R=T))


if __name__ == "__main__":
    main()
