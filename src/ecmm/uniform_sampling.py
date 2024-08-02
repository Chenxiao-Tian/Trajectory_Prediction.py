import math
import random

import numpy as np
from scipy import linalg


def uniformsampling(Q, T, a, b):
    """
    Sampling from Markov chains.

    This function simulates sample paths from an
    end-point conditioned, continuous-time Markov chains (CTMC)
    with finite state space using uniform sampling algorithm.


    Parameters
    ----------
    Q : numpy.ndarray
        The rate matrix for the CTMC.
        The sum of each row add up to 0.
    T : float
        Endpoint of the time interval.
    a : int
        State at the start point (t = 0)
    b : int
        State at the end point (t = T)

    Returns
    -------
    Lists
        the sample path simulated
    """
    d = len(Q)
    mu = max(np.diagonal(Q))
    R = np.identity(d) + Q / mu
    P = linalg.expm(T * Q)
    path = []

    u = random.uniform(0, 1)
    summ = 0
    n = 0
    while summ < u:
        n = n + 1
        Rn = np.linalg.matrix_power(R, n)
        summ = (
            summ
            + np.exp(mu * T) * (mu * T) ** n / math.factorial(n) * Rn[a, b] / P[a, b]
        )

    if n == 0:
        path.append((a, 0))
        path.append((a, T))

    elif n == 1 and a == b:
        path.append((a, 0))
        path.append((a, T))

    elif n == 1 and a != b:
        tau = random.uniform(0, T)
        path.append((a, 0))
        path.append((b, tau))
        path.append((b, T))

    else:
        random_tau = [random.uniform(0, T) for i in range(n)]
        tau = sorted(random_tau)
        path.append((a, 0))
        x = a
        for i in range(n):
            Rp1 = np.linalg.matrix_power(R, n - i - 1)
            Rp1 = Rp1[:, b]
            Rp2 = np.linalg.matrix_power(R, n - i)
            Rp2 = Rp2[x, b]
            nominator = np.multiply(R[x, :], Rp1)
            x = random.choices(population=range(d), weights=nominator / Rp2, k=1)[0]
            path.append((x, tau[i]))
        path.append((b, T))

    return path
