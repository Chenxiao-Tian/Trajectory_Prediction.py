import random

import numpy as np


class Path:
    """
    A class describing the CTMC jumps

    :parameter
    length : int
        the number of jumps
    array_a : int
        every state
    array_b : float
        every jumping time
    """

    def __init__(self, length, a, b):
        self.length = length
        self.array_a = a
        self.array_b = b

    def update_path(self, newst, newt):
        self.length += 1
        self.array_a.append(newst)
        self.array_b.append(newt)

    def print_path(self):
        print("Length:", self.length)
        print("Array State:", self.array_a)
        print("Array Time:", self.array_b)


def simpleforward(bgnst, ratem, bgnt, tm):
    """
    Sampling from Markov chains using simple forward method.

    This function simulates sample paths from an
    end-point conditioned, continuous-time Markov chains (CTMC)
    with finite state space using simple sampling algorithm.


    Inputs
    ----------
    ratem : numpy.ndarray
        The rate matrix for the CTMC.
        The sum of each row add up to 0.
    bgnst : int
        State at the start point (t = bgnt)
    bgnt : float
        Time at the start point (t = tm)
    tm : float
        Endpoint of the time interval.

    Returns
    -------
    Path
        A class including these parameters
        length : int
            Numbers of nodes
        array_a : int
            Every state of jumping
        array_b : float
            Every time of jumping
    """
    p = Path(1, [bgnst], [bgnt])
    nst = len(ratem[0])
    stspace = list(range(nst))

    cnt = 0
    tau = bgnt
    newst = bgnst

    while tau < tm:
        if cnt != 0:
            p.update_path(newst, tau)
        oldst = p.array_a[cnt]
        tau = tau + random.expovariate(-ratem[oldst][oldst])
        weight = np.copy(ratem[oldst])
        weight[oldst] = 0
        newst = random.choices(stspace, weight)[0]
        cnt += 1
    return p


def modifiedforward(bgnst, endst, ratem, tm):
    """
    Sampling from Markov chains using modified rejection forward method.

    This function simulates sample paths from an
    end-point conditioned, continuous-time Markov chains (CTMC)
    with finite state space using modified rejection forward algorithm.


    Inputs
    ----------
    bgnst : int
        State at the start point (t = 0)
    endst : float
        State at the end point (t = tm)
    ratem : numpy.ndarray
        The rate matrix for the CTMC.
        The sum of each row add up to 0.
    tm : float
        Endpoint of the time interval.

    Returns
    -------
    Path
        A class including these parameters
        length : int
            Numbers of nodes
        array_a : int
            Every state of jumping
        array_b : float
            Every time of jumping
    """
    if bgnst == endst:
        p = simpleforward(bgnst, ratem, 0, tm)
        while p.array_a[p.length - 1] != endst:
            p = simpleforward(bgnst, ratem, 0, tm)
        p.print_path()
        return p
    else:
        tau = (
            np.log(1 - random.random() * (1 - np.exp(tm * ratem[bgnst][bgnst])))
            / ratem[bgnst][bgnst]
        )
        weight = np.copy(ratem[bgnst])
        weight[bgnst] = 0
        nst = len(ratem[0])
        stspace = list(range(nst))
        newst = random.choices(stspace, weight)[0]
        p = simpleforward(newst, ratem, tau, tm)
        while p.array_a[p.length - 1] != endst:
            p = simpleforward(newst, ratem, tau, tm)

        p.length += 1
        p.array_a = np.insert(p.array_a, 0, bgnst)
        p.array_b = np.insert(p.array_b, 0, 0)

        p.print_path()
        return p


# bgnst = 1  # State at the start point
# endst = 2  # State at the end point
# rate = np.array([[-2, 1, 1], [1, -2, 1], [1, 1, -2]])  # The rate matrix for the CTMC.


# # import time
# # start_time = time.time()

# # simpleforward(bgnst,rate,1,5)
# modifiedforward(bgnst, endst, rate, 100)

# end_time = time.time()
# execution_time = end_time - start_time

# print("Execution time: {} second".format(execution_time))
