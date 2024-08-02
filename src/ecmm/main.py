#!/usr/bin/python -tt
# =============================================================================
# File      : main.py -- some test
# Author    : Juergen Hackl <hackl@ibi.baug.ethz.ch>
# Creation  : 2018-08-22
# Time-stamp: <Fre 2019-02-22 14:10 juergen>
#
# Copyright (c) 2018 Juergen Hackl <hackl@ibi.baug.ethz.ch>
# =============================================================================
from functools import reduce

import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)


def sample_path(a, b, N, R):
    # Get the number of states and initialize vector of states
    n_states = len(R)

    # Construct the transition probability matrix and extract the a,b elem.
    P_ab = np.linalg.matrix_power(R, N)
    P_ab[a - 1, b - 1]

    # print(p_ab)
    # Initialize number of jumps and conditional probability of n jumps
    n_jumps = N

    # Initialize a 3rd order tensors for storing powers of R
    R_pow = np.zeros((n_states, n_states, n_jumps + 1))
    for i in range(n_jumps + 1):
        R_pow[:, :, i] = np.linalg.matrix_power(R, i)

    # initialize the path matrix
    path_nrows = n_jumps + 1
    path = np.zeros((path_nrows, 2))
    path[0, 0] = 0
    path[0, 1] = a
    # path[path_nrows - 1, 0] = N
    # path[path_nrows - 1, 1] = b

    # transition times are uniformly distributed in the
    # interval. Sample them, sort them, and place in path.
    # np.sort(np.random.uniform(0, 1, n_jumps))
    transitions = range(1, n_jumps + 1)
    path[1 : n_jumps + 1, 0] = transitions

    print(path)
    # Sample the states at the transition times
    for j in range(1, n_jumps + 1):
        Rx = R[int(path[j - 1, 1] - 1)]
        Rn1 = R_pow[:, b - 1, n_jumps - j]
        Rn2 = R_pow[int(path[j - 1, 1] - 1), b - 1, n_jumps - j + 1]
        print(type(Rx))
        print(Rn1)
        print(Rn2)
        print("------")
        state_probs = Rx * Rn1 / Rn2

        # NOTE: +1 is needed since state index is +1 of matrix index
        path[j, 1] = np.random.choice(n_states, 1, False, state_probs)[0] + 1

    # Determine which transitions are virtual transitions
    # keep_inds = np.ones(path_nrows, dtype=bool)
    # for j in range(1, n_jumps + 1):
    #     if path[j, 1] == path[j-1, 1]:
    #         keep_inds[j] = False

    # create a matrix for the complete path without virtual jumps
    # return path[keep_inds]
    return path.T[1]


def P_matrix(a, b, u, v, i, N, R):
    return (
        R[u, v]
        * np.linalg.matrix_power(R, N - i)[v, b]
        / np.linalg.matrix_power(R, N - i + 1)[u, b]
    )


def Muv(a, b, u, v, N, R):
    a -= 1
    b -= 1
    u -= 1
    v -= 1
    M = 0
    for i in range(1, N + 1):
        M += (
            np.linalg.matrix_power(R, i - 1)[a, u]
            * R[u, v]
            * np.linalg.matrix_power(R, N - i)[v, b]
        )

    return M


def path_prob(path, a, b, N, R):
    n_jumps = N

    _p = []
    _s = 0
    for i in range(1, n_jumps + 1):
        u = path[i - 1]
        v = path[i]
        p = P_matrix(a - 1, b - 1, u - 1, v - 1, i, N, R)
        _p.append(p)
        # print(_p)
        # _s += reduce(lambda x, y: x*y, _p)
        # print(_s)

    #    print(_p)
    _r = reduce(lambda x, y: x * y, _p)
    print(_r)
    # print(_r/_s)

    # a += 1
    # b += 1
    # # Get the number of states and initialize vector of states
    # n_states = len(R)

    # # Construct the transition probability matrix and extract the a,b elem.
    # P_ab = np.linalg.matrix_power(R, N)
    # p_ab = P_ab[a-1, b-1]

    # print(p_ab)
    # # Initialize number of jumps and conditional probability of n jumps
    # n_jumps = N

    # # Initialize a 3rd order tensors for storing powers of R
    # R_pow = np.zeros((n_states, n_states, n_jumps+1))
    # for i in range(n_jumps+1):
    #     R_pow[:, :, i] = np.linalg.matrix_power(R, i)

    # # initialize the path matrix
    # path_nrows = n_jumps + 1
    # path = np.zeros((path_nrows, 2))
    # path[0, 0] = 0
    # path[0, 1] = a
    # # path[path_nrows - 1, 0] = N
    # # path[path_nrows - 1, 1] = b

    # # transition times are uniformly distributed in the
    # # interval. Sample them, sort them, and place in path.
    # # np.sort(np.random.uniform(0, 1, n_jumps))
    # transitions = range(1, n_jumps+1)
    # path[1:n_jumps+1, 0] = transitions

    # #_x = np.array([1, 3, 2, 4, 7], dtype='int')
    # _x = np.array([1, 2, 4, 6, 7], dtype='int')
    # _p = []
    # # Sample the states at the transition times
    # for j in range(1, n_jumps+1):
    #     Rx = R[int(path[j-1, 1]-1)]
    #     Rn1 = R_pow[:, b-1, n_jumps-j]
    #     Rn2 = R_pow[int(path[j-1, 1]-1), b-1, n_jumps-j+1]
    #     state_probs = Rx*Rn1/Rn2
    #     print(state_probs)
    #     path[j, 1] = _x[j]
    #     print(state_probs[_x[j]-1])
    #     _p.append(state_probs[_x[j]-1])
    # print(_p)

    # I = np.zeros((n_states, n_states))
    # for i in range(0, n_jumps):
    #     I = I+np.linalg.matrix_power(R, i) * \
    #         np.linalg.matrix_power(R, n_jumps-i)

    # s = []
    # for j in range(1, n_jumps+1):
    #     s.append(I[_x[j-1]-1, _x[j]-1])

    # print(s)
    # print(sum(s))


def Icd(N, R):
    n_states = len(R)
    II = np.zeros((n_states, n_states))
    for i in range(0, N):
        II = II + np.linalg.matrix_power(R, i) * np.linalg.matrix_power(R, N - i)
    return II


def Da(a, b, u, N, R):
    a -= 1
    b -= 1
    u -= 1
    M = 0
    for i in range(N):
        M += np.linalg.matrix_power(R, i)[a, u] * np.linalg.matrix_power(R, N - i)[u, b]
    return M


def PNuv(a, b, u, v, K, N, R):
    a -= 1
    b -= 1
    u -= 1
    v -= 1

    n_states = len(R)
    U = np.zeros((n_states, n_states))
    U[u, v] = 1

    P = np.zeros((N + 1, N + 1, n_states, n_states))
    N + 1
    P[0, 0] = np.eye(n_states)
    P[0, 1] = R - R * U
    P[1, 1] = R * U

    for n in range(2, N + 1):
        P[0, n] = P[0, n - 1].dot(R - R * U)

    for n in range(1, N + 1):
        for k in range(1, n):
            P[k, n] = P[k - 1, n - 1].dot(R * U) + P[k, n - 1].dot(R - R * U)
            # _P = np.zeros((n_states, n_states))
            # for c in range(1, M):
            #     _P = _P + P[k, n, a, c] * R[c, b]
            # P[k, n] = _P

    print(P[1, 5, 0, 6])
    return P[3, 5, 0, 6]
    # n_states = len(R)
    # U = np.zeros((n_states, n_states))

    # U[u, v] = 1

    # Puv = {}
    # Puv[(0, 0)] = np.eye(n_states)
    # Puv[(0, 1)] = R - R * U
    # Puv[(1, 1)] = R * U

    # for k in range(1, N+1):
    #     Puv[(k, 0)] = np.eye(n_states)  # np.zeros((n_states, n_states))

    # for n in range(2, N+1):
    #     Puv[(0, n)] = Puv[(0, n-1)].dot(R-R*U)

    # for n in range(1, N+1):
    #     for k in range(1, N+1):
    #         misc = Puv.get((k, n), None)
    #         if misc is None:
    #             Puv[(k, n)] = Puv[(k-1, n-1)].dot(R*U) + \
    #                 Puv[(k, n-1)].dot(R-R*U)

    # print('p', Puv[4, 4][u, v])
    # return Puv[0, 2]

    pass


# def PDa(a, b, u, K, N, R):
#     a -= 1
#     b -= 1
#     u -= 1

#     n_states = len(R)
#     V = np.zeros((n_states, n_states))
#     U = np.zeros((n_states, n_states))
#     H = np.zeros((N+1, N+1))
#     U[u, u] = 1

#     P = np.zeros((K, N))
#     P[0, 0] = 0
#     P[1, 0] = 1
#     print(P)
# for i in range(n_states):
#     V[i, a] = 1

# Pa = {}
# Pa[(0, 0)] = np.eye(n_states) - U
# Pa[(1, 0)] = U

# for k in range(0, N+1):
#     Pa[(-1, k)] = np.eye(n_states) - U  # np.zeros((n_states, n_states))

# for k in range(1, N+1):
#     for j in range(k, N+1):
#         Pa[(j, k-1)] = np.zeros((n_states, n_states))
#         H[j, k-1] = 2
#     Pa[(k, k-1)] = U
#     H[k, k-1] = 1
# print(H)
# for k in range(0, N+1):
#     print('k', k)
#     for n in range(1, N+1):
#         print('n', n)
#         #     if k <= n:
#         Pa[(k, n)] = Pa[(k-1, n-1)].dot(R*V) + Pa[(k, n-1)].dot(R-R*V)

# print(Pa[1, 4].sum(axis=1))
# Puv = {}
# Puv[(0, 0)] = np.eye(n_states)
# Puv[(0, 1)] = R - R * U
# Puv[(1, 1)] = R * U

# for k in range(1, N+1):
#     Puv[(k, 0)] = np.eye(n_states)  # np.zeros((n_states, n_states))

# for n in range(2, N+1):
#     Puv[(0, n)] = Puv[(0, n-1)].dot(R-R*U)

# for n in range(1, N+1):
#     for k in range(1, N+1):
#         misc = Puv.get((k, n), None)
#         if misc is None:
#             Puv[(k, n)] = Puv[(k-1, n-1)].dot(R*U) + \
#                 Puv[(k, n-1)].dot(R-R*U)

# print('p', Puv[4, 4][u, v])
# return Puv[0, 2]


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

    np.array([1, 0, 0, 0, 0, 0, 0])

    # print(np.linalg.matrix_power(T, 10).dot(x))
    # path = sample_path(a=1, b=7, N=4, R=T)
    # print(path)
    # path = np.array([0, 2, 1, 3, 6], dtype='int')
    # path = np.array([0, 2, 1, 3, 6], dtype='int')

    # path = np.array([1, 2, 4, 6, 7], dtype='int')
    # path = np.array([1, 3, 2, 4, 7], dtype='int')
    # path = np.array([1, 3, 2, 4, 6, 7], dtype='int')

    # path_prob(path=path, a=1, b=7, N=5, R=T)

    # sim = []
    # for i in range(100):
    #     sim.append(sample_path(a=1, b=7, N=5, R=T).tolist())

    # print(sim.count([1, 3, 2, 4, 6, 7]))
    # print(sim.count([1, 2, 4, 6, 7]))
    # print(sim.count([1, 3, 2, 4, 7]))

    # M = Muv(a=1, b=7, u=2, v=4, N=3, R=T)
    # print(M)

    N = 20
    Icd(N, R=T)
    print(T)
    Pab = np.linalg.matrix_power(T, N)
    # print(I[1, 1])
    # print(Pab[0, 6])
    print(Pab)
    # print(I[1, 1]/Pab[0, 6])

    _D = Da(a=1, b=7, u=2, N=N, R=T)
    print(_D)
    print(_D / Pab[0, 6])

    _N = Muv(a=1, b=7, u=2, v=4, N=N, R=T)
    print(_N)
    print(_N / Pab[0, 6])

    # sim = []
    # for i in range(100):
    #     sim.append(sample_path(a=1, b=7, N=N, R=T).tolist())

    # #print(np.mean([s.count(2) for s in sim]))
    # r = []
    # for s in sim:
    #     c = 0
    #     for item1, item2 in zip(s, s[1:]):
    #         if item1 == 2 and item2 == 4:
    #             c += 1
    #     r.append(c)

    # print(np.mean(r))

    P = PNuv(a=1, b=7, u=2, v=4, K=3, N=5, R=T)
    Pab = np.linalg.matrix_power(T, 5)
    print(P / Pab[0, 6])
    # print(P[1, 3])
    # Q = P/Pab[0, 6]
    # print(Q[1, 1])

    print(sample_path(a=1, b=7, N=3, R=T))


#    P = PDa(a=1, b=7, u=2, K=4, N=4, R=T)


main()
# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:
