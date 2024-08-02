#!/usr/bin/python -tt
# =============================================================================
# File      : sample_paths.py -- Sampling paths from endpoint-conditioned MC
# Author    : Juergen Hackl <hackl@ibi.baug.ethz.ch>
# Creation  : 2018-08-22
# Time-stamp: <Mit 2018-08-22 17:03 juergen>
#
# Copyright (c) 2018 Juergen Hackl <hackl@ibi.baug.ethz.ch>
# =============================================================================
import numpy as np
from scipy.linalg import expm


def expected_number_of_jumps(a, b, t0, t1, Q):
    # Get the number of states and initialize vector of states
    n_states = len(Q)
    # states = np.zeros(n_states)

    # Get the length of the interval and the largest diagonal element of Q
    T = t1 - t0
    m = np.max(np.abs(Q.diagonal()))

    # Construct the transition probability matrix and extract the a,b elem.
    tpm = expm(Q * T)
    p_ab = tpm[a - 1, b - 1]

    # Generate the auxilliary transition matrix
    R = np.eye(n_states) + Q / m

    # expected number of jumps
    return m * T * (R.dot(tpm)[a - 1, b - 1]) / p_ab


def sample_path(a, b, t0, t1, Q):
    # Get the number of states and initialize vector of states
    n_states = len(Q)
    # states = np.zeros(n_states)

    # Get the length of the interval and the largest diagonal element of Q
    T = t1 - t0
    m = np.max(np.abs(Q.diagonal()))

    # Construct the transition probability matrix and extract the a,b elem.
    tpm = expm(Q * T)
    p_ab = tpm[a - 1, b - 1]
    print(p_ab)
    # Generate the auxilliary transition matrix
    R = np.eye(n_states) + Q / m
    # print(np.linalg.matrix_power(R, 1000))
    # Sample threshold for determining the number of states
    n_thresh = np.random.uniform()

    # Initialize number of jumps and conditional probability of n jumps
    n_jumps = 0
    c_prob = np.exp(-m * T) * (a == b) / p_ab

    # proceed with sampling by uniformization
    # first the cast when there are no jumps
    if c_prob > n_thresh:
        # initialize matrix
        path = np.zeros((2, 2))

        # fill matrix
        path[0, 0] = t0
        path[0, 1] = t1
        path[1, 0] = a
        path[1, 1] = b

        return path
    else:
        # increment the number of jumps and compute c_prob
        n_jumps += 1
        c_prob += (
            np.exp(-m * T)
            * np.power(m * T, n_jumps)
            / np.math.factorial(n_jumps)
            * R[a - 1, b - 1]
            / p_ab
        )

        # if there is exactly one jump
        if c_prob > n_thresh:
            # if the endpoints match, the only jump is a virtual one
            if a == b:
                # initialize matrix
                path = np.zeros((2, 2))

                # fill matrix
                path[0, 0] = t0
                path[0, 1] = t1
                path[1, 0] = a
                path[1, 1] = b

                return path
            else:
                # initialize matrix
                path = np.zeros((3, 2))

                # fill matrix
                path[0, 0] = t0
                path[0, 1] = a
                path[1, 0] = np.random.uniform(t0, t1)
                path[1, 1] = b
                path[2, 0] = t1
                path[2, 1] = b

                return path
        # else, there are at least two jumps
        else:
            # Initialize a 3rd order tensors for storing powers of R
            R_pow = np.zeros((n_states, n_states, 8))
            R_pow[:, :, 0] = np.eye(n_states)
            R_pow[:, :, 1] = R

            # Initialize a vector for storing the transition probabilities
            state_probs = np.zeros(n_states)

            # keep calculating the conditional probability of n jumps until
            # the threshold is exceeded. store powers of R accordingly.
            while c_prob < n_thresh:
                # NOTE: Probably a faster way would be using the power instead an array
                # NOTE: print(np.linalg.matrix_power(R, n_jumps))

                # increment the number of jumps
                n_jumps += 1

                # check whether to add additional slices to the R_pow tensor
                if n_jumps == R_pow.shape[2]:
                    R_pow = np.dstack((R_pow, np.zeros((n_states, n_states))))

                # Add the new power of R to the tensor and calculate c_prob
                R_pow[:, :, n_jumps] = R_pow[:, :, n_jumps - 1].dot(R)

                c_prob += (
                    np.exp(-m * T)
                    * np.power(m * T, n_jumps)
                    / np.math.factorial(n_jumps)
                    * R_pow[a - 1, b - 1, n_jumps]
                    / p_ab
                )

            # initialize the path matrix
            path_nrows = n_jumps + 2
            path = np.zeros((path_nrows, 2))
            path[0, 0] = t0
            path[0, 1] = a
            path[path_nrows - 1, 0] = t1
            path[path_nrows - 1, 1] = b

            # transition times are uniformly distributed in the
            # interval. Sample them, sort them, and place in path.
            transitions = np.sort(np.random.uniform(t0, t1, n_jumps))
            path[1 : n_jumps + 1, 0] = transitions

            # Sample the states at the transition times
            for j in range(1, n_jumps + 1):
                Rx = R[int(path[j - 1, 1] - 1)]
                Rn1 = R_pow[:, b - 1, n_jumps - j]
                Rn2 = R_pow[int(path[j - 1, 1] - 1), b - 1, n_jumps - j + 1]
                state_probs = Rx * Rn1 / Rn2

                # NOTE: +1 is needed since state index is +1 of matrix index
                path[j, 1] = np.random.choice(n_states, 1, False, state_probs)[0] + 1

            # Determine which transitions are virtual transitions
            keep_inds = np.ones(path_nrows, dtype=bool)
            for j in range(1, n_jumps + 1):
                if path[j, 1] == path[j - 1, 1]:
                    keep_inds[j] = False

            # create a matrix for the complete path without virtual jumps
            # return path
            return path[keep_inds]


def main():
    Q = np.array([[-1, 1, 0], [0, -0.75, 0.75], [0.5, 0.5, -1]])
    path = sample_path(a=1, b=3, t0=0, t1=5, Q=Q)
    print(path)


def net():
    np.array(
        [
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
        ]
    )

    np.array(
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
    print(noj)

    # sim = []
    # for i in range(1000):
    #     sim.append(len(sample_path(a=1, b=7, t0=0, t1=.2, Q=Q))-2)

    # print(np.mean(sim))

    path = sample_path(a=1, b=7, t0=0, t1=5, Q=Q)

    print(path)


# main()
net()
# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:
