"""
Created on Wed Jan 31 10:35:30 2024

@author: ct347
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


class SamplePath:
    def __init__(self, Q):
        self.Q = Q

    def expected_number_of_jumps(self, a, b, t0, t1):
        n_states = len(self.Q)
        T = t1 - t0
        m = np.max(np.abs(self.Q.diagonal()))
        tpm = expm(self.Q * T)
        p_ab = tpm[a - 1, b - 1]
        R = np.eye(n_states) + self.Q / m
        return m * T * (R.dot(tpm)[a - 1, b - 1]) / p_ab

    def sample_path(self, a, b, t0, t1):
        n_states = len(self.Q)
        T = t1 - t0
        m = np.max(np.abs(self.Q.diagonal()))
        tpm = expm(self.Q * T)
        p_ab = tpm[a - 1, b - 1]
        R = np.eye(n_states) + self.Q / m
        n_thresh = np.random.uniform()
        n_jumps = 0
        c_prob = np.exp(-m * T) * (a == b) / p_ab

        if c_prob > n_thresh:
            path = np.zeros((2, 2))
            path[:, 0] = [t0, a]
            path[:, 1] = [t1, b]
            return path
        else:
            while True:
                n_jumps += 1
                c_prob += (
                    np.exp(-m * T)
                    * (m * T) ** n_jumps
                    / np.math.factorial(n_jumps)
                    * np.linalg.matrix_power(R, n_jumps)[a - 1, b - 1]
                    / p_ab
                )
                if c_prob > n_thresh:
                    break

            transitions = np.sort(np.random.uniform(t0, t1, n_jumps))
            states = [a]
            for _i in range(n_jumps):
                current_state = states[-1] - 1

                transition_probs = (
                    R[current_state] * tpm[:, b - 1] / tpm[current_state, b - 1]
                )

                transition_probs = (
                    R[current_state] * tpm[:, b - 1] / tpm[current_state, b - 1]
                )

                transition_probs /= transition_probs.sum()

                next_state = (
                    np.random.choice(np.arange(n_states), p=transition_probs) + 1
                )
                states.append(next_state)
            states.append(b)

            path = np.zeros((n_jumps + 2, 2))
            path[:, 0] = np.concatenate(([t0], transitions, [t1]))
            path[:, 1] = states
            return path


class SamplePathPlotter:
    def __init__(self, sample_path_instance):
        self.sample_path_instance = sample_path_instance

    def plot_sample_path(self, a, b, t0, t1, show=True):
        path = self.sample_path_instance.sample_path(a, b, t0, t1)
        plt.figure(figsize=(10, 6))
        plt.step(path[:, 0], path[:, 1], where="post", label="Sample Path")
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.title(
            f"Sample Path from State {a} to State {b} between t0={t0} and t1={t1}"
        )
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()

    def plot_expected_number_of_jumps(self, a, b, t0, t1, show=True):
        N = self.sample_path_instance.expected_number_of_jumps(a, b, t0, t1)
        plt.figure(figsize=(6, 4))
        plt.bar(["Expected Jumps"], [N], color="skyblue")
        plt.title(f"Expected Number of Jumps from State {a} to State {b}")
        plt.ylabel("Number of Jumps")
        if show:
            plt.show()


# Example usage
Q = np.array([[-1, 1, 0], [0, -0.75, 0.75], [0.5, 0.5, -1]])
sample_path_instance = SamplePath(Q)
plotter = SamplePathPlotter(sample_path_instance)

# To plot a sample path
plotter.plot_sample_path(a=1, b=3, t0=0, t1=5)

# To plot expected number of jumps
plotter.plot_expected_number_of_jumps(a=1, b=3, t0=0, t1=5)

# Example of how to use the SamplePath class
Q = np.array([[-1, 1, 0], [0, -0.75, 0.75], [0.5, 0.5, -1]])
sample_path_instance = SamplePath(Q)
path = sample_path_instance.sample_path(a=1, b=3, t0=0, t1=5)
N = sample_path_instance.expected_number_of_jumps(a=1, b=3, t0=0, t1=5)
print(path)
print(N)
