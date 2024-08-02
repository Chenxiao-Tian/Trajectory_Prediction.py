"""
Created on Wed Jan 31 13:09:09 2024

@author: ct347
"""

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np


class EndpointConditionedMarkovModel:
    def __init__(self, transition_matrix):
        self.T = transition_matrix
        self.n_states = len(transition_matrix)

    def sample_path(self, a, b, N):
        R = self.T
        n_states = self.n_states
        R_pow = np.zeros((n_states, n_states, N + 1))
        for i in range(N + 1):
            R_pow[:, :, i] = np.linalg.matrix_power(R, i)

        path = np.zeros((N + 1, 2))
        path[0, :] = [0, a]

        for j in range(1, N + 1):
            Rx = R[int(path[j - 1, 1] - 1)]
            Rn1 = R_pow[:, b - 1, N - j]
            Rn2 = R_pow[int(path[j - 1, 1] - 1), b - 1, N - j + 1]
            state_probs = Rx * Rn1 / Rn2
            path[j, 1] = np.random.choice(n_states, 1, p=state_probs)[0] + 1
            path[j, 0] = j

        return path[:, 1]

    def P_matrix(self, a, b, u, v, i, N):
        R = self.T
        return (
            R[u, v]
            * np.linalg.matrix_power(R, N - i)[v, b]
            / np.linalg.matrix_power(R, N - i + 1)[u, b]
        )

    def Muv(self, a, b, u, v, N):
        R = self.T
        M = 0
        for i in range(1, N + 1):
            M += (
                np.linalg.matrix_power(R, i - 1)[a - 1, u - 1]
                * R[u - 1, v - 1]
                * np.linalg.matrix_power(R, N - i)[v - 1, b - 1]
            )
        return M

    def path_prob(self, path, a, b, N):
        _p = []
        for i in range(1, N + 1):
            u = int(path[i - 1])
            v = int(path[i])
            p = self.P_matrix(a - 1, b - 1, u - 1, v - 1, i, N)
            _p.append(p)
        _r = reduce(lambda x, y: x * y, _p)
        print(_r)

    def Icd(self, N):
        R = self.T
        II = np.zeros((self.n_states, self.n_states))
        for i in range(0, N):
            II += np.linalg.matrix_power(R, i) * np.linalg.matrix_power(R, N - i)
        return II

    def Da(self, a, b, u, N):
        R = self.T
        M = 0
        for i in range(N):
            M += (
                np.linalg.matrix_power(R, i)[a - 1, u - 1]
                * np.linalg.matrix_power(R, N - i)[u - 1, b - 1]
            )
        return M

    def PNuv(self, a, b, u, v, K, N):
        R = self.T
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
        # This function is more complex and depends on the specific use case
        # It may need additional logic based on the context in which it's used
        pass  # Placeholder for the function's logic


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

    model = EndpointConditionedMarkovModel(T)

    # Testing sample_path method
    path = model.sample_path(a=1, b=7, N=3)
    print("Sampled Path:", path)
    print(model.Muv(a=1, b=7, u=2, v=4, N=5))

    # Further tests can be added here for other methods


class MarkovModelPlotter:
    def __init__(self, markov_model):
        self.model = markov_model

    def plot_sample_path(self, a, b, N):
        path = self.model.sample_path(a, b, N)
        plt.figure(figsize=(10, 6))
        plt.plot(range(N + 1), path, marker="o", linestyle="-")
        plt.title(f"Sample Path from State {a} to State {b} over {N} Steps")
        plt.xlabel("Step")
        plt.ylabel("State")
        plt.xticks(range(N + 1))
        plt.yticks(np.unique(path))
        plt.grid(True)
        plt.show()

    def plot_transition_matrix(self):
        T = self.model.T
        plt.figure(figsize=(8, 8))
        plt.imshow(T, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Transition Probability")
        plt.title("Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.xticks(np.arange(self.model.n_states))
        plt.yticks(np.arange(self.model.n_states))
        plt.show()

    def plot_jumps(self, path):
        # Count jumps
        n_states = self.model.n_states
        jump_counts = np.zeros((n_states, n_states))
        for i in range(len(path) - 1):
            from_state = int(path[i]) - 1
            to_state = int(path[i + 1]) - 1
            jump_counts[from_state, to_state] += 1

        # Plotting the heatmap of jumps
        plt.figure(figsize=(8, 6))
        plt.imshow(jump_counts, cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Number of Jumps")
        plt.title("Jumps between States")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.xticks(np.arange(n_states))
        plt.yticks(np.arange(n_states))
        plt.show()


# Example usage
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
model = EndpointConditionedMarkovModel(T)
plotter = MarkovModelPlotter(model)

# To plot a sample path
plotter.plot_sample_path(a=1, b=7, N=10)
path = model.sample_path(a=1, b=7, N=10)
plotter.plot_jumps(path)
# To visualize the transition matrix
plotter.plot_transition_matrix()
if __name__ == "__main__":
    main()
