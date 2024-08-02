"""
Created on Mon Feb 26 15:03:06 2024

@author: ct347
"""

import matplotlib.pyplot as plt
import numpy as np

# 与之前相同的参数和BSDE求解过程

T = 1.0
N = 10
dt = T / N
a = 0.5
b = 0.2
xi = 1.0

Y = np.zeros((N + 1, 1000))  # 用于存储多条路径
entropies = np.zeros(N + 1)

for j in range(1000):  # 模拟1000条路径
    W = np.zeros(N + 1)
    dW = np.sqrt(dt) * np.random.randn(N)
    for i in range(1, N + 1):
        W[i] = W[i - 1] + dW[i - 1]

    Y[N, j] = xi
    for i in range(N, 0, -1):
        Z = (Y[i, j] - Y[i - 1, j]) / dW[i - 1] if dW[i - 1] != 0 else 0
        Y[i - 1, j] = Y[i, j] - (a * Y[i, j] + b * Z) * dt + Z * dW[i - 1]


def calculate_matrix_entropy(P):
    # 只考虑非零元素
    non_zero_probs = P[P > 0]
    entropy = np.sum(non_zero_probs * np.log(non_zero_probs))
    return entropy


Entropy = [calculate_matrix_entropy(x) for x in Y]

# # 计算每个时间步的熵
# for i in range(N+1):
#     entropies[i] = calculate_entropy(Y[i, :])

# # 绘制熵曲线图
plt.plot(np.linspace(0, T, N + 1), Entropy)
plt.xlabel("Time t")
plt.ylabel("Entropy")
plt.title("Entropy of $Y_t$ Over Time")
plt.show()
