"""
Created on Mon Feb 26 12:06:45 2024

@author: ct347
"""

import matplotlib.pyplot as plt
import numpy as np


# 计算条件概率转移矩阵的熵
def calculate_matrix_entropy(P):
    # 只考虑非零元素
    non_zero_probs = P[P > 0]
    entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
    return entropy


def calculate_conditional_transition_matrix(P, a, b, n):
    # 预先计算所有需要的矩阵幂
    powers = [np.linalg.matrix_power(P, k) for k in range(n + 1)]

    x = []
    conditional_matrix = np.zeros_like(P)

    for i in range(1, n):  # 对于每一个可能的i值
        conditional_matrix = np.zeros_like(P)
        for u in range(len(P)):
            for v in range(len(P)):
                # 计算分子，即从a经过u,v到b的联合概率

                numerator = powers[i - 1][a, u] * P[u, v] * powers[n - i - 1][v, b]

                # 计算分母，即从a直接到b的总概率
                denominator = powers[i - 1][a, u] * powers[n - i][u, b]

                if denominator > 0:
                    conditional_matrix[u, v] = numerator / denominator  # 累加对于每个i的贡献
        x.append(conditional_matrix)

    return x


# 绘制熵折线图的函数
def plot_entropy_curves(P, a, b, n):
    x = calculate_conditional_transition_matrix(P, a, b, n)
    t = [calculate_matrix_entropy(i) for i in x]

    plt.plot(t, label="Entropy")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Entropy Curve over Time")
    plt.legend()
    plt.show()


# 示例：使用马尔可夫概率转移矩阵P，起点a，终点b和步数n
P_example = np.array(
    [
        [0.4, 0.3, 0.3, 0.0, 0.0],
        [0.3, 0.4, 0.3, 0.0, 0.0],
        [0.0, 0.3, 0.4, 0.3, 0.0],
        [0.0, 0.0, 0.3, 0.4, 0.3],
        [0.0, 0.0, 0.3, 0.3, 0.4],
    ]
)
a_example = 0  # 起点
b_example = 2  # 终点
n_example = 5  # 步数

# 绘制熵折线图
plot_entropy_curves(P_example, a_example, b_example, n_example)
