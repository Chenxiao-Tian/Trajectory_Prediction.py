import random

import matplotlib.pyplot as plt
import numpy as np


# 定义查找包括停留在内的所有路径的函数
def find_paths_including_stays(adj_matrix, start, end, T, allow_stay=True):
    all_paths = []

    def dfs(current_node, path, steps_remaining):
        if steps_remaining == 0 and current_node == end:
            all_paths.append(path.copy())
            return
        elif steps_remaining < 0:
            return

        for next_node in range(len(adj_matrix[current_node])):
            if adj_matrix[current_node][next_node] > 0 or (
                allow_stay and next_node == current_node
            ):
                path.append(next_node)
                dfs(next_node, path, steps_remaining - 1)
                path.pop()

    dfs(start, [start], T)
    return all_paths


# 定义为路径分配概率权重的函数
def assign_probability_weights(all_paths, N):
    if N > len(all_paths):
        N = len(all_paths)
    selected_paths = random.sample(all_paths, N)
    weights = np.random.rand(N)
    weights /= weights.sum()  # 归一化权重
    return list(zip(selected_paths, weights))


# 定义计算单步转移矩阵的熵的函数
def calculate_entropy(matrix):
    entropy = 0
    for row in matrix:
        row = row[row > 0]  # 只考虑非零元素
        if row.size > 0:
            entropy += -np.sum(row * np.log(row))
    return entropy


# 定义计算条件概率转移矩阵和熵的函数
def calculate_transition_matrix_and_entropy(paths_with_probs, num_nodes, end, step, T):
    transition_matrix = np.zeros((num_nodes, num_nodes))
    denom = np.zeros(num_nodes)

    for path, weight in paths_with_probs:
        if step < T - 1:  # 非最后一步
            transition_matrix[path[step], path[step + 1]] += weight
            denom[path[step]] += weight
        elif step == T - 1:  # 最后一步
            if path[step] == end:  # 只考虑在终点的路径
                transition_matrix[path[step], end] += weight
                denom[path[step]] += weight

    # 归一化转移矩阵以获得条件概率
    for i in range(num_nodes):
        if denom[i] > 0:
            transition_matrix[i, :] /= denom[i]
        elif i == end:  # 最后一步终点到终点的转移概率为1
            transition_matrix[i, i] = 1

    # 计算熵
    entropy = calculate_entropy(transition_matrix)
    return entropy


# 定义绘制熵折线图的函数
def plot_entropy_curves(adj_matrix, start, end, T, N_values):
    num_nodes = len(adj_matrix)
    for N in N_values:
        paths = find_paths_including_stays(adj_matrix, start, end, T)
        path_weights = assign_probability_weights(paths, N)
        entropies = []
        for t in range(T):
            entropy = calculate_transition_matrix_and_entropy(
                path_weights, num_nodes, end, t, T
            )
            entropies.append(entropy)
        plt.plot(range(1, T + 1), entropies, label=f"N={N}")

    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Entropy Curves for Different N")
    plt.legend()
    plt.show()


# 定义绘制熵面积折线图的函数
def plot_entropy_area_curve(adj_matrix, start, end, T, N_values):
    num_nodes = len(adj_matrix)
    areas = []
    for N in N_values:
        paths = find_paths_including_stays(adj_matrix, start, end, T)
        path_weights = assign_probability_weights(paths, N)
        entropies = []
        for t in range(T):
            entropy = calculate_transition_matrix_and_entropy(
                path_weights, num_nodes, end, t, T
            )
            entropies.append(entropy)
        area = np.trapz(entropies)
        areas.append(area)

    plt.plot(N_values, areas)
    plt.xlabel("Number of Paths (N)")
    plt.ylabel("Area Under Entropy Curve")
    plt.title("Area Under Entropy Curve vs. N")
    plt.show()


# 示例：使用3x3网格邻接矩阵、起点、终点和步数T
adj_matrix_example = [
    [1, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 1],
]
start_example = 0
end_example = 8
T_example = 7
N_values_example = [
    1,
    2,
    3,
    4,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    100,
    200,
    300,
    400,
    500,
    600,
    630,
]  # 指定几个N值以绘制熵折线图和熵面积折线图

# 绘制熵折线图和熵面积折线图
plot_entropy_curves(
    adj_matrix_example, start_example, end_example, T_example, N_values_example
)
plot_entropy_area_curve(
    adj_matrix_example, start_example, end_example, T_example, N_values_example
)
