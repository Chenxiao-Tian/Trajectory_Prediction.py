# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 07:24:18 2024

@author: ct347
"""

import osmnx as ox
import networkx as nx
import geopy.distance
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def heuristic(node1, node2, scale=1000):
    # 基本距离
    coords_1 = (G.nodes[node1]['y'], G.nodes[node1]['x'])
    coords_2 = (G.nodes[node2]['y'], G.nodes[node2]['x'])
    base_distance = geopy.distance.geodesic(coords_1, coords_2).meters
    
    # 引入正态分布随机因子来增加路径长度
    random_factor = norm.rvs(loc=1, scale=scale)  # loc是均值，scale是标准差
    increased_distance = base_distance * random_factor
    return max(base_distance, increased_distance)  # 确保至少是基本距离

# 获取旧金山地区的道路网络
G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

# 定义起点和终点
origin_node = 9226304189  # 旧金山机场
destination_node = 65365766  # 伯克利

lengths = []
routes=[]
for _ in range(100):
    # 使用A*算法计算路径
    route = nx.astar_path(G, origin_node, destination_node, heuristic=lambda x, y: heuristic(x, y, 1000), weight='length')
    routes.append(route)
  
    length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
    lengths.append(length)
print(routes)
# # 打印路径长度
for length in lengths:
    print(f'Path length: {length:.2f} meters')
ox.plot_graph_route(G, routes[0], route_color='red', node_size=0, bgcolor='k')

# fig, ax = ox.plot_graph(G, bgcolor='k', edge_linewidth=0.5, node_size=0, show=True, close=False)

# for route in routes:
#     # 确保路径不为空
#     if route:
#         ox.plot_graph_route(G, route, route_color='red', route_linewidth=6, ax=ax, show=True)
#     else:
#         print("Empty route")

# plt.show()  # 确保调用plt.show()来显示图形
def process_and_sort_strict_weights(lengths, routes):
    # Find the maximum number of nodes among all routes
    max_nodes = max(len(route) for route in routes)

    # Pad routes to have the same number of nodes as the longest route
    padded_routes = [route + [route[-1]] * (max_nodes - len(route)) for route in routes]

    # Sort routes and lengths from longest to shortest based on original lengths
    lengths_routes_pairs = sorted(zip(lengths, padded_routes), key=lambda x: x[0], reverse=True)
    sorted_lengths, sorted_padded_routes = zip(*lengths_routes_pairs)

    # Calculate weights based on the inverse of sorted lengths, then normalize
    inverse_lengths = [1 / length for length in sorted_lengths]
    total_inverse = sum(inverse_lengths)
    weights_strict = [inv / total_inverse for inv in inverse_lengths]

    return list(sorted_padded_routes), list(sorted_lengths), weights_strict

# Use the provided lengths and routes for testing
sorted_routes, sorted_lengths, weights = process_and_sort_strict_weights(lengths, routes)


print(sorted_routes, sorted_lengths, weights)
import matplotlib.pyplot as plt
import numpy as np

def compute_entropy_for_ith(sorted_routes, weights, ith):
    Si = set(route[ith] for route in sorted_routes if ith < len(route))
    Si_next = set(route[ith + 1] for route in sorted_routes if ith + 1 < len(route))
    entropy = 0

    for X in Si:
        weight_X = sum(weights[idx] for idx, route in enumerate(sorted_routes) if route[ith] == X)
        for Y in Si_next:
            weight_XY = sum(weights[idx] for idx, route in enumerate(sorted_routes) if route[ith] == X and route[ith + 1] == Y)
            if weight_X > 0 and weight_XY > 0:
                P_XY = weight_XY / weight_X
                entropy -= P_XY * np.log(P_XY)

    return entropy

max_length = max(len(route) for route in sorted_routes)
entropies = [compute_entropy_for_ith(sorted_routes, weights, ith) for ith in range(max_length - 1)]

# 绘制熵随序数变化的折线图
plt.plot(range(max_length - 1), entropies)
plt.xlabel('Node Sequence Index (ith)')
plt.ylabel('Entropy')
plt.title('Entropy Variation with Node Sequence Index')
plt.show()







