# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:03:51 2024

@author: ct347
"""

import osmnx as ox
import networkx as nx
import geopy.distance
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 基于正态分布增加路径长度的启发式函数
def heuristic(node1, node2, G, scale=1000):
    coords_1 = (G.nodes[node1]['y'], G.nodes[node1]['x'])
    coords_2 = (G.nodes[node2]['y'], G.nodes[node2]['x'])
    base_distance = geopy.distance.geodesic(coords_1, coords_2).meters
    random_factor = norm.rvs(loc=1, scale=scale)
    increased_distance = base_distance * random_factor
    return max(base_distance, increased_distance)

# 将路径和长度排序并计算权重的函数
def process_and_sort_strict_weights(lengths, routes):
    max_nodes = max(len(route) for route in routes)
    padded_routes = [route + [route[-1]] * (max_nodes - len(route)) for route in routes]
    lengths_routes_pairs = sorted(zip(lengths, padded_routes), key=lambda x: x[0], reverse=True)
    sorted_lengths, sorted_padded_routes = zip(*lengths_routes_pairs)
    inverse_lengths = [1 / length for length in sorted_lengths]
    total_inverse = sum(inverse_lengths)
    weights_strict = [inv / total_inverse for inv in inverse_lengths]
    return list(sorted_padded_routes), list(sorted_lengths), weights_strict

# 计算给定序数的熵值的函数
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

# 获取旧金山地区的道路网络
G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

# 定义起点和终点
origin_node = 9226304189  # 旧金山机场
destination_node = 65365766  # 伯克利

area_results = []

for num_paths in range(1, 41):
    lengths = []
    routes = []
    for _ in range(num_paths):
        route = nx.astar_path(G, origin_node, destination_node, heuristic=lambda x, y: heuristic(x, y, G, 1000), weight='length')
        routes.append(route)
        length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
        lengths.append(length)
    sorted_routes, sorted_lengths, weights = process_and_sort_strict_weights(lengths, routes)
    max_length = max(len(route) for route in sorted_routes)
    entropies = [compute_entropy_for_ith(sorted_routes, weights, ith) for ith in range(max_length - 1)]
    area = sum(entropies)  # 计算熵曲线下围成的面积
    area_results.append(area)
    
plt.plot(range(1, 41), area_results)
plt.xlabel('Number of Paths')
plt.ylabel('Area under Entropy Curve')
plt.title('Area under Entropy Curve vs. Number of Paths')
plt.show()

