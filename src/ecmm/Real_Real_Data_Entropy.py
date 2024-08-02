# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:03:24 2024

@author: ct347
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import osmnx as ox
import networkx as nx
import geopy.distance

def read_paths(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')

    paths = []
    for block in data:
        lines = block.split('\n')
        path = []
        for line in lines:
            try:
                # 尝试提取并转换经纬度坐标
                lat, lon = map(float, line.split(';')[:2])
                path.append((lat, lon))
            except ValueError:
                # 如果转换失败，则跳过当前行
                print(f"Skipping invalid line: {line}")
                continue
        paths.append(path)
    return paths

def coords_to_osm_nodes(G, paths):
    # 将每个经纬度坐标转换为最近的OSM节点ID
    node_paths = []
    for path in paths:
        node_path = [ox.get_nearest_node(G, (lat, lon), method='euclidean') for lat, lon in path]
        node_paths.append(node_path)
    return node_paths

def add_uniform_start_end(paths):
    start_points = [path[0] for path in paths]
    end_points = [path[-1] for path in paths]
    avg_start = tuple(np.mean(start_points, axis=0))
    avg_end = tuple(np.mean(end_points, axis=0))

    uniform_paths = [[avg_start] + path + [avg_end] for path in paths]
    return uniform_paths
    
def process_and_sort_strict_weights(lengths, routes):
    max_nodes = max(len(route) for route in routes)
    padded_routes = [route + [route[-1]] * (max_nodes - len(route)) for route in routes]
    lengths_routes_pairs = sorted(zip(lengths, padded_routes), key=lambda x: x[0], reverse=True)
    sorted_lengths, sorted_padded_routes = zip(*lengths_routes_pairs)

    inverse_lengths = [1 / len(sorted_lengths) for length in sorted_lengths]
    total_inverse = sum(inverse_lengths)
    weights_strict = [inv / total_inverse for inv in inverse_lengths]

    return list(sorted_padded_routes), list(sorted_lengths), weights_strict

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

def main(file_path):
    # 加载街道网络
    G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

    paths = read_paths(file_path)
    uniform_paths = add_uniform_start_end(paths)
    
    # 将经纬度坐标转换为OSM节点ID
    node_paths = coords_to_osm_nodes(G, uniform_paths)
    
    # 这里假设每条路径的长度等于它包含的节点数
    lengths = [len(path) for path in node_paths]
    sorted_routes, sorted_lengths, weights = process_and_sort_strict_weights(lengths, node_paths)

    max_length = max(len(route) for route in sorted_routes)
    entropies = [compute_entropy_for_ith(sorted_routes, weights, ith) for ith in range(max_length - 1)]

    plt.plot(range(max_length - 1), entropies)
    plt.xlabel('Node Sequence Index (ith)')
    plt.ylabel('Entropy')
    plt.title('Entropy Variation with Node Sequence Index')
    plt.show()
    return node_paths

file_path = 'max_cluster_details.txt'
A=main(file_path)
print(A)
