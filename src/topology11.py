# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:18:22 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:27:38 2024

@author: 92585
"""

import osmnx as ox
import networkx as nx
import geopy.distance
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

def heuristic(node1, node2, scale=1000):
    coords_1 = (G.nodes[node1]['y'], G.nodes[node1]['x'])
    coords_2 = (G.nodes[node2]['y'], G.nodes[node2]['x'])
    base_distance = geopy.distance.geodesic(coords_1, coords_2).meters
    random_factor = norm.rvs(loc=1, scale=scale)
    increased_distance = base_distance * random_factor
    return max(base_distance, increased_distance)

def get_nearest_node(G, point):
    return ox.distance.nearest_nodes(G, point[1], point[0])

def predict_segment_routes(G, origin_node, destination_node, num_paths=10, num_predictions=100):
    lengths = []
    routes = []

    for _ in range(num_paths):
        try:
            route = nx.astar_path(G, origin_node, destination_node, heuristic=lambda x, y: heuristic(x, y, 1000), weight='length')
            routes.append(route)
            length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
            if length > 0:
                lengths.append(length)
        except nx.NetworkXNoPath:
            print(f"Path {_+1}: No path found between nodes {origin_node} and {destination_node}.")
            continue

    if not lengths:
        return []

    inverse_lengths = [1.0 / length for length in lengths]
    mean_inverse_length = np.mean(inverse_lengths)
    std_dev_inverse = np.std(inverse_lengths)

    weights = norm.pdf(inverse_lengths, loc=mean_inverse_length, scale=std_dev_inverse)
    normalized_weights = weights / np.sum(weights)

    predicted_routes = random.choices(routes, weights=normalized_weights, k=num_predictions)
    return predicted_routes

def read_and_parse_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        lat, lon, flag, _ = map(float, parts)
        data.append((lat, lon, int(flag)))
    
    return data

def find_segments(data):
    segments = []
    current_segment = []

    for i, (lat, lon, flag) in enumerate(data):
        if flag == 1 and current_segment:
            segments.append(current_segment)
            current_segment = []
        current_segment.append((lat, lon))
    
    if current_segment:
        segments.append(current_segment)
    
    return segments

# 读取和解析数据
data = read_and_parse_data('taxi2.txt')
segments = find_segments(data)

# 获取旧金山地区的道路网络
G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

# 获取每个段的起点和终点
waypoints = [(segment[0], segment[-1]) for segment in segments]

# 获取每个点的最近节点
nodes = [(get_nearest_node(G, start), get_nearest_node(G, end)) for start, end in waypoints]

# 检查每对节点是否在同一个连通子图中
for i in range(len(nodes)):
    if not nx.has_path(G, nodes[i][0], nodes[i][1]):
        raise nx.NetworkXNoPath(f"Node {nodes[i][0]} not reachable from {nodes[i][1]}.")

# 存储完整预测路径
complete_predicted_routes = []

for _ in range(10):  # 生成10条完整的预测路径
    full_route = []
    for origin_node, destination_node in nodes:
        segment_routes = predict_segment_routes(G, origin_node, destination_node, num_paths=10, num_predictions=1)
        if segment_routes:
            if full_route:
                full_route.extend(segment_routes[0][1:])
            else:
                full_route.extend(segment_routes[0])
    complete_predicted_routes.append(full_route)

# 绘制地图背景
fig, ax = ox.plot_graph(G, bgcolor='k', edge_linewidth=0.5, node_size=0, show=False, close=False)

# 绘制所有完整路径
for route in complete_predicted_routes:
    ox.plot_graph_route(G, route, route_color='red', route_linewidth=2, route_alpha=0.8, orig_dest_node_size=0, ax=ax)

plt.show()
