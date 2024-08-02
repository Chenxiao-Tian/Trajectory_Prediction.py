Created on Thu Mar 14 12:34:29 2024

@author: ct347
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:20:14 2024

@author: ct347
"""

import osmnx as ox
import matplotlib.pyplot as plt

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
for _ in range(101):
    # 使用A*算法计算路径
    route = nx.astar_path(G, origin_node, destination_node, heuristic=lambda x, y: heuristic(x, y, 1000), weight='length')
    routes.append(route)
  

# 绘制地图背景
fig, ax = ox.plot_graph(G, bgcolor='k', edge_linewidth=0.5, node_size=0, 
                        show=False, close=False)

# 确保所有路径都是有效的
valid_routes = [route for route in routes if route]

# 如果有有效路径，则一次性绘制所有路径上的边
if valid_routes:
    ox.plot_graph_routes(G, valid_routes, route_colors='red', route_linewidth=2, route_alpha=0.8, orig_dest_size=0, ax=ax)

# 绘制所有路径上的节点
all_nodes = set(node for route in valid_routes for node in route)
nc = ['red' if node in all_nodes else 'none' for node in G.nodes()]
ns = [50 if node in all_nodes else 0 for node in G.nodes()]
ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_zorder=3, edge_linewidth=0, bgcolor='k')

plt.show()
