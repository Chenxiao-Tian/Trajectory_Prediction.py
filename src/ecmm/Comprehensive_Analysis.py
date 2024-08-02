# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:15:20 2024

@author: ct347
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:19:17 2024

@author: ct347
"""

def process_taxi_data(input_file, output_file_zero, output_file_one):
    """Process taxi data to separate into two files based on status changes."""
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Open output files for writing
    file_zero = open(output_file_zero, 'w')
    file_one = open(output_file_one, 'w')
    
    # Initialize variables to track the status and the file to write to
    last_status = None
    current_file = None

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # Skip any malformed lines
        
        current_status = parts[2]
        
        if current_status != last_status:
            # Status has changed, determine the correct file and add a new line if needed
            if current_status == '0':
                current_file = file_zero
            elif current_status == '1':
                current_file = file_one
            
            if last_status is not None:  # Not the first line, so add a separation line
                current_file.write('\n')

        # Write the current line to the appropriate file
        current_file.write(line)
        
        # Update the last status
        last_status = current_status
    
    # Close the files
    file_zero.close()
    file_one.close()

# # Define the file paths
# input_file_path = 'taxi1.txt'  # Path to the original taxi data file
# output_file_path_zero = 'status_0.txt'  # Output path for status 0 data
# output_file_path_one = 'status_1.txt'  # Output path for status 1 data

# Process the data
#process_taxi_data(input_file_path, output_file_path_zero, output_file_path_one)
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:35:53 2024

@author: ct347
"""

import numpy as np
from sklearn.cluster import DBSCAN
from haversine import haversine

def read_data_from_file(file_path):
    """Read data from file."""
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def parse_data(data):
    """Parse blocks of data where each block represents a path with start and end coordinates."""
    blocks = data.strip().split('\n\n')
    paths = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        try:
            start_parts = lines[0].split()
            end_parts = lines[-1].split()
            start_coords = tuple(map(float, start_parts[:2]))
            end_coords = tuple(map(float, end_parts[:2]))
            paths.append({'coords': (start_coords, end_coords), 'block': block})
        except ValueError:
            continue
    return paths

def cluster_paths(paths, eps_km):
    """Cluster paths based on the maximum distance between their start and end coordinates."""
    n = len(paths)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            start_distance = haversine(paths[i]['coords'][0], paths[j]['coords'][0])
            end_distance = haversine(paths[i]['coords'][1], paths[j]['coords'][1])
            composite_distance = max(start_distance, end_distance)  # Using max distance for clustering
            distance_matrix[i, j] = distance_matrix[j, i] = composite_distance

    db = DBSCAN(eps=eps_km, min_samples=1, metric='precomputed').fit(distance_matrix)
    labels = db.labels_

    clusters = {}
    for i in range(n):
        label = labels[i]
        if label in clusters:
            clusters[label].append(paths[i])
        else:
            clusters[label] = [paths[i]]

    return clusters

def main1(file_path,x,l):
    data = read_data_from_file(file_path)
    paths = parse_data(data)
    clusters = cluster_paths(paths, eps_km=l)

    max_cluster_size = max(len(cluster) for cluster in clusters.values())
    max_cluster = max(clusters.items(), key=lambda x: len(x[1]))

    # 输出到文件
    with open('details'+str(x)+'.txt', 'w', encoding='utf-8') as f:
        f.write(f'最大聚类包含路径数量: {max_cluster_size}\n\n')
        for path in max_cluster[1]:
            f.write(f'{path["block"]}\n\n')
    print(f'最大聚类包含路径数量: {max_cluster_size}')
    print('最大聚类中的路径节点:')
    print(f"最大聚类的详细信息已保存到文件:"+x)

# file_path = 'status_0.txt'  # or 'status_1.txt'
# file_path2 = 'status_1.txt'  # or 'status_1.txt'
# main(file_path,"0",0.4)
# main(file_path2,"1",0.4)
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
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split('\n\n')  # 根据空行分割不同的路径块

    paths = []
    for block in data:
        lines = block.split('\n')
        path = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # 跳过格式错误的行
            try:
                lat, lon = float(parts[0]), float(parts[1])
                path.append((lat, lon))
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue
        if path:
            paths.append(path)
    return paths


def coords_to_osm_nodes(G, paths):
    # 将每个经纬度坐标转换为最近的OSM节点ID
    node_paths = []
    for path in paths:
        node_path = [ox.nearest_nodes(G, lon, lat) for lat, lon in path]  # 注意lat和lon的顺序
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

def main(file_path,x):
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
    plt.title('Entropy Variation with Path Node Sequence Index:'+"x")
    plt.show()
    return entropies

def plot_multiple_entropies(entropies_list, labels, title):
    """
    Plots multiple entropy series on the same graph with different colors.

    Parameters:
    - entropies_list: List of lists, where each sublist contains entropy values for a series.
    - labels: List of strings, labels for each entropy series for the legend.
    - title: String, title of the plot.
    """
    # Determine the maximum length of any series
    max_length = max(len(entropies) for entropies in entropies_list)
    
    # Extend all series to the maximum length
    extended_entropies_list = [entropies + [0] * (max_length - len(entropies)) for entropies in entropies_list]
    
    # Set up the plot
    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(extended_entropies_list)))  # Generate color map

    for entropies, label, color in zip(extended_entropies_list, labels, colors):
        plt.plot(range(max_length), entropies, label=label, color=color, marker='o', linestyle='-')

    plt.xlabel('Node Sequence Index (ith)')
    plt.ylabel('Entropy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
B=['taxi1.txt','taxi2.txt','taxi3.txt','taxi4.txt','taxi5.txt','taxi6.txt','taxi7.txt']
E1=[]
E2=[]
for x in B:
    output_file_path_zero = 'status_0.txt'  # Output path for status 0 data
    output_file_path_one = 'status_1.txt'  # Output path for status 1 data
    input_file_path=x
    # Process the data
    process_taxi_data(input_file_path, output_file_path_zero, output_file_path_one)
    file_path = 'status_0.txt'  # or 'status_1.txt'
    file_path2 = 'status_1.txt'  # or 'status_1.txt'
    main1(file_path,"0",0.4)
    main1(file_path2,"1",0.4)
    file_path = 'details0.txt'
    A=main(file_path,"without passenger data")
    E1.append(A)
    file_path = 'details1.txt'
    A=main(file_path,"With passenger data")
    E2.append(A)
plot_multiple_entropies(E1,['taxi1','taxi2','taxi3','taxi4','taxi5','taxi6','taxi7'], title="Entropy Variation with Path Node Sequence Index:Without Passenger Case")
plot_multiple_entropies(E2,['taxi1','taxi2','taxi3','taxi4','taxi5','taxi6','taxi7'], title="Entropy Variation with Path Node Sequence Index:With Passenger Case")
# file_path = 'details0.txt'
# A=main(file_path,"without passenger data")
# file_path = 'details1.txt'
# A=main(file_path,"With passenger data")
