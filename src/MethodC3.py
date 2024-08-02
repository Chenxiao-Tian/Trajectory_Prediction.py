# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:41:01 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
from scipy.spatial.distance import euclidean

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

def calculate_parameters(data):
    increments = np.diff(data[:, :2], axis=0)
    sigma = np.std(increments, axis=0) / np.sqrt(dt)
    mu = np.mean(increments, axis=0) / dt
    return sigma, mu

def find_similar_segments(start, end, passenger_status, all_segments):
    similar_segments = []
    for seg in all_segments:
        if seg[0, 2] == passenger_status:  # 仅考虑相同状态的路径段
            distance_start = euclidean(start, seg[0, :2])
            distance_end = euclidean(end, seg[-1, :2])
            total_distance = distance_start + distance_end
            similar_segments.append((total_distance, seg))
    similar_segments.sort(key=lambda x: x[0])
    return similar_segments[:1]  # 选取最相似的1个段

def segment_based_simulation_with_history(data, segments, sigma, mu, all_segments):
    predicted_paths = np.zeros((100, len(data), 2))
    for j in range(100):
        for seg in segments:
            start_idx = seg[0]
            end_idx = seg[-1]
            x0 = data[start_idx, :2]
            xT = data[end_idx, :2]
            
            similar_segment = find_similar_segments(x0, xT, data[start_idx, 2], all_segments)
            if similar_segment:
                similar_path = similar_segment[0][1][:, :2]
                for i in range(1, len(similar_path)):
                    if i == 1:
                        predicted_paths[j, start_idx] = x0
                    else:
                        predicted_paths[j, start_idx + i - 1] = similar_path[i]
            else:
                for i in range(start_idx, end_idx + 1):
                    if i == start_idx:
                        predicted_paths[j, i] = x0
                    else:
                        remaining_time = (end_idx - i) * dt
                        drift = mu + (xT - predicted_paths[j, i - 1]) / remaining_time
                        diffusion = sigma * np.sqrt(dt) * np.random.normal(size=2)
                        predicted_paths[j, i] = predicted_paths[j, i - 1] + drift * dt + diffusion
                        if not np.all(np.isfinite(predicted_paths[j, i])):  # 确保没有inf或nan
                            predicted_paths[j, i] = predicted_paths[j, i - 1]
    return predicted_paths

def simulate_brownian_motion(data, sigma, mu):
    predicted_paths = np.zeros((100, len(data), 2))
    for j in range(100):
        predicted_paths[j, 0] = data[0, :2]
        for i in range(1, len(data)):
            predicted_paths[j, i] = predicted_paths[j, i - 1] + mu * dt + sigma * np.sqrt(dt) * np.random.normal(size=2)
            if not np.all(np.isfinite(predicted_paths[j, i])):  # 确保没有inf或nan
                predicted_paths[j, i] = predicted_paths[j, i - 1]
    return predicted_paths

def calculate_aee_per_step(true_path, predicted_paths):
    num_steps = true_path.shape[0]
    aee_per_step = np.zeros(num_steps)
    for i in range(num_steps):
        errors = np.sqrt(np.sum((true_path[i, :2] - predicted_paths[:, i, :]) ** 2, axis=1))
        aee_per_step[i] = np.mean(errors)
    return aee_per_step

def snap_to_nearest_road(G, path):
    A = np.array(path[:, 1])
    B = np.array(path[:, 0])
    valid_indices = np.isfinite(A) & np.isfinite(B)
    A_valid = A[valid_indices]
    B_valid = B[valid_indices]
    coordinates = np.column_stack((A_valid, B_valid))
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    coords = np.array(list(zip(gdf_nodes['x'], gdf_nodes['y'])))
    tree = KDTree(coords)
    distances, indices = tree.query(coordinates)
    nearest_coords = coords[indices]
    snapped_path = np.zeros_like(path)
    snapped_path[valid_indices] = np.column_stack((nearest_coords[:, 1], nearest_coords[:, 0]))
    snapped_path[~valid_indices] = path[~valid_indices]
    return snapped_path

def calculate_segment_lengths(path, segments):
    segment_lengths = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment = path[start_idx:end_idx+1]
        valid_segment = segment[np.all(np.isfinite(segment), axis=1)]  # 确保没有inf或nan
        segment_length = np.sum(np.sqrt(np.sum(np.diff(valid_segment, axis=0)**2, axis=1)))
        if np.isfinite(segment_length):
            segment_lengths.append(segment_length)
    return segment_lengths

# Main script
taxi_files = [
    'taxi1.txt',
    'taxi2.txt',
    'taxi3.txt',
    'taxi4.txt',
    'taxi5.txt',
    'taxi6.txt'
]
taxi_data = [load_data(file) for file in taxi_files]
taxi7 = load_data('taxi7.txt')

data = taxi7
N = len(data) - 1
T = 1
dt = T / N
t = np.linspace(0, T, N + 1)
sigma_estimated, mu_estimated = calculate_parameters(data)

# Identify segments based on the 'Segment' column
segments = []
current_segment = []
current_value = data[0, 2]

for idx, row in enumerate(data):
    if row[2] == current_value:
        current_segment.append(idx)
    else:
        segments.append(current_segment)
        current_segment = [idx]
        current_value = row[2]
segments.append(current_segment)

# Split all taxi data into segments
all_segments = []
for taxi in taxi_data:
    current_segment = []
    current_value = taxi[0, 2]
    for idx, row in enumerate(taxi):
        if row[2] == current_value:
            current_segment.append(row)
        else:
            all_segments.append(np.array(current_segment))
            current_segment = [row]
            current_value = row[2]
    all_segments.append(np.array(current_segment))

predicted_paths_bridge_with_history = segment_based_simulation_with_history(data, segments, sigma_estimated, mu_estimated, all_segments)
predicted_paths_brownian = simulate_brownian_motion(data, sigma_estimated, mu_estimated)

G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

snapped_true_path = snap_to_nearest_road(G, data[:, :2])
snapped_predicted_paths_bridge_with_history = np.zeros_like(predicted_paths_bridge_with_history)
for j in range(predicted_paths_bridge_with_history.shape[0]):
    snapped_predicted_paths_bridge_with_history[j] = snap_to_nearest_road(G, predicted_paths_bridge_with_history[j])

aee_bridge_with_history = calculate_aee_per_step(snapped_true_path, snapped_predicted_paths_bridge_with_history)
aee_brownian = calculate_aee_per_step(snapped_true_path, predicted_paths_brownian)

# Calculate segment lengths
true_segment_lengths = calculate_segment_lengths(snapped_true_path, segments)  # Use original data for true segment lengths
predicted_segment_lengths_with_history = []
for j in range(predicted_paths_bridge_with_history.shape[0]):
    predicted_segment_lengths_with_history.extend(calculate_segment_lengths(snapped_predicted_paths_bridge_with_history[j], segments))

# 更新画图代码
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Path Segment Lengths', density=True)
plt.hist(predicted_segment_lengths_with_history, bins=30, alpha=0.5, label='Predicted Path Segment Lengths (with History)', density=True)
plt.xlabel('Segment Length')
plt.ylabel('Proportion')
plt.title('Distribution of Segment Lengths')
plt.legend()
plt.show()

# Kolmogorov-Smirnov检验
ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths_with_history)
print(f"KS Statistic: {ks_stat}, P-value: {p_value}")
# KS检验结论
if p_value < 0.05:
    print("The null hypothesis that the two distributions are the same is rejected at the 5% significance level.")
else:
    print("The null hypothesis that the two distributions are the same cannot be rejected at the 5% significance level.")
# Plot simulated paths
fig, ax = plt.subplots(figsize=(30, 30))
ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor='w', show=False, close=False)

# Plot all Brownian Motion paths, extending beyond the map boundaries
for j in range(100):
    ax.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)

# Plot true path and snapped segmented Brownian Bridge paths with history
for j in range(1):  # Reduced the number of plotted snapped paths to 10 for clarity
    ax.plot(snapped_predicted_paths_bridge_with_history[j, :, 1], snapped_predicted_paths_bridge_with_history[j, :, 0], color='red', alpha=0.5, label='Segmented Brownian Bridge with History' if j == 0 else "")
ax.plot(snapped_true_path[:, 1], snapped_true_path[:, 0], label='True Path', color='blue', linewidth=2)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Simulated Paths using Segmented Brownian Bridge and Brownian Motion with Estimated Parameters')
ax.legend()
plt.show()

# Plot AEE per step
plt.figure(figsize=(10, 6))
plt.plot(t, aee_bridge_with_history, label='AEE (Segmented Brownian Bridge with History)', color='red')
plt.plot(t, aee_brownian, label='AEE (Brownian Motion)', color='green')
plt.xlabel('Time')
plt.ylabel('AEE')
plt.title('AEE per Step for Segmented Brownian Bridge and Brownian Motion')
plt.legend()
plt.show()