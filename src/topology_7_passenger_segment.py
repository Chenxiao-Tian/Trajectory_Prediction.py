# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:03:47 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 20:06:17 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.spatial import KDTree

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

def calculate_parameters(data):
    increments = np.diff(data[:, :2], axis=0)
    sigma = np.std(increments, axis=0) / np.sqrt(dt)
    mu = np.mean(increments, axis=0) / dt
    return sigma, mu

def segment_based_simulation(data, segments, sigma, mu):
    predicted_paths = np.zeros((100, len(data), 2))
    for j in range(100):
        for seg in segments:
            start_idx = seg[0]
            end_idx = seg[-1]
            x0 = data[start_idx, :2]
            xT = data[end_idx, :2]
            for i in range(start_idx, end_idx + 1):
                if i == start_idx:
                    predicted_paths[j, i] = x0
                else:
                    remaining_time = (end_idx - i) * dt
                    drift = mu + (xT - predicted_paths[j, i - 1]) / remaining_time
                    diffusion = sigma * np.sqrt(dt) * np.random.normal(size=2)
                    predicted_paths[j, i] = predicted_paths[j, i - 1] + drift * dt + diffusion
    return predicted_paths

def simulate_brownian_motion(data, sigma, mu):
    predicted_paths = np.zeros((100, len(data), 2))
    for j in range(100):
        predicted_paths[j, 0] = data[0, :2]
        for i in range(1, len(data)):
            predicted_paths[j, i] = predicted_paths[j, i - 1] + mu * dt + sigma * np.sqrt(dt) * np.random.normal(size=2)
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

# Main script
file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
data = load_data(file_path)
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

predicted_paths_bridge = segment_based_simulation(data, segments, sigma_estimated, mu_estimated)
predicted_paths_brownian = simulate_brownian_motion(data, sigma_estimated, mu_estimated)

G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

snapped_true_path = snap_to_nearest_road(G, data[:, :2])
snapped_predicted_paths_bridge = np.zeros_like(predicted_paths_bridge)
for j in range(predicted_paths_bridge.shape[0]):
    snapped_predicted_paths_bridge[j] = snap_to_nearest_road(G, predicted_paths_bridge[j])

aee_bridge = calculate_aee_per_step(snapped_true_path, snapped_predicted_paths_bridge)
aee_brownian = calculate_aee_per_step(snapped_true_path, predicted_paths_brownian)

# Plot simulated paths
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor='w', show=False, close=False)

# Plot all Brownian Motion paths, extending beyond the map boundaries
for j in range(100):
    ax.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)

# Plot true path and snapped segmented Brownian Bridge paths
for j in range(10):  # Reduced the number of plotted snapped paths to 10 for clarity
    ax.plot(snapped_predicted_paths_bridge[j, :, 1], snapped_predicted_paths_bridge[j, :, 0], color='red', alpha=0.5, label='Segmented Brownian Bridge' if j == 0 else "")
ax.plot(snapped_true_path[:, 1], snapped_true_path[:, 0], label='True Path', color='blue', linewidth=2)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Simulated Paths using Segmented Brownian Bridge and Brownian Motion with Estimated Parameters')
ax.legend()
plt.show()

# Plot AEE per step
plt.figure(figsize=(10, 6))
plt.plot(t, aee_bridge, label='AEE (Segmented Brownian Bridge)', color='red')
plt.plot(t, aee_brownian, label='AEE (Brownian Motion)', color='green')
plt.xlabel('Time')
plt.ylabel('AEE')
plt.title('AEE per Step for Segmented Brownian Bridge and Brownian Motion')
plt.legend()
plt.show()
