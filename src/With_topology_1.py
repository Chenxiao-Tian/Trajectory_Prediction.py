# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:15:51 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:24:09 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Zero', 'Timestamp'])
    return df[['Latitude', 'Longitude']].values

def calculate_parameters(data):
    increments = np.diff(data, axis=0)
    sigma = np.std(increments, axis=0) / np.sqrt(dt)
    mu = np.mean(increments, axis=0) / dt
    return sigma, mu

def segment_based_simulation(data, segment_points, sigma, mu):
    predicted_paths = np.zeros((100, N+1, 2))
    for j in range(100):
        for seg in range(len(segment_points)-1):
            start_idx = segment_points[seg]
            end_idx = segment_points[seg + 1]
            x0 = data[start_idx]
            xT = data[end_idx]
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
    predicted_paths = np.zeros((100, N + 1, 2))
    for j in range(100):
        predicted_paths[j, 0] = data[0]
        for i in range(1, N + 1):
            predicted_paths[j, i] = predicted_paths[j, i - 1] + mu * dt + sigma * np.sqrt(dt) * np.random.normal(size=2)
    return predicted_paths

def calculate_aee_per_step(true_path, predicted_paths):
    num_steps = true_path.shape[0]
    aee_per_step = np.zeros(num_steps)
    for i in range(num_steps):
        errors = np.sqrt(np.sum((true_path[i] - predicted_paths[:, i, :]) ** 2, axis=1))
        aee_per_step[i] = np.mean(errors)
    return aee_per_step

file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
data = load_data(file_path)
N = len(data) - 1
T = 1
dt = T / N
t = np.linspace(0, T, N + 1)
sigma_estimated, mu_estimated = calculate_parameters(data)
segment_points = np.arange(0, N, 50)
if segment_points[-1] != N:
    segment_points = np.append(segment_points, N)

predicted_paths_bridge = segment_based_simulation(data, segment_points, sigma_estimated, mu_estimated)
predicted_paths_brownian = simulate_brownian_motion(data, sigma_estimated, mu_estimated)

# Plot the simulated paths
plt.figure(figsize=(10, 6))

for j in range(100):
    print(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0])
    plt.plot(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0], color='red', alpha=0.1)
    plt.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)
plt.plot(data[:, 1], data[:, 0], label='True Path', color='blue', linewidth=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Simulated Paths using Segmented Brownian Bridge and Brownian Motion with Estimated Parameters')
plt.legend(['Segmented Brownian Bridge', 'Brownian Motion', 'True Path'])
plt.show()

aee_bridge = calculate_aee_per_step(data, predicted_paths_bridge)
aee_brownian = calculate_aee_per_step(data, predicted_paths_brownian)

# Plot AEE per step
plt.figure(figsize=(10, 6))
plt.plot(t, aee_bridge, label='AEE (Segmented Brownian Bridge)', color='red')
plt.plot(t, aee_brownian, label='AEE (Brownian Motion)', color='green')
plt.xlabel('Time')
plt.ylabel('AEE')
plt.title('AEE per Step for Segmented Brownian Bridge and Brownian Motion')
plt.legend()
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:27:40 2024

@author: 92585
"""

import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

for j in range(1):
    #print(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0])
    # Sample data for A and B arrays
    A=np.array(predicted_paths_bridge[j, :, 1])
    B=np.array(predicted_paths_bridge[j, :, 0])
# A = np.array([-122.39769, -122.40066221, -122.40237026, -122.43191135, -122.43221025, -np.inf])
# B = np.array([37.7638, 37.7633858, 37.77482195, 37.76238295, 37.77735981, -np.inf])

    # Remove -inf values
    valid_indices = np.isfinite(A) & np.isfinite(B)
    A = A[valid_indices]
    B = B[valid_indices]
    
    # Combine A and B into a list of coordinate pairs
    coordinates = np.column_stack((A, B))
    
    # Download the road network
    G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')
    
    # Get node coordinates from the graph
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    coords = np.array(list(zip(gdf_nodes['x'], gdf_nodes['y'])))
    
    # Create KDTree for fast nearest-neighbor search
    tree = KDTree(coords)
    
    # Find the nearest nodes
    distances, indices = tree.query(coordinates)
    nearest_coords = coords[indices]

    plt.plot(nearest_coords[:, 0], nearest_coords[:, 1], color='red', alpha=0.1)
    # predicted_paths_bridge[j, :, 1]=nearest_coords[:, 0]
    # predicted_paths_bridge[j, :, 0]=nearest_coords[:, 1]
    plt.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)
plt.plot(data[:, 1], data[:, 0], label='True Path', color='blue', linewidth=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Simulated Paths using Segmented Brownian Bridge and Brownian Motion with Estimated Parameters')
plt.legend(['Segmented Brownian Bridge', 'Brownian Motion', 'True Path'])
plt.show()

aee_bridge = calculate_aee_per_step(data, predicted_paths_bridge)
aee_brownian = calculate_aee_per_step(data, predicted_paths_brownian)

# Plot AEE per step
plt.figure(figsize=(10, 6))
plt.plot(t, aee_bridge, label='AEE (Segmented Brownian Bridge)', color='red')
plt.plot(t, aee_brownian, label='AEE (Brownian Motion)', color='green')
plt.xlabel('Time')
plt.ylabel('AEE')
plt.title('AEE per Step for Segmented Brownian Bridge and Brownian Motion')
plt.legend()
plt.show()
# # Print the converted coordinates
# print("Converted Coordinates (lat, lon):")
# for coord in nearest_coords:
#     print(coord)

# # Plot the results
# fig, ax = plt.subplots(figsize=(10, 10))
# ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor='k', show=False, close=False)

# # Plot original coordinates
# ax.scatter(A, B, c='red', s=50, label='Original Coordinates', alpha=0.7)

# # Plot nearest nodes
# ax.scatter(nearest_coords[:, 0], nearest_coords[:, 1], c='blue', s=50, label='Nearest Nodes', alpha=0.7)

# ax.legend()
# plt.show()

