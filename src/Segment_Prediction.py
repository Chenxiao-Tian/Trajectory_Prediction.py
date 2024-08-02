# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:29:12 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Zero', 'Timestamp'])
data = df[['Latitude', 'Longitude']].values

# Parameters
T = 1
N = len(data) - 1
dt = T / N
t = np.linspace(0, T, N)

# Calculate increments and estimate parameters
increments = np.diff(data, axis=0)
sigma_estimated = np.std(increments, axis=0) / np.sqrt(dt)
mu_estimated = np.mean(increments, axis=0) / dt

# Define segment points (e.g., every 50 points is a segment point)
segment_points = np.arange(0, N, 50)
if segment_points[-1] != N:
    segment_points = np.append(segment_points, N)  # Ensure the last point is included

# Simulation storage
predicted_paths_bridge = np.zeros((100, N+1, 2))
predicted_paths_brownian = np.zeros((100, N+1, 2))

# Segment-based simulation
for j in range(100):
    for seg in range(len(segment_points)-1):
        start_idx = segment_points[seg]
        end_idx = segment_points[seg + 1]
        x0 = data[start_idx]
        xT = data[end_idx]

        # Simulate each segment using Brownian Bridge
        for i in range(start_idx, end_idx + 1):
            if i == start_idx:
                predicted_paths_bridge[j, i] = x0
            else:
                elapsed_time = (i - start_idx) * dt
                remaining_time = (end_idx - i) * dt
                drift = mu_estimated + (xT - predicted_paths_bridge[j, i-1]) / remaining_time
                diffusion = sigma_estimated * np.sqrt(dt) * np.random.normal(size=2)
                predicted_paths_bridge[j, i] = predicted_paths_bridge[j, i-1] + drift * dt + diffusion

# Plot results
plt.figure(figsize=(10, 6))
for j in range(1):  # Only plot one simulation for clarity
    plt.plot(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0], 'r-', alpha=0.1)
plt.plot(data[:, 1], data[:, 0], 'b-', linewidth=2, label='True Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Segmented Brownian Bridge Simulation')
plt.legend()
plt.show()
