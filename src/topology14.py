# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:26:27 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:40:00 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:36:47 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.spatial import KDTree
from scipy.stats import ks_2samp

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment', 'Timestamp']].values

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
            
            if np.array_equal(x0, xT):
                for i in range(start_idx, end_idx + 1):
                    predicted_paths[j, i] = x0
            else:
                for i in range(start_idx, end_idx + 1):
                    if i == start_idx:
                        predicted_paths[j, i] = x0
                    else:
                        remaining_time = (end_idx - i) * dt
                        drift = mu + (xT - predicted_paths[j, i - 1]) / remaining_time
                        
                        if data[start_idx, 2] == 0:
                            diffusion = sigma * 1.1 * np.sqrt(dt) * np.random.normal(size=2)
                        else:
                            diffusion = sigma * 0.9 * np.sqrt(dt) * np.random.normal(size=2)
                        
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

def calculate_aee_per_step(true_path, predicted_paths, indices):
    num_steps = len(indices)
    aee_per_step = np.zeros(num_steps)
    for i, idx in enumerate(indices):
        errors = np.sqrt(np.sum((true_path[idx, :2] - predicted_paths[:, idx, :]) ** 2, axis=1))
        aee_per_step[i] = np.mean(errors)
    return aee_per_step

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
file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
data = load_data(file_path)
N = len(data) - 1
T = 1
dt = T / N
timestamps = data[:, 3]
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

# Use original data for true segment lengths
true_segment_lengths = calculate_segment_lengths(data, segments)

# Resample true path to match predicted path timestamps
def resample_true_path(true_path, predicted_paths, timestamps):
    resampled_true_path = np.zeros_like(predicted_paths[0])
    time_indices = np.linspace(0, len(true_path) - 1, len(predicted_paths[0])).astype(int)
    resampled_true_path[:, :2] = true_path[time_indices, :2]
    return resampled_true_path, time_indices

resampled_true_path, time_indices = resample_true_path(data, predicted_paths_bridge, timestamps)

aee_bridge = calculate_aee_per_step(data, predicted_paths_bridge, time_indices)
aee_brownian = calculate_aee_per_step(data, predicted_paths_brownian, time_indices)

predicted_segment_lengths = []
for j in range(predicted_paths_bridge.shape[0]):
    predicted_segment_lengths.extend(calculate_segment_lengths(predicted_paths_bridge[j], segments))

# 过滤掉长度为0的段
true_segment_lengths = [length for length in true_segment_lengths if length != 0.0]
predicted_segment_lengths = [length for length in predicted_segment_lengths if length != 0.0]

# 设置x轴的最大范围
max_segment_length = min(max(true_segment_lengths), max(predicted_segment_lengths), 1.2)

# 更新画图代码
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Path Segment Lengths', density=True, range=(0, max_segment_length))
plt.hist(predicted_segment_lengths, bins=30, alpha=0.5, label='Predicted Path Segment Lengths', density=True, range=(0, max_segment_length))
plt.xlabel('Segment Length')
plt.ylabel('Proportion')
plt.title('Distribution of Segment Lengths')
plt.legend()
plt.xlim(0, max_segment_length)  # 限制x轴范围
plt.show()

# Kolmogorov-Smirnov检验
ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths)
print(f"KS Statistic: {ks_stat}, P-value: {p_value}")
# KS检验结论
if p_value < 0.05:
    print("The null hypothesis that the two distributions are the same is rejected at the 5% significance level.")
else:
    print("The null hypothesis that the two distributions are the same cannot be rejected at the 5% significance level.")

# Plot simulated paths
fig, ax = plt.subplots(figsize=(30, 30))
ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor='w', show=False, close=False)

# Plot all Brownian Motion paths
for j in range(100):
    ax.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)

# Plot true path and snapped segmented Brownian Bridge paths
for j in range(1):  # Reduced the number of plotted snapped paths to 10 for clarity
    ax.plot(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0], color='red', alpha=0.5, label='Segmented Brownian Bridge' if j == 0 else "")
ax.plot(data[:, 1], data[:, 0], label='True Path', color='blue', linewidth=2)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Simulated Paths using Segmented Brownian Bridge and Brownian Motion with Estimated Parameters')
ax.legend()
plt.show()

# Plot AEE per step
plt.figure(figsize=(10, 6))
plt.plot(timestamps[time_indices], aee_bridge, label='AEE (Segmented Brownian Bridge)', color='red')
plt.plot(timestamps[time_indices], aee_brownian, label='AEE (Brownian Motion)', color='green')
plt.xlabel('Timestamp')
plt.ylabel('AEE')
plt.title('AEE per Step for Segmented Brownian Bridge and Brownian Motion')
plt.legend()
plt.show()

def calculate_mae(true_path, predicted_paths, indices):
    errors = np.abs(true_path[indices, :2] - predicted_paths[:, indices, :2])
    mae = np.mean(errors)
    return mae

def calculate_mse(true_path, predicted_paths, indices):
    errors = (true_path[indices, :2] - predicted_paths[:, indices, :2]) ** 2
    mse = np.mean(errors)
    return mse

def calculate_max_ae(true_path, predicted_paths, indices):
    errors = np.abs(true_path[indices, :2] - predicted_paths[:, indices, :2])
    max_ae = np.max(errors)
    return max_ae

def calculate_segment_level_mae(true_segments, predicted_segments):
    segment_mae = np.mean([np.abs(true_segments[i] - predicted_segments[i]) for i in range(len(true_segments))])
    return segment_mae

def calculate_segment_level_mse(true_segments, predicted_segments):
    segment_mse = np.mean([(true_segments[i] - predicted_segments[i]) ** 2 for i in range(len(true_segments))])
    return segment_mse

def calculate_r_squared(true_segments, predicted_segments):
    ss_res = np.sum([(true_segments[i] - predicted_segments[i]) ** 2 for i in range(len(true_segments))])
    ss_tot = np.sum([(true_segments[i] - np.mean(true_segments)) ** 2 for i in range(len(true_segments))])
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Calculate additional metrics
mae_bridge = calculate_mae(data, predicted_paths_bridge, time_indices)
mae_brownian = calculate_mae(data, predicted_paths_brownian, time_indices)

mse_bridge = calculate_mse(data, predicted_paths_bridge, time_indices)
mse_brownian = calculate_mse(data, predicted_paths_brownian, time_indices)

max_ae_bridge = calculate_max_ae(data, predicted_paths_bridge, time_indices)
max_ae_brownian = calculate_max_ae(data, predicted_paths_brownian, time_indices)

segment_mae_bridge = calculate_segment_level_mae(true_segment_lengths, predicted_segment_lengths)
segment_mse_bridge = calculate_segment_level_mse(true_segment_lengths, predicted_segment_lengths)
r_squared_bridge = calculate_r_squared(true_segment_lengths, predicted_segment_lengths)

# Print additional metrics
print(f"MAE (Segmented Brownian Bridge): {mae_bridge}")
print(f"MAE (Brownian Motion): {mae_brownian}")

print(f"MSE (Segmented Brownian Bridge): {mse_bridge}")
print(f"MSE (Brownian Motion): {mse_brownian}")

print(f"Max AE (Segmented Brownian Bridge): {max_ae_bridge}")
print(f"Max AE (Brownian Motion): {max_ae_brownian}")

print(f"Segment Level MAE (Segmented Brownian Bridge): {segment_mae_bridge}")
print(f"Segment Level MSE (Segmented Brownian Bridge): {segment_mse_bridge}")
print(f"R-squared (Segmented Brownian Bridge): {r_squared_bridge}")
