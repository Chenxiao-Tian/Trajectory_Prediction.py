# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 23:01:22 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

# Segment paths
def segment_paths(data):
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
    return segments

# Linear Extrapolation function
def linear_extrapolation(start, end, num_points):
    segment_path = []
    for t in range(num_points):
        x_p = start[0] + (end[0] - start[0]) * t / (num_points - 1)
        y_p = start[1] + (end[1] - start[1]) * t / (num_points - 1)
        segment_path.append([x_p, y_p])
    return segment_path

# Main function to predict the entire path
def predict_path(data, segments):
    predicted_path = []
    for seg_indices in segments:
        start_idx = seg_indices[0]
        end_idx = seg_indices[-1]
        start = data[start_idx, :2]
        end = data[end_idx, :2]
        num_points = end_idx - start_idx + 1
        segment_path = linear_extrapolation(start, end, num_points)
        predicted_path.extend(segment_path)
    return np.array(predicted_path)

# Calculate Average Euclidean Error (AEE) per step
def calculate_aee_per_step(true_path, predicted_path):
    errors = np.sqrt(np.sum((true_path[:, :2] - predicted_path[:, :2]) ** 2, axis=1))
    return errors

# Calculate segment lengths
def calculate_segment_lengths(path, segments):
    segment_lengths = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment = path[start_idx:end_idx+1]
        segment_length = np.sum(np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1)))
        segment_lengths.append(segment_length)
    return segment_lengths

# Plotting the results
def plot_paths(true_path, predicted_path):
    plt.figure(figsize=(10, 6))
    plt.plot(true_path[:, 1], true_path[:, 0], label='True Path', color='blue')
    plt.plot(predicted_path[:, 1], predicted_path[:, 0], label='Predicted Path', color='red', linestyle='--')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True Path vs Predicted Path (Linear Extrapolation)')
    plt.legend()
    plt.show()

# Plot AEE per step
def plot_aee_per_step(aee_per_step):
    plt.figure(figsize=(10, 6))
    plt.plot(aee_per_step, label='AEE per Step', color='green')
    plt.xlabel('Step')
    plt.ylabel('AEE')
    plt.title('AEE per Step for Linear Extrapolation')
    plt.legend()
    plt.show()

# Main function to load data, segment paths, apply linear extrapolation, calculate AEE and segment lengths, and plot results
def main():
    taxi_data_file = 'taxi7.txt'
    data = load_data(taxi_data_file)
    segments = segment_paths(data)
    predicted_path = predict_path(data, segments)
    
    aee_per_step = calculate_aee_per_step(data, predicted_path)
    print(f"Average AEE per step: {np.nanmean(aee_per_step)}")

    true_segment_lengths = calculate_segment_lengths(data, segments)
    predicted_segment_lengths = calculate_segment_lengths(predicted_path, segments)

    # Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths)
    print(f"KS Statistic: {ks_stat}, P-value: {p_value}")

    # Plot paths
    plot_paths(data, predicted_path)

    # Plot AEE per step
    plot_aee_per_step(aee_per_step)

    # Plot segment length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths', density=True)
    plt.hist(predicted_segment_lengths, bins=30, alpha=0.5, label='Predicted Segment Lengths', density=True)
    plt.xlabel('Segment Length')
    plt.ylabel('Density')
    plt.title('Segment Length Distribution')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
