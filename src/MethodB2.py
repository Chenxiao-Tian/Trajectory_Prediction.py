# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:39:08 2024

@author: 92585
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df

def segment_paths(data):
    segments = []
    current_segment = []
    current_value = data.iloc[0, 2]

    for idx, row in data.iterrows():
        if row['Segment'] == current_value:
            current_segment.append(idx)
        else:
            segments.append(current_segment)
            current_segment = [idx]
            current_value = row['Segment']
    segments.append(current_segment)
    return segments

def find_similar_segments(start, end, passenger_status, all_segments):
    similar_segments = []
    for seg in all_segments:
        if seg.iloc[0, 2] == passenger_status:  # 仅考虑相同状态的路径段
            distance_start = euclidean(start, seg.iloc[0, :2])
            distance_end = euclidean(end, seg.iloc[-1, :2])
            total_distance = distance_start + distance_end
            similar_segments.append((total_distance, seg))
    similar_segments.sort(key=lambda x: x[0])
    return similar_segments[:5]  # 选取最相似的5个段

def method_b_predict(start, end, similar_segments, segment_length):
    predicted_path = []
    weights = []

    for distance, seg in similar_segments:
        weights.append(1 / (distance + 1e-6))
        path = np.linspace(start, end, segment_length)
        predicted_path.append(path)

    predicted_path = np.array(predicted_path)
    weights = np.array(weights) / np.sum(weights)
    
    # 计算加权平均值
    final_prediction = np.average(predicted_path, axis=0, weights=weights)
    
    return final_prediction

def calculate_segment_lengths(path, segments):
    segment_lengths = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment = path[start_idx:end_idx+1]
        segment_length = np.sum(np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1)))
        segment_lengths.append(segment_length)
    return segment_lengths

def calculate_aee_per_step(true_path, predicted_path, segments):
    errors = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment_errors = np.sqrt(np.sum((true_path[start_idx:end_idx+1, :2] - predicted_path[start_idx:end_idx+1]) ** 2, axis=1))
        errors.append(segment_errors)
    aee_per_step = np.concatenate(errors).mean()
    return aee_per_step

def main():
    file_paths = ['taxi1.txt', 'taxi2.txt', 'taxi3.txt', 'taxi4.txt', 'taxi5.txt', 'taxi6.txt']
    historical_traces = [load_data(file) for file in file_paths]
    current_trace = load_data('taxi7.txt')

    historical_segments = [segment_paths(trace) for trace in historical_traces]
    current_segments = segment_paths(current_trace)

    all_segments = [data.iloc[seg[0]:seg[-1]+1] for data, segs in zip(historical_traces, historical_segments) for seg in segs]

    predicted_paths = []

    for seg_indices in current_segments:
        start_idx = seg_indices[0]
        end_idx = seg_indices[-1]
        start = current_trace.iloc[start_idx, :2].values
        end = current_trace.iloc[end_idx, :2].values
        passenger_status = current_trace.iloc[start_idx, 2]
        segment_length = end_idx - start_idx + 1

        similar_segments = find_similar_segments(start, end, passenger_status, all_segments)
        predicted_path = method_b_predict(start, end, similar_segments, segment_length)

        predicted_paths.append(predicted_path)

    predicted_paths = np.concatenate(predicted_paths, axis=0)

    true_path = current_trace[['Latitude', 'Longitude']].values
    aee_per_step = calculate_aee_per_step(true_path, predicted_paths, current_segments)
    print(f"AEE per step: {aee_per_step}")

    true_segment_lengths = calculate_segment_lengths(true_path, current_segments)
    predicted_segment_lengths = calculate_segment_lengths(predicted_paths, current_segments)

    ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths)
    print(f"KS Statistic: {ks_stat}, P-value: {p_value}")

    plt.figure(figsize=(10, 6))
    plt.plot(true_path[:, 1], true_path[:, 0], label='True Path', color='blue')
    plt.plot(predicted_paths[:, 1], predicted_paths[:, 0], label='Predicted Path', color='red', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True Path vs Predicted Path')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    steps = np.arange(len(true_path))
    aee_errors = np.sqrt(np.sum((true_path[:, :2] - predicted_paths) ** 2, axis=1))
    plt.plot(steps, aee_errors, label='AEE per step', color='green')
    plt.xlabel('Step')
    plt.ylabel('AEE')
    plt.title('AEE per Step')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
    plt.hist(predicted_segment_lengths, bins=30, alpha=0.5, label='Predicted Segment Lengths')
    plt.xlabel('Segment Length')
    plt.ylabel('Frequency')
    plt.title('Segment Length Distribution')
    plt.legend()

    plt.text(0.95, 0.95, f'KS Statistic: {ks_stat:.2f}\nP-value: {p_value:.2e}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

if __name__ == "__main__":
    main()
