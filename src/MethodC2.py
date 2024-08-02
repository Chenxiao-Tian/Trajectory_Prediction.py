# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:51:55 2024

@author: 92585
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances_argmin_min

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
def normalize_time(data, segments):
    normalized_data = data.copy()
    for segment in segments:
        t0 = data.iloc[segment[0], 3]
        t1 = data.iloc[segment[-1], 3]
        if t1 != t0:
            for idx in segment:
                normalized_data.at[idx, 'Timestamp'] = (data.at[idx, 'Timestamp'] - t0) / (t1 - t0)
        else:
            for idx in segment:
                normalized_data.at[idx, 'Timestamp'] = 0.5  # Set to the middle of the segment
    return normalized_data

def match_traces(current_segment, historical_traces):
    start = current_segment[0, :2]
    end = current_segment[-1, :2]
    distances, indices = pairwise_distances_argmin_min([start, end], historical_traces.reshape(-1, 2))
    indices = indices.astype(int)  # 确保索引为整数类型
    matched_segments = historical_traces[indices // historical_traces.shape[1]]
    return matched_segments
def brownian_bridge_prediction(start, end, t, T, sigma_m):
    mu_t = start + (end - start) * t / T
    sigma_t = np.sqrt(t * (T - t) / T) * sigma_m
    return np.random.normal(mu_t, sigma_t)

def predict_location(start, end, t, T, sigma_m):
    mu_t = start + (end - start) * t / T
    sigma_t = np.sqrt(t * (T - t) / T) * sigma_m
    predicted_location = np.random.normal(mu_t, sigma_t)
    return predicted_location
def calculate_aee(true_path, predicted_path):
    return np.mean(np.sqrt(np.sum((true_path - predicted_path) ** 2, axis=1)))

def calculate_segment_lengths(path, segments):
    segment_lengths = []
    for segment in segments:
        start_idx = segment[0]
        end_idx = segment[-1]
        segment = path[start_idx:end_idx+1]
        segment_length = np.sum(np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1)))
        segment_lengths.append(segment_length)
    return segment_lengths
def main():
    # 加载数据
    file_paths = ['taxi1.txt', 'taxi2.txt', 'taxi3.txt', 'taxi4.txt', 'taxi5.txt', 'taxi6.txt']
    historical_traces = [load_data(file) for file in file_paths]
    current_trace = load_data('taxi7.txt')

    # 分割轨迹
    historical_segments = [segment_paths(trace) for trace in historical_traces]
    current_segments = segment_paths(current_trace)

    # 归一化时间
    normalized_historical_traces = [normalize_time(trace, segments) for trace, segments in zip(historical_traces, historical_segments)]
    normalized_current_trace = normalize_time(current_trace, current_segments)

    # 将归一化的历史轨迹转换为合适的结构
    historical_traces_array = np.concatenate([trace.values for trace in normalized_historical_traces], axis=0)
    current_trace_array = normalized_current_trace.values

    # 匹配相似轨迹
    matched_segments = []
    for segment_indices in current_segments:
        segment_data = current_trace_array[segment_indices]
        matched_segment = match_traces(segment_data, historical_traces_array)
        matched_segments.append(matched_segment)

    # 预测位置
    predicted_locations = []
    for segment_indices in current_segments:
        start_idx = segment_indices[0]
        end_idx = segment_indices[-1]
        start = current_trace.iloc[start_idx, :2].values
        end = current_trace.iloc[end_idx, :2].values
        t = np.linspace(0, 1, end_idx - start_idx + 1)
        T = 1
        sigma_m = np.std(current_trace.iloc[segment_indices, :2], axis=0)
        for tx in t:
            predicted_location = predict_location(start, end, tx, T, sigma_m)
            predicted_locations.append(predicted_location)

    predicted_locations = np.array(predicted_locations)

    # 计算AEE误差
    true_path = current_trace[['Latitude', 'Longitude']].values
    aee_error = calculate_aee(true_path, predicted_locations)
    print(f"AEE Error: {aee_error}")

    # 计算段长度分布
    true_segment_lengths = calculate_segment_lengths(true_path, current_segments)
    predicted_segment_lengths = calculate_segment_lengths(predicted_locations, current_segments)

    # Kolmogorov-Smirnov检验
    ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths)
    print(f"KS Statistic: {ks_stat}, P-value: {p_value}")

    # 可视化真实路径和预测路径
    plt.figure(figsize=(10, 6))
    plt.plot(true_path[:, 1], true_path[:, 0], label='True Path', color='blue')
    plt.plot(predicted_locations[:, 1], predicted_locations[:, 0], label='Predicted Path', color='red', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True Path vs Predicted Path')
    plt.legend()
    plt.show()

    # 绘制段长度分布图并加上K-S检验结果
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

    # 绘制AEE误差随时间变化的图
    time_stamps = current_trace['Timestamp'].values
    errors = np.sqrt(np.sum((true_path - predicted_locations) ** 2, axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, errors, label='AEE per Time Step', color='green')
    plt.xlabel('Timestamp')
    plt.ylabel('AEE')
    plt.title('AEE per Time Step')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
