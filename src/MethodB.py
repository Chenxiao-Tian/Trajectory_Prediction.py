# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:07:42 2024

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

def calculate_edit_distance(trace1, trace2):
    # 计算两条轨迹之间的编辑距离
    min_length = min(len(trace1), len(trace2))
    trace1 = trace1[:min_length]
    trace2 = trace2[:min_length]
    return np.sum(np.sqrt(np.sum((trace1 - trace2) ** 2, axis=1)))
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
def method_b_prediction(current_trace, matched_segments, current_segments):
    predicted_locations = []
    for segment_indices in current_segments:
        start_idx = segment_indices[0]
        end_idx = segment_indices[-1]
        start = current_trace.iloc[start_idx, :2].values
        end = current_trace.iloc[end_idx, :2].values
        
        segment_predictions = []
        distances = []
        
        for matched_segment in matched_segments:
            matched_start = matched_segment[0, :2]
            matched_end = matched_segment[-1, :2]
            matched_distance = calculate_edit_distance(current_trace.iloc[segment_indices, :2].values, matched_segment[:, :2])
            
            # 预测位置向前扩展一步
            predicted_location = predict_location(matched_start, matched_end, 1, 1, np.std(matched_segment[:, :2], axis=0))
            segment_predictions.append(predicted_location)
            distances.append(1 / (matched_distance + 1e-6))  # 防止除零错误
        
        # 计算加权平均位置
        predicted_location = np.average(segment_predictions, axis=0, weights=distances)
        predicted_locations.extend([predicted_location] * len(segment_indices))
    
    return np.array(predicted_locations)
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

    # Method B 预测位置
    predicted_locations_method_b = method_b_prediction(current_trace, matched_segments, current_segments)

    # 计算AEE误差
    true_path = current_trace[['Latitude', 'Longitude']].values
    aee_error_method_b = calculate_aee(true_path, predicted_locations_method_b)
    print(f"AEE Error (Method B): {aee_error_method_b}")

    # 计算段长度分布
    true_segment_lengths = calculate_segment_lengths(true_path, current_segments)
    predicted_segment_lengths_method_b = calculate_segment_lengths(predicted_locations_method_b, current_segments)

    # Kolmogorov-Smirnov检验
    ks_stat_method_b, p_value_method_b = ks_2samp(true_segment_lengths, predicted_segment_lengths_method_b)
    print(f"KS Statistic (Method B): {ks_stat_method_b}, P-value: {p_value_method_b}")

    # 可视化真实路径和预测路径
    plt.figure(figsize=(10, 6))
    plt.plot(true_path[:, 1], true_path[:, 0], label='True Path', color='blue')
    plt.plot(predicted_locations_method_b[:, 1], predicted_locations_method_b[:, 0], label='Predicted Path (Method B)', color='red', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True Path vs Predicted Path (Method B)')
    plt.legend()
    plt.show()

    # 绘制段长度分布图并加上K-S检验结果
    plt.figure(figsize=(10, 6))
    plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
    plt.hist(predicted_segment_lengths_method_b, bins=30, alpha=0.5, label='Predicted Segment Lengths (Method B)')
    plt.xlabel('Segment Length')
    plt.ylabel('Frequency')
    plt.title('Segment Length Distribution')
    plt.legend()
    plt.text(0.95, 0.95, f'KS Statistic: {ks_stat_method_b:.2f}\nP-value: {p_value_method_b:.2e}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    # 绘制AEE误差随时间变化的图
    time_stamps = current_trace['Timestamp'].values
    errors_method_b = np.sqrt(np.sum((true_path - predicted_locations_method_b) ** 2, axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, errors_method_b, label='AEE per Time Step (Method B)', color='green')
    plt.xlabel('Timestamp')
    plt.ylabel('AEE')
    plt.title('AEE per Time Step (Method B)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
