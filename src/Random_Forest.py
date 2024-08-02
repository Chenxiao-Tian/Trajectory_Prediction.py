# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:09:07 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

# 分段函数
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

# 路径相似性计算
def find_similar_segments(start, end, passenger_status, all_segments):
    similar_segments = []
    for seg in all_segments:
        if seg[0, 2] == passenger_status:  # 仅考虑相同状态的路径段
            distance_start = euclidean(start, seg[0, :2])
            distance_end = euclidean(end, seg[-1, :2])
            total_distance = distance_start + distance_end
            similar_segments.append((total_distance, seg))
    similar_segments.sort(key=lambda x: x[0])
    return similar_segments[:5]  # 选取最相似的5个段

# 预测路径（随机森林方法）
def predict_path_rf(start, end, similar_segments, segment_length):
    rf = RandomForestRegressor(n_estimators=100)
    X_train = []
    y_train = []
    for s in similar_segments:
        seg_length = len(s[1])
        X_train.append(np.linspace(0, 1, seg_length).reshape(-1, 1))
        y_train.append(s[1][:, :2])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    rf.fit(X_train, y_train)

    X_pred = np.linspace(0, 1, segment_length).reshape(-1, 1)
    predicted_path = rf.predict(X_pred)
    return predicted_path

# 计算每个路径段的长度
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

# 加载出租车数据
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

# 获取每个出租车的路径段
segments = [segment_paths(data) for data in taxi_data]
all_segments = [data[seg[0]:seg[-1]+1] for data, segs in zip(taxi_data, segments) for seg in segs]

# 找到第七个出租车路径段的相似段并进行预测（随机森林方法）
taxi7_segments = segment_paths(taxi7)
predicted_paths_rf = []

for seg_indices in taxi7_segments:
    start_idx = seg_indices[0]
    end_idx = seg_indices[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    passenger_status = taxi7[start_idx, 2]
    segment_length = end_idx - start_idx + 1
    
    similar_segments = find_similar_segments(start, end, passenger_status, all_segments)
    predicted_path = predict_path_rf(start, end, similar_segments, segment_length)
    
    # 确保起点和终点不变
    predicted_path[0] = start
    predicted_path[-1] = end
    
    predicted_paths_rf.append(predicted_path)

predicted_paths_rf = np.concatenate(predicted_paths_rf, axis=0)

# 计算AEE per step（随机森林方法）
def calculate_aee_per_step(true_path, predicted_path, segments):
    errors = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment_errors = np.sqrt(np.sum((true_path[start_idx:end_idx+1, :2] - predicted_path[start_idx:end_idx+1]) ** 2, axis=1))
        errors.append(segment_errors)
    aee_per_step = np.concatenate(errors).mean()
    return aee_per_step

aee_per_step_rf = calculate_aee_per_step(taxi7, predicted_paths_rf, taxi7_segments)
print(f"AEE per step (Random Forest): {aee_per_step_rf}")

# 计算段长度分布（随机森林方法）
true_segment_lengths = calculate_segment_lengths(taxi7, taxi7_segments)
predicted_segment_lengths_rf = calculate_segment_lengths(predicted_paths_rf, taxi7_segments)

# Kolmogorov-Smirnov检验（随机森林方法）
ks_stat_rf, p_value_rf = ks_2samp(true_segment_lengths, predicted_segment_lengths_rf)
print(f"KS Statistic (Random Forest): {ks_stat_rf}, P-value (Random Forest): {p_value_rf}")

# 可视化真实路径和预测路径（随机森林方法）
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths_rf[:, 1], predicted_paths_rf[:, 0], label='Predicted Path (Random Forest)', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path (Random Forest)')
plt.legend()
plt.show()

# 绘制AEE per step图（随机森林方法）
plt.figure(figsize=(10, 6))
steps = np.arange(len(taxi7))
aee_errors_rf = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths_rf) ** 2, axis=1))
plt.plot(steps, aee_errors_rf, label='AEE per step (Random Forest)', color='green')
plt.xlabel('Step')
plt.ylabel('AEE')
plt.title('AEE per Step (Random Forest)')
plt.legend()
plt.show()

# 绘制段长度分布图并加上K-S检验结果（随机森林方法）
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths_rf, bins=30, alpha=0.5, label='Predicted Segment Lengths')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# 在图中添加K-S检验结果
plt.text(0.95, 0.95, f'KS Statistic: {ks_stat_rf:.2f}\nP-value: {p_value_rf:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
