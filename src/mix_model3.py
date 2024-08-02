# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:19:40 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
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

# 估计布朗桥SDE参数
def estimate_brownian_bridge_params(segment):
    T = len(segment) - 1
    if T == 0:
        return np.array([0, 0]), np.array([0, 0])  # 如果路径段长度为1，返回零参数
    increments = np.diff(segment[:, :2], axis=0)
    dt = 1 / T
    sigma = np.std(increments, axis=0) / np.sqrt(dt)
    mu = np.mean(increments, axis=0) / dt
    return mu, sigma

# 构建特征向量
def build_feature_vectors(data, segments):
    feature_vectors = []
    for seg in segments:
        segment = data[seg[0]:seg[-1]+1]
        start = segment[0, :2]
        end = segment[-1, :2]
        passenger_status = segment[0, 2]
        mu, sigma = estimate_brownian_bridge_params(segment)
        distance = euclidean(start, end)
        features = np.concatenate([start, end, [distance], mu, sigma, [passenger_status]])
        feature_vectors.append(features)
    return np.array(feature_vectors)

# 预测路径
def predict_segment_path(start, end, mu, sigma, segment_length):
    T = segment_length - 1
    if T == 0:
        path = np.zeros((segment_length, 2))
        path[0] = start
        path[-1] = end
        return path  # 如果路径段长度为1，返回起点终点
    dt = 1 / T
    path = np.zeros((segment_length, 2))
    path[0] = start
    path[-1] = end
    for t in range(1, T):
        remaining_time = (T - t) * dt
        drift = mu + (end - path[t-1]) / remaining_time
        diffusion = sigma * np.sqrt(dt) * np.random.normal(size=2)
        path[t] = path[t-1] + drift * dt + diffusion
    return path

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
all_segments = [seg for segs in segments for seg in segs]
all_data = np.concatenate(taxi_data, axis=0)

# 构建历史路径段的特征向量
feature_vectors = build_feature_vectors(all_data, all_segments)

# 使用KNN找到最相似的历史路径段
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(feature_vectors[:, :-1], feature_vectors[:, -1])

# 找到第七个出租车路径段的相似段并进行预测
taxi7_segments = segment_paths(taxi7)
predicted_paths = np.zeros((taxi7.shape[0], 2))

for seg_indices in taxi7_segments:
    start_idx = seg_indices[0]
    end_idx = seg_indices[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    passenger_status = taxi7[start_idx, 2]
    segment_length = end_idx - start_idx + 1

    # 构建预测路径段的特征向量
    distance = euclidean(start, end)
    target_features = np.concatenate([start, end, [distance], [0, 0], [0, 0], [passenger_status]])

    # 使用KNN找到最相似的历史路径段
    indices = knn.kneighbors([target_features[:-1]], return_distance=False)[0]
    similar_segments = [feature_vectors[idx] for idx in indices]

    # 预测路径
    segment_predictions = []
    for seg in similar_segments:
        mu = seg[5:7]
        sigma = seg[7:9]
        predicted_path = predict_segment_path(start, end, mu, sigma, segment_length)
        segment_predictions.append(predicted_path)
    
    segment_predictions = np.mean(segment_predictions, axis=0)
    predicted_paths[start_idx:end_idx+1] = segment_predictions

# 计算AEE per step
def calculate_aee_per_step(true_path, predicted_path, segments):
    errors = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment_errors = np.sqrt(np.sum((true_path[start_idx:end_idx+1, :2] - predicted_path[start_idx:end_idx+1]) ** 2, axis=1))
        errors.append(segment_errors)
    aee_per_step = np.concatenate(errors).mean()
    return aee_per_step

aee_per_step = calculate_aee_per_step(taxi7, predicted_paths, taxi7_segments)
print(f"AEE per step: {aee_per_step}")

# 计算段长度分布
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

true_segment_lengths = calculate_segment_lengths(taxi7, taxi7_segments)
predicted_segment_lengths = calculate_segment_lengths(predicted_paths, taxi7_segments)

# Kolmogorov-Smirnov检验
ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths)
print(f"KS Statistic: {ks_stat}, P-value: {p_value}")

# 可视化真实路径和预测路径
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths[:, 1], predicted_paths[:, 0], label='Predicted Path', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()

# 绘制AEE per step图
plt.figure(figsize=(10, 6))
steps = np.arange(len(taxi7))
aee_errors = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths) ** 2, axis=1))
plt.plot(steps, aee_errors, label='AEE per step', color='green')
plt.xlabel('Step')
plt.ylabel('AEE')
plt.title('AEE per Step')
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

# 在图中添加K-S检验结果
plt.text(0.95, 0.95, f'KS Statistic: {ks_stat:.2f}\nP-value: {p_value:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
