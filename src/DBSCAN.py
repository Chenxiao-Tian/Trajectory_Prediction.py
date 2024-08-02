# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:22:03 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
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

# 预测路径
def predict_path(start, end, similar_segments, segment_length):
    all_points = np.concatenate([s[1][:, :2] for s in similar_segments], axis=0)
    
    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=0.001, min_samples=2).fit(all_points)
    labels = db.labels_
    
    # 筛选聚类中的主路径
    unique_labels = set(labels)
    longest_cluster = None
    longest_length = 0
    for label in unique_labels:
        if label == -1:  # 忽略噪声
            continue
        cluster_points = all_points[labels == label]
        cluster_length = len(cluster_points)
        if cluster_length > longest_length:
            longest_length = cluster_length
            longest_cluster = cluster_points
            
    # 如果没有找到合适的聚类，直接返回空路径
    if longest_cluster is None:
        return np.zeros((segment_length, 2))
    
    # 预测路径
    x_pred = np.linspace(0, 1, segment_length)
    y_pred_lat = np.interp(x_pred, np.linspace(0, 1, len(longest_cluster)), longest_cluster[:, 0])
    y_pred_lon = np.interp(x_pred, np.linspace(0, 1, len(longest_cluster)), longest_cluster[:, 1])
    predicted_path = np.column_stack((y_pred_lat, y_pred_lon))
    
    return predicted_path

# 计算每个路径段的长度
def calculate_segment_lengths(path, segments):
    segment_lengths = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment = path[start_idx:end_idx+1]
        segment_length = np.sum(np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1)))
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

# 找到第七个出租车路径段的相似段并进行预测
taxi7_segments = segment_paths(taxi7)
predicted_paths = []

for seg_indices in taxi7_segments:
    start_idx = seg_indices[0]
    end_idx = seg_indices[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    passenger_status = taxi7[start_idx, 2]
    segment_length = end_idx - start_idx + 1
    
    similar_segments = find_similar_segments(start, end, passenger_status, all_segments)
    predicted_path = predict_path(start, end, similar_segments, segment_length)
    
    # 确保起点和终点不变
    predicted_path[0] = start
    predicted_path[-1] = end
    
    predicted_paths.append(predicted_path)

predicted_paths = np.concatenate(predicted_paths, axis=0)

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
