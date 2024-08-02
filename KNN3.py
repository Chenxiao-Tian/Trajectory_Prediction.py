# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:55:23 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:43:33 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import ks_2samp

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment', 'Timestamp']].values

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
    knn = KNeighborsRegressor(n_neighbors=5)
    X_train = []
    y_train = []
    for s in similar_segments:
        seg_length = len(s[1])
        X_train.append(np.linspace(0, 1, seg_length).reshape(-1, 1))
        y_train.append(s[1][:, :2])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    knn.fit(X_train, y_train)

    X_pred = np.linspace(0, 1, segment_length).reshape(-1, 1)
    predicted_path = knn.predict(X_pred)
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

# 计算AEE per step
def calculate_aee_per_step(true_path, predicted_path, timestamps):
    num_steps = len(timestamps)
    aee_per_step = np.zeros(num_steps)
    for i in range(num_steps):
        true_position = true_path[i, :2]
        predicted_position = predicted_path[i, :]
        errors = np.sqrt(np.sum((true_position - predicted_position) ** 2))
        aee_per_step[i] = errors
    return aee_per_step

# 计算 MAE
def calculate_mae(true_path, predicted_path):
    errors = np.abs(true_path[:, :2] - predicted_path)
    mae = np.mean(errors)
    return mae

# 计算 MSE
def calculate_mse(true_path, predicted_path):
    errors = (true_path[:, :2] - predicted_path) ** 2
    mse = np.mean(errors)
    return mse

# 计算最大绝对误差 (Max AE)
def calculate_max_ae(true_path, predicted_path):
    errors = np.abs(true_path[:, :2] - predicted_path)
    max_ae = np.max(errors)
    return max_ae

# 计算段级别 MAE
def calculate_segment_level_mae(true_segments, predicted_segments):
    segment_mae = np.mean([np.abs(true_segments[i] - predicted_segments[i]) for i in range(len(true_segments))])
    return segment_mae

# 计算段级别 MSE
def calculate_segment_level_mse(true_segments, predicted_segments):
    segment_mse = np.mean([(true_segments[i] - predicted_segments[i]) ** 2 for i in range(len(true_segments))])
    return segment_mse

# 计算 R-squared
def calculate_r_squared(true_segments, predicted_segments):
    ss_res = np.sum([(true_segments[i] - predicted_segments[i]) ** 2 for i in range(len(true_segments))])
    ss_tot = np.sum([(true_segments[i] - np.mean(true_segments)) ** 2 for i in range(len(true_segments))])
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

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

# 获取时间戳
timestamps = taxi7[:, 3]

# 计算 AEE per step
aee_per_step = calculate_aee_per_step(taxi7, predicted_paths, timestamps)

# 计算段长度分布
true_segment_lengths = calculate_segment_lengths(taxi7, taxi7_segments)
predicted_segment_lengths = calculate_segment_lengths(predicted_paths, taxi7_segments)

# 计算各种metrics
mae = calculate_mae(taxi7, predicted_paths)
mse = calculate_mse(taxi7, predicted_paths)
max_ae = calculate_max_ae(taxi7, predicted_paths)
segment_mae = calculate_segment_level_mae(true_segment_lengths, predicted_segment_lengths)
segment_mse = calculate_segment_level_mse(true_segment_lengths, predicted_segment_lengths)
r_squared = calculate_r_squared(true_segment_lengths, predicted_segment_lengths)

# 打印各种metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"Max AE: {max_ae}")
print(f"Segment Level MAE: {segment_mae}")
print(f"Segment Level MSE: {segment_mse}")
print(f"R-squared: {r_squared}")

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

# 绘制 AEE per step 图
plt.figure(figsize=(10, 6))
plt.plot(timestamps, aee_per_step, label='AEE per step', color='green')
plt.xlabel('Timestamp')
plt.ylabel('AEE')
plt.title('AEE per Step')
plt.legend()
plt.gca().invert_xaxis()  # 反转x轴
plt.show()

# 绘制段长度分布图并加上 K-S 检验结果
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths, bins=30, alpha=0.5, label='Predicted Segment Lengths')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# 在图中添加 K-S 检验结果
plt.text(0.95, 0.95, f'KS Statistic: {ks_stat:.2f}\nP-value: {p_value:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
