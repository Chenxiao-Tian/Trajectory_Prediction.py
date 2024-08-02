# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:16:53 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:29:08 2024

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
    return df[['Latitude', 'Longitude', 'Segment']].values
def load_data2(file_path):
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
data = load_data2('taxi7.txt')
timestamps = data[:, 3]
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
plt.plot(timestamps, aee_errors, label='AEE per step', color='green')
plt.xlabel('Timestamp')
plt.ylabel('AEE')
plt.title('AEE per Timestamp')
plt.legend()
plt.gca().invert_xaxis()  # 反转x轴
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

def calculate_mae(true_path, predicted_path):
    errors = np.abs(true_path[:, :2] - predicted_path[:, :2])
    mae = np.mean(errors)
    return mae

def calculate_mse(true_path, predicted_path):
    errors = (true_path[:, :2] - predicted_path[:, :2]) ** 2
    mse = np.mean(errors)
    return mse

def calculate_max_ae(true_path, predicted_path):
    errors = np.abs(true_path[:, :2] - predicted_path[:, :2])
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

# Calculate additional metrics for KNN method
mae_knn = calculate_mae(taxi7, predicted_paths)
mse_knn = calculate_mse(taxi7, predicted_paths)
max_ae_knn = calculate_max_ae(taxi7, predicted_paths)
segment_mae_knn = calculate_segment_level_mae(true_segment_lengths, predicted_segment_lengths)
segment_mse_knn = calculate_segment_level_mse(true_segment_lengths, predicted_segment_lengths)
r_squared_knn = calculate_r_squared(true_segment_lengths, predicted_segment_lengths)

# Print additional metrics
print(f"MAE (KNN): {mae_knn}")
print(f"MSE (KNN): {mse_knn}")
print(f"Max AE (KNN): {max_ae_knn}")
print(f"Segment Level MAE (KNN): {segment_mae_knn}")
print(f"Segment Level MSE (KNN): {segment_mse_knn}")
print(f"R-squared (KNN): {r_squared_knn}")

# Visualize additional metrics
# Plot MAE and MSE
plt.figure(figsize=(10, 6))
plt.bar(['MAE', 'MSE', 'Max AE', 'Segment MAE', 'Segment MSE', 'R-squared'], 
        [mae_knn, mse_knn, max_ae_knn, segment_mae_knn, segment_mse_knn, r_squared_knn])
plt.ylabel('Value')
plt.title('Additional Metrics for KNN Method')
plt.show()

# Plot Segment Length Distribution
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths, bins=30, alpha=0.5, label='Predicted Segment Lengths')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# Add K-S test result to the plot
plt.text(0.95, 0.95, f'KS Statistic: {ks_stat:.2f}\nP-value: {p_value:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
