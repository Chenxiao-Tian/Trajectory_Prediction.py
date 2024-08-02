# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:57:06 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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
    return similar_segments[:1]  # 选取最相似的5个段

# 随机森林回归预测
def random_forest_predict(start, end, similar_segments, segment_length):
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

# 支持向量机回归预测
def svm_predict(start, end, similar_segments, segment_length):
    svr_lat = SVR()
    svr_lon = SVR()
    X_train = []
    y_train_lat = []
    y_train_lon = []
    for s in similar_segments:
        seg_length = len(s[1])
        X_train.append(np.linspace(0, 1, seg_length).reshape(-1, 1))
        y_train_lat.append(s[1][:, 0])
        y_train_lon.append(s[1][:, 1])
    X_train = np.concatenate(X_train, axis=0)
    y_train_lat = np.concatenate(y_train_lat, axis=0)
    y_train_lon = np.concatenate(y_train_lon, axis=0)
    svr_lat.fit(X_train, y_train_lat)
    svr_lon.fit(X_train, y_train_lon)

    X_pred = np.linspace(0, 1, segment_length).reshape(-1, 1)
    predicted_lat = svr_lat.predict(X_pred)
    predicted_lon = svr_lon.predict(X_pred)
    predicted_path = np.column_stack((predicted_lat, predicted_lon))
    return predicted_path

# 加载历史数据
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
predicted_paths_rf = []
predicted_paths_svr = []

for seg_indices in taxi7_segments:
    start_idx = seg_indices[0]
    end_idx = seg_indices[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    passenger_status = taxi7[start_idx, 2]
    segment_length = end_idx - start_idx + 1
    
    similar_segments = find_similar_segments(start, end, passenger_status, all_segments)
    
    predicted_path_rf = random_forest_predict(start, end, similar_segments, segment_length)
    predicted_path_svr = svm_predict(start, end, similar_segments, segment_length)
    
    # 确保起点和终点不变
    predicted_path_rf[0] = start
    predicted_path_rf[-1] = end
    predicted_path_svr[0] = start
    predicted_path_svr[-1] = end
    
    predicted_paths_rf.append(predicted_path_rf)
    predicted_paths_svr.append(predicted_path_svr)

predicted_paths_rf = np.concatenate(predicted_paths_rf, axis=0)
predicted_paths_svr = np.concatenate(predicted_paths_svr, axis=0)

# 计算 AEE per step
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
aee_per_step_svr = calculate_aee_per_step(taxi7, predicted_paths_svr, taxi7_segments)
print(f"AEE per step (Random Forest): {aee_per_step_rf}")
print(f"AEE per step (SVR): {aee_per_step_svr}")

# 计算段长度分布
def calculate_segment_lengths(path, segments):
    segment_lengths = []
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        segment = path[start_idx:end_idx+1]
        segment_length = np.sum(np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1)))
        segment_lengths.append(segment_length)
    return segment_lengths

true_segment_lengths = calculate_segment_lengths(taxi7, taxi7_segments)
predicted_segment_lengths_rf = calculate_segment_lengths(predicted_paths_rf, taxi7_segments)
predicted_segment_lengths_svr = calculate_segment_lengths(predicted_paths_svr, taxi7_segments)

# Kolmogorov-Smirnov检验
ks_stat_rf, p_value_rf = ks_2samp(true_segment_lengths, predicted_segment_lengths_rf)
ks_stat_svr, p_value_svr = ks_2samp(true_segment_lengths, predicted_segment_lengths_svr)
print(f"KS Statistic (Random Forest): {ks_stat_rf}, P-value: {p_value_rf}")
print(f"KS Statistic (SVR): {ks_stat_svr}, P-value: {p_value_svr}")

# 可视化真实路径和预测路径
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths_rf[:, 1], predicted_paths_rf[:, 0], label='Predicted Path (Random Forest)', color='red', alpha=0.6)
plt.plot(predicted_paths_svr[:, 1], predicted_paths_svr[:, 0], label='Predicted Path (SVR)', color='green', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()

# 绘制 AEE per step 图
plt.figure(figsize=(10, 6))
steps = np.arange(len(taxi7))
aee_errors_rf = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths_rf) ** 2, axis=1))
aee_errors_svr = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths_svr) ** 2, axis=1))
plt.plot(steps, aee_errors_rf, label='AEE per step (Random Forest)', color='red')
plt.plot(steps, aee_errors_svr, label='AEE per step (SVR)', color='green')
plt.xlabel('Step')
plt.ylabel('AEE')
plt.title('AEE per Step')
plt.legend()
plt.show()

# 绘制段长度分布图并加上 K-S 检验结果
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths_rf, bins=30, alpha=0.5, label='Predicted Segment Lengths (Random Forest)')
plt.hist(predicted_segment_lengths_svr, bins=30, alpha=0.5, label='Predicted Segment Lengths (SVR)')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# 在图中添加 K-S 检验结果
plt.text(0.95, 0.95, f'KS Statistic (RF): {ks_stat_rf:.2f}, P-value (RF): {p_value_rf:.2e}\nKS Statistic (SVR): {ks_stat_svr:.2f}, P-value (SVR): {p_value_svr:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
