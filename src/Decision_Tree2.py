# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:04:14 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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

# 线性回归预测
def linear_regression_predict(start, end, similar_segments, segment_length):
    lr = LinearRegression()
    X_train = []
    y_train = []
    for s in similar_segments:
        seg_length = len(s[1])
        X_train.append(np.linspace(0, 1, seg_length).reshape(-1, 1))
        y_train.append(s[1][:, :2])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    lr.fit(X_train, y_train)

    X_pred = np.linspace(0, 1, segment_length).reshape(-1, 1)
    predicted_path = lr.predict(X_pred)
    return predicted_path

# 决策树回归预测
def decision_tree_predict(start, end, similar_segments, segment_length):
    dt = DecisionTreeRegressor()
    X_train = []
    y_train = []
    for s in similar_segments:
        seg_length = len(s[1])
        X_train.append(np.linspace(0, 1, seg_length).reshape(-1, 1))
        y_train.append(s[1][:, :2])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    dt.fit(X_train, y_train)

    X_pred = np.linspace(0, 1, segment_length).reshape(-1, 1)
    predicted_path = dt.predict(X_pred)
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
predicted_paths_lr = []
predicted_paths_dt = []

for seg in taxi7_segments:
    start_idx = seg[0]
    end_idx = seg[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    passenger_status = taxi7[start_idx, 2]
    segment_length = end_idx - start_idx + 1
    
    similar_segments = find_similar_segments(start, end, passenger_status, all_segments)
    
    predicted_path_lr = linear_regression_predict(start, end, similar_segments, segment_length)
    predicted_path_dt = decision_tree_predict(start, end, similar_segments, segment_length)
    
    # 确保起点和终点不变
    predicted_path_lr[0] = start
    predicted_path_lr[-1] = end
    predicted_path_dt[0] = start
    predicted_path_dt[-1] = end
    
    predicted_paths_lr.append(predicted_path_lr)
    predicted_paths_dt.append(predicted_path_dt)

predicted_paths_lr = np.concatenate(predicted_paths_lr, axis=0)
predicted_paths_dt = np.concatenate(predicted_paths_dt, axis=0)

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

aee_per_step_lr = calculate_aee_per_step(taxi7, predicted_paths_lr, taxi7_segments)
aee_per_step_dt = calculate_aee_per_step(taxi7, predicted_paths_dt, taxi7_segments)
print(f"AEE per step (Linear Regression): {aee_per_step_lr}")
print(f"AEE per step (Decision Tree): {aee_per_step_dt}")

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
predicted_segment_lengths_lr = calculate_segment_lengths(predicted_paths_lr, taxi7_segments)
predicted_segment_lengths_dt = calculate_segment_lengths(predicted_paths_dt, taxi7_segments)

# Kolmogorov-Smirnov检验
ks_stat_lr, p_value_lr = ks_2samp(true_segment_lengths, predicted_segment_lengths_lr)
ks_stat_dt, p_value_dt = ks_2samp(true_segment_lengths, predicted_segment_lengths_dt)
print(f"KS Statistic (Linear Regression): {ks_stat_lr}, P-value: {p_value_lr}")
print(f"KS Statistic (Decision Tree): {ks_stat_dt}, P-value: {p_value_dt}")

# 计算其他度量
def calculate_metrics(true_path, predicted_path):
    mae = np.mean(np.sqrt(np.sum((true_path[:, :2] - predicted_path) ** 2, axis=1)))
    mse = np.mean(np.sum((true_path[:, :2] - predicted_path) ** 2, axis=1))
    max_ae = np.max(np.sqrt(np.sum((true_path[:, :2] - predicted_path) ** 2, axis=1)))
    segment_mae = np.mean([np.mean(np.sqrt(np.sum((true_path[seg[0]:seg[-1]+1, :2] - predicted_path[seg[0]:seg[-1]+1]) ** 2, axis=1))) for seg in taxi7_segments])
    segment_mse = np.mean([np.mean(np.sum((true_path[seg[0]:seg[-1]+1, :2] - predicted_path[seg[0]:seg[-1]+1]) ** 2, axis=1)) for seg in taxi7_segments])
    ss_total = np.sum((true_path[:, :2] - np.mean(true_path[:, :2], axis=0)) ** 2)
    ss_res = np.sum((true_path[:, :2] - predicted_path) ** 2)
    r_squared = 1 - (ss_res / ss_total)
    return mae, mse, max_ae, segment_mae, segment_mse, r_squared

metrics_lr = calculate_metrics(taxi7, predicted_paths_lr)
metrics_dt = calculate_metrics(taxi7, predicted_paths_dt)

print(f"MAE (Linear Regression): {metrics_lr[0]}")
print(f"MSE (Linear Regression): {metrics_lr[1]}")
print(f"Max AE (Linear Regression): {metrics_lr[2]}")
print(f"Segment Level MAE (Linear Regression): {metrics_lr[3]}")
print(f"Segment Level MSE (Linear Regression): {metrics_lr[4]}")
print(f"R-squared (Linear Regression): {metrics_lr[5]}")

print(f"MAE (Decision Tree): {metrics_dt[0]}")
print(f"MSE (Decision Tree): {metrics_dt[1]}")
print(f"Max AE (Decision Tree): {metrics_dt[2]}")
print(f"Segment Level MAE (Decision Tree): {metrics_dt[3]}")
print(f"Segment Level MSE (Decision Tree): {metrics_dt[4]}")
print(f"R-squared (Decision Tree): {metrics_dt[5]}")

# 可视化真实路径和预测路径
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths_lr[:, 1], predicted_paths_lr[:, 0], label='Predicted Path (Linear Regression)', color='red', alpha=0.6)
plt.plot(predicted_paths_dt[:, 1], predicted_paths_dt[:, 0], label='Predicted Path (Decision Tree)', color='green', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()

# 绘制 AEE per step 图
plt.figure(figsize=(10, 6))
steps = np.arange(len(taxi7))
aee_errors_lr = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths_lr) ** 2, axis=1))
aee_errors_dt = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths_dt) ** 2, axis=1))
plt.plot(steps, aee_errors_lr, label='AEE per step (Linear Regression)', color='red')
plt.plot(steps, aee_errors_dt, label='AEE per step (Decision Tree)', color='green')
plt.xlabel('Step')
plt.ylabel('AEE')
plt.title('AEE per Step')
plt.legend()
plt.show()

# 绘制段长度分布图并加上 K-S 检验结果
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths_lr, bins=30, alpha=0.5, label='Predicted Segment Lengths (Linear Regression)')
plt.hist(predicted_segment_lengths_dt, bins=30, alpha=0.5, label='Predicted Segment Lengths (Decision Tree)')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# 在图中添加 K-S 检验结果
plt.text(0.95, 0.95, f'KS Statistic (LR): {ks_stat_lr:.2f}, P-value (LR): {p_value_lr:.2e}\nKS Statistic (DT): {ks_stat_dt:.2f}, P-value (DT): {p_value_dt:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
