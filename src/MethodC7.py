# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 02:44:27 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 02:11:28 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
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
def find_similar_segment(start, end, passenger_status, all_segments):
    similar_segments = []
    for seg in all_segments:
        if seg[0, 2] == passenger_status:  # 仅考虑相同状态的路径段
            distance_start = euclidean(start, seg[0, :2])
            distance_end = euclidean(end, seg[-1, :2])
            total_distance = distance_start + distance_end
            similar_segments.append((total_distance, seg))
    similar_segments.sort(key=lambda x: x[0])
    return similar_segments[0] if similar_segments else None  # 选取最相似的一个段

# 布朗桥分段生成
def brownian_bridge_segment(start, end, steps, sigma, mu):
    t = np.linspace(0, 1, steps)
    bridge = np.zeros((steps, 2))
    dt = 1 / steps
    bridge[0] = start
    for i in range(1, steps):
        remaining_time = 1 - t[i]
        drift = mu + (end - bridge[i - 1]) / remaining_time if remaining_time > 0 else mu
        diffusion = sigma * np.sqrt(dt) * np.random.normal(size=2)
        bridge[i] = bridge[i - 1] + drift * dt + diffusion
    bridge[-1] = end
    return bridge

# 预测路径
def predict_with_brownian_bridge(start, end, similar_segment, segment_length):
    if similar_segment is None or len(similar_segment) < 2:
        return np.linspace(start, end, segment_length).reshape(-1, 2)
    
    # 归一化时间
    similar_length = len(similar_segment)
    time_normalized_similar_segment = np.linspace(0, 1, similar_length).reshape(-1, 1)
    time_normalized_predicted_path = np.linspace(0, 1, segment_length).reshape(-1, 1)

    # 初始化预测路径
    predicted_path = np.zeros((segment_length, 2))
    predicted_path[0] = start
    predicted_path[-1] = end
    
    for i in range(1, segment_length - 1):
        t = time_normalized_predicted_path[i]
        idx = int(t * (similar_length - 1))
        if idx >= similar_length - 1:
            break
        sub_start = similar_segment[idx, :2]
        sub_end = similar_segment[idx + 1, :2]
        sub_steps = segment_length // (similar_length - 1)
        if sub_steps <= 1:
            sub_steps = 2
        sub_path = brownian_bridge_segment(sub_start, sub_end, sub_steps, sigma=np.std(sub_end - sub_start), mu=np.mean(sub_end - sub_start))
        predicted_path[i] = sub_path[int(t * (sub_steps - 1))]
    
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
predicted_paths = []

for seg in taxi7_segments:
    start_idx = seg[0]
    end_idx = seg[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    passenger_status = taxi7[start_idx, 2]
    segment_length = end_idx - start_idx + 1
    
    similar_segment = find_similar_segment(start, end, passenger_status, all_segments)
    if similar_segment is not None:
        similar_segment = similar_segment[1]  # 获取路径段
    predicted_path = predict_with_brownian_bridge(start, end, similar_segment, segment_length)
    
    predicted_paths.append(predicted_path)

predicted_paths = np.concatenate(predicted_paths, axis=0)

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

aee_per_step = calculate_aee_per_step(taxi7, predicted_paths, taxi7_segments)
print(f"AEE per step (Brownian Bridge): {aee_per_step}")

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
predicted_segment_lengths = calculate_segment_lengths(predicted_paths, taxi7_segments)

# Kolmogorov-Smirnov检验
ks_stat, p_value = ks_2samp(true_segment_lengths, predicted_segment_lengths)
print(f"KS Statistic (Brownian Bridge): {ks_stat}, P-value: {p_value}")

# 可视化真实路径和预测路径
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths[:, 1], predicted_paths[:, 0], label='Predicted Path (Brownian Bridge)', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()

# 绘制 AEE per step 图
plt.figure(figsize=(10, 6))
steps = np.arange(len(taxi7))
aee_errors = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths) ** 2, axis=1))
plt.plot(steps, aee_errors, label='AEE per step (Brownian Bridge)', color='red')
plt.xlabel('Step')
plt.ylabel('AEE')
plt.title('AEE per Step')
plt.legend()
plt.show()

# 绘制段长度分布图并加上 K-S 检验结果
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths, bins=30, alpha=0.5, label='Predicted Segment Lengths (Brownian Bridge)')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# 在图中添加 K-S 检验结果
plt.text(0.95, 0.95, f'KS Statistic: {ks_stat:.2f}, P-value: {p_value:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
