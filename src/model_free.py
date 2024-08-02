# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:22:12 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsRegressor

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
def find_similar_segments(start, end, all_segments):
    similar_segments = []
    for seg in all_segments:
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
    segment_length = end_idx - start_idx + 1
    
    similar_segments = find_similar_segments(start, end, all_segments)
    predicted_path = predict_path(start, end, similar_segments, segment_length)
    predicted_paths.append(predicted_path)

predicted_paths = np.concatenate(predicted_paths, axis=0)

# 计算AEE per step
def calculate_aee_per_step(true_path, predicted_path):
    errors = np.sqrt(np.sum((true_path[:, :2] - predicted_path) ** 2, axis=1))
    aee_per_step = np.mean(errors)
    return aee_per_step

aee_per_step = calculate_aee_per_step(taxi7, predicted_paths)
print(f"AEE per step: {aee_per_step}")

# 可视化真实路径和预测路径
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths[:, 1], predicted_paths[:, 0], label='Predicted Path', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()
