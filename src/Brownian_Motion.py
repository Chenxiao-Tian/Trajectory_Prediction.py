import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

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

# 纯布朗运动预测
def brownian_motion_predict(start, end, segment_length):
    if segment_length <= 1:
        return np.array([start, end])
    
    dt = 1 / (segment_length - 1)
    sigma = 0.1  # 标准差，可以根据实际情况调整
    path = np.zeros((segment_length, 2))
    path[0] = start
    for i in range(1, segment_length):
        path[i] = path[i-1] + np.random.normal(0, sigma, size=2) * np.sqrt(dt)
    path[-1] = end
    return path

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
predicted_paths_bm = []

for seg in taxi7_segments:
    start_idx = seg[0]
    end_idx = seg[-1]
    start = taxi7[start_idx, :2]
    end = taxi7[end_idx, :2]
    segment_length = end_idx - start_idx + 1
    
    predicted_path_bm = brownian_motion_predict(start, end, segment_length)
    
    # 确保起点和终点不变
    predicted_path_bm[0] = start
    predicted_path_bm[-1] = end
    
    predicted_paths_bm.append(predicted_path_bm)

predicted_paths_bm = np.concatenate(predicted_paths_bm, axis=0)

# 确保预测路径长度与真实路径长度匹配
predicted_paths_bm = predicted_paths_bm[:len(taxi7)]

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

aee_per_step_bm = calculate_aee_per_step(taxi7, predicted_paths_bm, taxi7_segments)
print(f"AEE per step (Brownian Motion): {aee_per_step_bm}")

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
predicted_segment_lengths_bm = calculate_segment_lengths(predicted_paths_bm, taxi7_segments)

# Kolmogorov-Smirnov检验
from scipy.stats import ks_2samp
ks_stat_bm, p_value_bm = ks_2samp(true_segment_lengths, predicted_segment_lengths_bm)
print(f"KS Statistic (Brownian Motion): {ks_stat_bm}, P-value: {p_value_bm}")

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

metrics_bm = calculate_metrics(taxi7, predicted_paths_bm)
print(f"MAE (Brownian Motion): {metrics_bm[0]}")
print(f"MSE (Brownian Motion): {metrics_bm[1]}")
print(f"Max AE (Brownian Motion): {metrics_bm[2]}")
print(f"Segment Level MAE (Brownian Motion): {metrics_bm[3]}")
print(f"Segment Level MSE (Brownian Motion): {metrics_bm[4]}")
print(f"R-squared (Brownian Motion): {metrics_bm[5]}")

# 可视化真实路径和预测路径
plt.figure(figsize=(10, 6))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths_bm[:, 1], predicted_paths_bm[:, 0], label='Predicted Path (Brownian Motion)', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()

# 绘制 AEE per step 图
plt.figure(figsize=(10, 6))
steps = np.arange(len(taxi7))
aee_errors_bm = np.sqrt(np.sum((taxi7[:, :2] - predicted_paths_bm) ** 2, axis=1))
plt.plot(steps, aee_errors_bm, label='AEE per step (Brownian Motion)', color='red')
plt.xlabel('Step')
plt.ylabel('AEE')
plt.title('AEE per Step')
plt.legend()
plt.show()

# 绘制段长度分布图并加上 K-S 检验结果
plt.figure(figsize=(10, 6))
plt.hist(true_segment_lengths, bins=30, alpha=0.5, label='True Segment Lengths')
plt.hist(predicted_segment_lengths_bm, bins=30, alpha=0.5, label='Predicted Segment Lengths (Brownian Motion)')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend()

# 在图中添加 K-S 检验结果
plt.text(0.95, 0.95, f'KS Statistic (BM): {ks_stat_bm:.2f}, P-value (BM): {p_value_bm:.2e}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
