# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:10:14 2024

@author: 92585
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:29:12 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Zero', 'Timestamp'])
data = df[['Latitude', 'Longitude']].values

# Parameters
T = 1
N = len(data) - 1
dt = T / N
t = np.linspace(0, T, N)

# Calculate increments and estimate parameters
increments = np.diff(data, axis=0)
sigma_estimated = np.std(increments, axis=0) / np.sqrt(dt)
mu_estimated = np.mean(increments, axis=0) / dt

# Define segment points (e.g., every 50 points is a segment point)
segment_points = np.arange(0, N, 50)
if segment_points[-1] != N:
    segment_points = np.append(segment_points, N)  # Ensure the last point is included

# Simulation storage
predicted_paths_bridge = np.zeros((100, N+1, 2))
predicted_paths_brownian = np.zeros((100, N+1, 2))

# Segment-based simulation
for j in range(100):
    for seg in range(len(segment_points)-1):
        start_idx = segment_points[seg]
        end_idx = segment_points[seg + 1]
        x0 = data[start_idx]
        xT = data[end_idx]

        # Simulate each segment using Brownian Bridge
        for i in range(start_idx, end_idx + 1):
            if i == start_idx:
                predicted_paths_bridge[j, i] = x0
            else:
                elapsed_time = (i - start_idx) * dt
                remaining_time = (end_idx - i) * dt
                drift = mu_estimated + (xT - predicted_paths_bridge[j, i-1]) / remaining_time
                diffusion = sigma_estimated * np.sqrt(dt) * np.random.normal(size=2)
                predicted_paths_bridge[j, i] = predicted_paths_bridge[j, i-1] + drift * dt + diffusion

# Plot results
plt.figure(figsize=(10, 6))
for j in range(1):  # Only plot one simulation for clarity
    plt.plot(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0], 'r-', alpha=0.1)
plt.plot(data[:, 1], data[:, 0], 'b-', linewidth=2, label='True Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Segmented Brownian Bridge Simulation')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:51:35 2024

@author: ct347
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # 初始化参数
# T = 100 # 总时间
# N =25094 # 时间步数
# dt = T / N

# # 时间序列
# t = np.linspace(0, T, N+1)

# # 读取txt文件
# file_path = 'taxi2.txt'
# df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Zero', 'Timestamp'])

# # 提取经纬度数据
# data = df[['Latitude', 'Longitude']].values
# true_path = data

# # 终点位置
# x0 = true_path[0]
# xT = true_path[-1]
# 初始化参数
T = 1  # 总时间
N =23494 # 时间步数
dt = T / N

# 时间序列
t = np.linspace(0, T, N+1)

# 读取txt文件
file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Zero', 'Timestamp'])

# 提取经纬度数据
data = df[['Latitude', 'Longitude']].values
true_path = data

# 终点位置
x0 = true_path[0]
xT = true_path[-1]
# 计算增量
increments = np.diff(true_path, axis=0)

# 估计sigma
sigma_estimated = np.std(increments, axis=0) / np.sqrt(dt)
print(f'Estimated sigma: {sigma_estimated}')
# 估计mu
mu_estimated = np.mean(increments, axis=0) / dt
print(f'Estimated mu: {mu_estimated}')
# 初始化参数
T = 1  # 总时间
N =25094 # 时间步数
dt = T / N

# 时间序列
t = np.linspace(0, T, N+1)

# 读取txt文件
file_path = 'c:/Users/92585/PROJ-2023-ECMM/src/taxi2.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Zero', 'Timestamp'])

# 提取经纬度数据
data = df[['Latitude', 'Longitude']].values
true_path = data

# 终点位置
x0 = true_path[0]
xT = true_path[-1]
# 绘制真实路径
plt.plot(true_path[:, 1], true_path[:, 0], label='True Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path of Object in San Francisco')
plt.legend()
plt.show()

# # 初始化布朗桥预测路径数组
# predicted_paths_bridge = np.zeros((100, N+1, 2))

# # 使用估计的参数重新模拟路径
# for j in range(100):
#     predicted_paths_bridge[j, 0] = x0  # 起点
#     predicted_paths_bridge[j, -1] = xT  # 终点
#     for i in range(1, N+1):
#         t_i = t[i]
#         drift = mu_estimated + (xT - predicted_paths_bridge[j, i-1]) * (1 / (T - t_i))
#         diffusion = sigma_estimated * np.sqrt(dt) * np.random.normal(size=2)
#         predicted_paths_bridge[j, i] = predicted_paths_bridge[j, i-1] + drift * dt + diffusion

# 初始化布朗运动预测路径数组
predicted_paths_brownian = np.zeros((100, N+1, 2))

# 使用布朗运动重新模拟路径
for j in range(100):
    predicted_paths_brownian[j, 0] = x0  # 起点
    for i in range(1, N+1):
        predicted_paths_brownian[j, i] = predicted_paths_brownian[j, i-1] +mu_estimated*dt+ sigma_estimated * np.sqrt(dt) * np.random.normal(size=2)

# 绘制布朗桥和布朗运动模拟路径
plt.figure(figsize=(10, 6))
for j in range(1):
    plt.plot(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0], color='red', alpha=0.1)
    plt.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)
plt.plot(true_path[:, 1], true_path[:, 0], label='True Path', color='blue', linewidth=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Path Prediction using Segment Brownian Bridge vs Brownian Motion with Estimated Parameters')
plt.legend(['Segment Brownian Bridge', 'Brownian Motion', 'True Path'])
plt.show()
# 计算每一步的AEE误差
def calculate_aee_per_step(true_path, predicted_paths):
    num_steps = true_path.shape[0]  # 真实路径的时间步数
    num_paths = predicted_paths.shape[0]  # 预测路径的数量
    aee_per_step = np.zeros(num_steps)  # 初始化AEE数组
    for i in range(num_steps):  # 对每个时间步进行循环
        errors = np.sqrt((true_path[i, 0] - predicted_paths[:, i, 0])**2 + (true_path[i, 1] - predicted_paths[:, i, 1])**2)
        aee_per_step[i] = np.mean(errors)  # 计算该时间步的平均误差
    return aee_per_step

# 计算两种方法的AEE误差
aee_bridge_per_step = calculate_aee_per_step(true_path, predicted_paths_bridge)
aee_brownian_per_step = calculate_aee_per_step(true_path, predicted_paths_brownian)

# 绘制AEE误差图
plt.figure(figsize=(10, 6))
plt.plot(t, aee_bridge_per_step, label='AEE (Brownian Bridge)', color='red')
plt.plot(t, aee_brownian_per_step, label='AEE (Brownian Motion)', color='green')
plt.xlabel('Time')
plt.ylabel('AEE')
plt.title('AEE per Step for Brownian Bridge and Brownian Motion')
plt.legend()
plt.show()
