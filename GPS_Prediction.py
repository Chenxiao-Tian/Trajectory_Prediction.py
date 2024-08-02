# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:01:39 2024

@author: ct347
"""

import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
latitude_start = 37.7749  # 旧金山的纬度
longitude_start = -122.4194  # 旧金山的经度
x0 = np.array([latitude_start, longitude_start])  # 起点
T = 1.0  # 总时间
N = 1000  # 时间步数
dt = T / N

# 时间序列
t = np.linspace(0, T, N+1)

# 初始化布朗运动路径
true_path = np.zeros((N+1, 2))
true_path[0] = x0

# 生成布朗运动路径
for i in range(1, N+1):
    true_path[i, 0] = true_path[i-1, 0] + np.sqrt(dt) * np.random.normal() * 0.0001
    true_path[i, 1] = true_path[i-1, 1] + np.sqrt(dt) * np.random.normal() * 0.0001

# 终点位置
xT = true_path[-1]

# 计算增量
increments = np.diff(true_path, axis=0)

# 估计sigma
sigma_estimated = np.std(increments, axis=0) / np.sqrt(dt)
print(f'Estimated sigma: {sigma_estimated}')

# 估计mu
mu_estimated = np.mean(increments, axis=0) / dt
print(f'Estimated mu: {mu_estimated}')

# 绘制真实路径
plt.plot(true_path[:, 1], true_path[:, 0], label='True Path (Brownian Motion)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path of Brownian Motion in San Francisco')
plt.legend()
plt.show()

# 初始化布朗桥预测路径数组
predicted_paths_bridge = np.zeros((100, N+1, 2))

# 使用估计的参数重新模拟路径
for j in range(100):
    predicted_paths_bridge[j, 0] = x0  # 起点
    predicted_paths_bridge[j, -1] = xT  # 终点
    for i in range(1, N):
        t_i = t[i]
        drift = (xT - predicted_paths_bridge[j, i-1]) / (T - t_i)
        diffusion = sigma_estimated * np.sqrt(dt) * np.random.normal(size=2)
        predicted_paths_bridge[j, i] = predicted_paths_bridge[j, i-1] + drift * dt + diffusion

# 初始化布朗运动预测路径数组
predicted_paths_brownian = np.zeros((100, N+1, 2))

# 使用布朗运动重新模拟路径
for j in range(100):
    predicted_paths_brownian[j, 0] = x0  # 起点
    for i in range(1, N+1):
        predicted_paths_brownian[j, i] = predicted_paths_brownian[j, i-1] + sigma_estimated * np.sqrt(dt) * np.random.normal(size=2)

# 绘制布朗桥和布朗运动模拟路径
plt.figure(figsize=(10, 6))
for j in range(100):
    plt.plot(predicted_paths_bridge[j, :, 1], predicted_paths_bridge[j, :, 0], color='red', alpha=0.1)
    plt.plot(predicted_paths_brownian[j, :, 1], predicted_paths_brownian[j, :, 0], color='green', alpha=0.1)
plt.plot(true_path[:, 1], true_path[:, 0], label='True Path', color='blue', linewidth=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('100 Simulated Paths using Brownian Bridge and Brownian Motion with Estimated Parameters')
plt.legend(['Brownian Bridge', 'Brownian Motion', 'True Path'])
plt.show()

# 计算每一步的AEE误差
def calculate_aee_per_step(true_path, predicted_paths):
    num_steps = true_path.shape[0]
    num_paths = predicted_paths.shape[0]
    aee_per_step = np.zeros(num_steps)
    for i in range(num_steps):
        errors = np.sqrt((true_path[i, 0] - predicted_paths[:, i, 0])**2 + (true_path[i, 1] - predicted_paths[:, i, 1])**2)
        aee_per_step[i] = np.mean(errors)
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
plt.title('AEE per Step for Brownian Bridge and Brownian Motion with Estimated Parameters')
plt.legend()
plt.show()
