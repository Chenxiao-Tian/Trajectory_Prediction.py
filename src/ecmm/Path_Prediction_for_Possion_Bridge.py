# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:04:36 2024

@author: ct347
"""

import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
x0 = 0   # 起点
T = 1.0  # 总时间
N = 1000 # 时间步数
dt = T / N
lambda_rate= 4
# 时间序列
t = np.linspace(0, T, N+1)

# 初始化布朗运动路径
true_path = np.zeros(N+1)
true_path[0] = x0

# 生成布朗运动路径
for i in range(1, N+1):
    true_path[i] = true_path[i-1] + np.sqrt(dt) * np.random.normal()

# 终点位置
xT = true_path[-1]

# 计算增量
increments = np.diff(true_path)

# 估计sigma
sigma_estimated = np.std(increments) / np.sqrt(dt)
print(f'Estimated sigma: {sigma_estimated}')

# 估计mu
mu_estimated = np.mean(increments) / dt
print(f'Estimated mu: {mu_estimated}')

# 绘制真实路径
plt.plot(t, true_path, label='True Path (Brownian Motion)')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('True Path of Brownian Motion')
plt.legend()
plt.show()

# 初始化布朗桥预测路径数组
predicted_paths_bridge = np.zeros((1000, N+1))

# 使用估计的参数重新模拟路径
for j in range(1000):
    predicted_paths_bridge[j, 0] = true_path[0]  # 起点
    predicted_paths_bridge[j, -1] = true_path[-1]  # 终点
    for i in range(1, N):
        t_i = t[i]
        drift = mu_estimated - (predicted_paths_bridge[j, i-1] - (T-t_i)/T * true_path[-1]) / (T-t_i)
        diffusion = sigma_estimated * np.sqrt(dt) * np.random.normal()
        predicted_paths_bridge[j, i] = predicted_paths_bridge[j, i-1] + drift * dt + diffusion

# 初始化布朗运动预测路径数组
predicted_paths_brownian = np.zeros((1000, N+1))

# 使用布朗运动重新模拟路径
for j in range(1000):
    predicted_paths_brownian[j, 0] = true_path[0]  # 起点
    for i in range(1, N+1):
        predicted_paths_brownian[j, i] = predicted_paths_brownian[j, i-1] + sigma_estimated * np.sqrt(dt) * np.random.normal()

# 初始化泊松过程路径
true_path_poisson = np.zeros(N+1)
true_path_poisson[0] = x0

# 生成泊松过程路径
for i in range(1, N+1):
    true_path_poisson[i] = true_path_poisson[i-1] + np.random.poisson(lambda_rate * dt)

# 估计lambda
lambda_estimated = np.mean(np.diff(true_path_poisson)) / dt
print(f'Estimated lambda: {lambda_estimated}')

# 初始化泊松过程预测路径数组
predicted_paths_poisson = np.zeros((1000, N+1))

# 使用估计的参数重新模拟路径
for j in range(1000):
    predicted_paths_poisson[j, 0] = true_path_poisson[0]  # 起点
    for i in range(1, N+1):
        predicted_paths_poisson[j, i] = predicted_paths_poisson[j, i-1] + np.random.poisson(lambda_estimated * dt)

# 绘制布朗桥和泊松过程模拟路径
plt.figure(figsize=(10, 6))
for j in range(1000):
    plt.plot(t, predicted_paths_bridge[j], color='red', alpha=0.1)
    plt.plot(t, predicted_paths_poisson[j], color='purple', alpha=0.1)
plt.plot(t, true_path, label='True Path (Brownian Bridge)', color='blue', linewidth=2)
plt.plot(t, true_path_poisson, label='True Path (Poisson Process)', color='green', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('1000 Simulated Paths using Brownian Bridge and Poisson Process with Estimated Parameters')
plt.legend(['Brownian Bridge', 'Poisson Process', 'True Path (Brownian Bridge)', 'True Path (Poisson Process)'])
plt.show()

# 计算每一步的AEE误差
def calculate_aee_per_step(true_path, predicted_paths):
    num_steps = true_path.shape[0]
    num_paths = predicted_paths.shape[0]
    aee_per_step = np.zeros(num_steps)
    for i in range(num_steps):
        errors = np.abs(true_path[i] - predicted_paths[:, i])
        aee_per_step[i] = np.mean(errors)
    return aee_per_step

# 计算布朗桥的AEE误差
aee_bridge_per_step = calculate_aee_per_step(true_path, predicted_paths_bridge)

# 计算泊松过程的AEE误差
aee_poisson_per_step = calculate_aee_per_step(true_path_poisson, predicted_paths_poisson)

# 绘制AEE误差图
plt.figure(figsize=(10, 6))
plt.plot(t, aee_bridge_per_step, label='AEE (Poisson Bridge)', color='red')
plt.plot(t, aee_poisson_per_step, label='AEE (Poisson Process)', color='purple')
plt.xlabel('Time')
plt.ylabel('AEE')
plt.title('AEE per Step for Brownian Bridge and Poisson Process with Estimated Parameters')
plt.legend()
plt.show()
