# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:10:38 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

# Segment paths function
def segment_paths(data):
    segments = []
    current_segment = []
    current_value = data[0, 2]
    for idx, row in enumerate(data):
        if row[2] == current_value:
            current_segment.append(row)
        else:
            segments.append(np.array(current_segment))
            current_segment = [row]
            current_value = row[2]
    segments.append(np.array(current_segment))
    return segments

# 准备CNN训练数据
def prepare_data_for_cnn(data, scaler):
    features = []
    labels = []
    for segment in data:
        if len(segment.shape) == 1:  # Ensure segment is 2D
            segment = segment.reshape(1, -1)
        if segment.shape[0] < 2:
            continue
        start = segment[0, :2]
        end = segment[-1, :2]
        passenger_status = segment[0, 2]
        segment_data = segment[:, :2]
        segment_scaled = scaler.transform(segment_data)
        
        features.append(np.concatenate([start, end, [passenger_status]]))
        labels.append(segment_scaled)
    
    features = np.array(features)
    labels = [np.array(label, dtype=np.float32) for label in labels]  # Ensure labels are float arrays
    return features, labels

# 定义CNN模型
def create_cnn_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(np.prod(output_shape), activation='linear'),
        tf.keras.layers.Reshape(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 预测函数
def predict_trajectory(model, scaler, data, max_segment_length):
    predictions = []
    for segment in segment_paths(data):
        if len(segment.shape) == 1:  # Ensure segment is 2D
            segment = segment.reshape(1, -1)
        if segment.shape[0] < 2:
            continue
        start = segment[0, :2]
        end = segment[-1, :2]
        passenger_status = segment[0, 2]
        
        features = np.concatenate([start, end, [passenger_status]]).reshape(1, -1)
        pred = model.predict(features)
        pred_rescaled = scaler.inverse_transform(pred.reshape(-1, 2)[:len(segment)])  # Ensure correct shape after prediction
        
        # 强制起点和终点一致
        pred_rescaled[0] = start
        pred_rescaled[-1] = end

        predictions.append(pred_rescaled)
    return np.concatenate(predictions, axis=0)

# 训练和预测函数
def train_and_predict(train_data, test_data, scaler, epochs=20):
    # 分离有乘客和无乘客状态的数据
    train_with_passenger = [segment for segment in segment_paths(train_data) if segment[0, 2] == 1]
    train_without_passenger = [segment for segment in segment_paths(train_data) if segment[0, 2] == 0]

    print(f"Number of segments with passengers: {len(train_with_passenger)}")
    print(f"Number of segments without passengers: {len(train_without_passenger)}")
    
    if len(train_with_passenger) == 0 or len(train_without_passenger) == 0:
        print("Not enough segments for training.")
        return None

    # 准备训练数据
    X_with, y_with = prepare_data_for_cnn(train_with_passenger, scaler)
    X_without, y_without = prepare_data_for_cnn(train_without_passenger, scaler)

    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with, y_with, test_size=0.2, shuffle=True)
    X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without, y_without, test_size=0.2, shuffle=True)

    input_shape = X_train_with.shape[1]
    max_segment_length_with = max(max(len(label) for label in y_train_with), max(len(label) for label in y_test_with))
    max_segment_length_without = max(max(len(label) for label in y_train_without), max(len(label) for label in y_test_without))

    output_shape_with = (max_segment_length_with, 2)
    output_shape_without = (max_segment_length_without, 2)

    y_train_with_padded = np.array([np.pad(label, ((0, max_segment_length_with - len(label)), (0, 0)), 'constant') for label in y_train_with])
    y_test_with_padded = np.array([np.pad(label, ((0, max_segment_length_with - len(label)), (0, 0)), 'constant') for label in y_test_with])
    
    y_train_without_padded = np.array([np.pad(label, ((0, max_segment_length_without - len(label)), (0, 0)), 'constant') for label in y_train_without])
    y_test_without_padded = np.array([np.pad(label, ((0, max_segment_length_without - len(label)), (0, 0)), 'constant') for label in y_test_without])

    # 创建和训练模型
    model_with = create_cnn_model(input_shape, output_shape_with)
    model_without = create_cnn_model(input_shape, output_shape_without)

    model_with.fit(X_train_with, y_train_with_padded, epochs=epochs, batch_size=32, validation_data=(X_test_with, y_test_with_padded))
    model_without.fit(X_train_without, y_train_without_padded, epochs=epochs, batch_size=32, validation_data=(X_test_without, y_test_without_padded))

    # 预测
    test_with_passenger = [segment for segment in segment_paths(test_data) if segment[0, 2] == 1]
    test_without_passenger = [segment for segment in segment_paths(test_data) if segment[0, 2] == 0]

    pred_with = predict_trajectory(model_with, scaler, np.concatenate(test_with_passenger), max_segment_length_with)
    pred_without = predict_trajectory(model_without, scaler, np.concatenate(test_without_passenger), max_segment_length_without)

    return np.concatenate([pred_with, pred_without], axis=0)

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

# 归一化数据
all_data = np.concatenate(taxi_data)
scaler = MinMaxScaler()
scaler.fit(all_data[:, :2])

# 训练和预测
predicted_path = train_and_predict(np.concatenate(taxi_data), taxi7, scaler, epochs=20)

if predicted_path is not None:
    # 可视化真实路径和预测路径
    plt.figure(figsize=(12, 8))
    plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
    plt.plot(predicted_path[:, 1], predicted_path[:, 0], label='Predicted Path', color='red', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True Path vs Predicted Path')
    plt.legend()
    plt.show()
else:
    print("No predictions were made due to lack of segments.")
