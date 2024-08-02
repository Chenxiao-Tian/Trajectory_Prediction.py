# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:09:05 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

# Segment paths function
def segment_paths(data):
    data = np.array(data)
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

# Split data by passenger status
def split_by_passenger_status(data):
    data = np.array(data)
    with_passenger = []
    without_passenger = []
    for segment in segment_paths(data):
        if data[segment[0], 2] == 1:
            with_passenger.append(data[segment])
        else:
            without_passenger.append(data[segment])
    return with_passenger, without_passenger

# Plot trajectories
def plot_trajectories(map_graph, trajectories, color, label):
    for traj in trajectories:
        plt.plot(traj[:, 1], traj[:, 0], color=color, alpha=0.6)
    plt.title(label)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

# Prepare data for CNN
def prepare_data_for_cnn(data, scaler, max_length):
    features = []
    labels = []
    for segment in data:
        segment_data = segment[:, :2]
        segment_scaled = scaler.transform(segment_data)
        
        # Pad sequences to max_length
        if len(segment_scaled) < max_length:
            padding = np.zeros((max_length - len(segment_scaled), 2))
            segment_scaled = np.vstack((segment_scaled, padding))
        else:
            segment_scaled = segment_scaled[:max_length]
        
        features.append(segment_scaled)
        labels.append(segment_scaled)
    
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Define CNN model
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(50, 4, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation='linear'),
        tf.keras.layers.Reshape(input_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Predict function
def predict_trajectory(model, scaler, data, max_length):
    predictions = []
    for segment in segment_paths(data):
        segment_data = segment[:, :2]
        segment_scaled = scaler.transform(segment_data)
        
        # Pad sequences to max_length
        if len(segment_scaled) < max_length:
            padding = np.zeros((max_length - len(segment_scaled), 2))
            segment_scaled = np.vstack((segment_scaled, padding))
        else:
            segment_scaled = segment_scaled[:max_length]

        segment_scaled = segment_scaled.reshape(1, max_length, 2)
        pred = model.predict(segment_scaled)
        pred_rescaled = scaler.inverse_transform(pred[0])
        predictions.append(pred_rescaled[:len(segment)])
    return np.concatenate(predictions, axis=0)

# Train and predict
def train_and_predict(data_with_passenger, data_without_passenger, scaler, max_length, epochs=20):
    X_with, y_with = prepare_data_for_cnn(data_with_passenger, scaler, max_length)
    X_without, y_without = prepare_data_for_cnn(data_without_passenger, scaler, max_length)

    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with, y_with, test_size=0.2, shuffle=True)
    X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without, y_without, test_size=0.2, shuffle=True)

    model_with = create_cnn_model((max_length, 2))
    model_without = create_cnn_model((max_length, 2))

    model_with.fit(X_train_with, y_train_with, epochs=epochs, batch_size=32, validation_data=(X_test_with, y_test_with))
    model_without.fit(X_train_without, y_train_without, epochs=epochs, batch_size=32, validation_data=(X_test_without, y_test_without))

    pred_with = predict_trajectory(model_with, scaler, taxi7, max_length)
    pred_without = predict_trajectory(model_without, scaler, taxi7, max_length)

    return pred_with, pred_without

# Load taxi data
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

# Split by passenger status
taxi_data_with_passenger = []
taxi_data_without_passenger = []

for data in taxi_data:
    with_passenger, without_passenger = split_by_passenger_status(data)
    taxi_data_with_passenger.extend(with_passenger)
    taxi_data_without_passenger.extend(without_passenger)

# Plot trajectories
G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

plt.figure(figsize=(12, 8))
ox.plot_graph(G, show=False, close=False)
plot_trajectories(G, taxi_data_with_passenger, 'blue', 'With Passenger')
plt.show()

plt.figure(figsize=(12, 8))
ox.plot_graph(G, show=False, close=False)
plot_trajectories(G, taxi_data_without_passenger, 'red', 'Without Passenger')
plt.show()

# Normalize data
all_data = np.concatenate(taxi_data)
scaler = MinMaxScaler()
scaler.fit(all_data[:, :2])

# Find the maximum segment length
max_length = max(len(segment) for segment in taxi_data_with_passenger + taxi_data_without_passenger)

# Train and predict
pred_with, pred_without = train_and_predict(taxi_data_with_passenger, taxi_data_without_passenger, scaler, max_length, epochs=20)

# 拼接预测结果
predicted_paths = np.concatenate([pred_with, pred_without], axis=0)

# 可视化总的预测路径
plt.figure(figsize=(12, 8))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_paths[:, 1], predicted_paths[:, 0], label='Predicted Path', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()
